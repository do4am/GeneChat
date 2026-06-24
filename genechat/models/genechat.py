import logging
import esm

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from argparse import ArgumentParser
import json

from genechat.common.registry import registry
from genechat.models.blip2 import Blip2Base, disabled_train
from genechat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

#Import the Gene Encoder Libraries
from genechat.models.gene_encoder import HyenaDNAPreTrainedModel, CharacterTokenizer

#Transformer Modules
from transformers import AutoTokenizer, EsmModel, AutoModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import time
from typing import List

_STOP_CODONS = {'TAA', 'TAG', 'TGA'}

def _reverse_complement(seq):
    comp = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(comp)[::-1]

def find_longest_orf(seq, min_nt=300):
    """
    Find the longest open reading frame (ATG ... stop codon) in the sequence.
    Scans all 6 reading frames (3 forward + 3 reverse complement).
    Returns the ORF sequence if >= min_nt bp, otherwise returns the original sequence.

    For mRNA: the longest ORF is almost always the true CDS.
    For genomic DNA (fallback): checks both strands to handle minus-strand genes.
    For ENCODE regulatory elements (no protein product): no long ORF → returns full seq.
    """
    seq_upper = seq.upper()
    best_orf = ""

    for strand_seq in (seq_upper, _reverse_complement(seq_upper)):
        for frame in range(3):
            i = frame
            start = None
            while i + 3 <= len(strand_seq):
                codon = strand_seq[i:i+3]
                if codon == 'ATG' and start is None:
                    start = i
                elif codon in _STOP_CODONS and start is not None:
                    orf = strand_seq[start:i+3]
                    if len(orf) > len(best_orf):
                        best_orf = orf
                    start = None
                i += 3

    if len(best_orf) >= min_nt:
        return best_orf
    return seq  # fallback: no long ORF found (non-coding / regulatory)


@registry.register_model("genechat")
class GeneChat(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "",
    }

    def __init__(
        self,
        freeze_gene_encoder=True,
        gene_model="",
        max_gene_length=5000,
        freeze_adaptor=False,
        freeze_llama=True,
        llama_model="",
        embedding_agg=1,
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        keyword_loss_alpha=3.0,  # upweight gene-name / biology keyword tokens in loss
        adaptor_type="linear",   # "linear" (default) or "mlp" (two-layer with GELU)
        encoder_type="dnabert2", # "dnabert2" (default) or "esm2" (protein sequence encoder)
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.embedding_agg = embedding_agg
        
        ######################################################################################################  <MAJOR CHANGE>
        '''
        ######################################################################################################  HYENADNA
        print('\n\n---->Loading Gene Encoder - HyenaDNA...')
        # we need these for the decoder head, if using
        use_head = False

        # you can override with your own backbone config here if you want,
        # otherwise we'll load the HF one in None
        backbone_cfg = None

        # Mode - Pool, Sum, Last
        mode = 'pool'

        # Max Length
        self.max_gene_length = max_gene_length

        # Gene Encoder - Hyena DNA Model
        self.gene_encoder = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            gene_model,
            download=True,
            config=backbone_cfg,
            device=torch.cuda.current_device(),
            use_head=use_head,
            mode=mode
        )

        # Gene Tokenizer
        self.gene_tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  
            model_max_length=self.max_gene_length + 2,  
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )

        # Pooling layer to pool the output of HyenaDNA
        self.gene_pool_width = gene_pool_width
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=gene_pool_width, stride=gene_pool_width)
        '''
    
        self.max_gene_length = max_gene_length
        self.encoder_type = encoder_type

        if encoder_type == "esm2":
            ######################################################################################################  ESM2
            print('\n\n---->Loading Gene Encoder - ESM-2 (650M, protein sequences)...')
            from transformers import EsmModel, EsmTokenizer
            self.gene_encoder = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.gene_encoder = self.gene_encoder.to(torch.cuda.current_device())
            self.gene_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            _encoder_lora_targets = ["query", "key", "value", "dense"]
        elif encoder_type == "nt_v2":
            ######################################################################################################  NT-v2
            print('\n\n---->Loading Gene Encoder - Nucleotide Transformer v2 (500M)...')
            from transformers import AutoModelForMaskedLM
            # NT-v2 only registers AutoModelForMaskedLM (no plain AutoModel).
            # Load with trust_remote_code so the custom FFN (intermediate=8192) is used,
            # then extract .esm (the base transformer encoder) and discard the LM head.
            _ntv2_id = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
            _mlm = AutoModelForMaskedLM.from_pretrained(_ntv2_id, trust_remote_code=True)
            self.gene_encoder = _mlm.esm   # EsmModel — outputs last_hidden_state
            del _mlm                        # free LM head weights
            import gc; gc.collect(); torch.cuda.empty_cache()
            self.gene_encoder = self.gene_encoder.to(torch.cuda.current_device())
            self.gene_tokenizer = AutoTokenizer.from_pretrained(_ntv2_id, trust_remote_code=True)
            _encoder_lora_targets = ["query", "key", "value", "dense"]
        else:
            ######################################################################################################  DNABERT2
            print('\n\n---->Loading Gene Encoder - DNABERT2...')
            self.gene_encoder = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            self.gene_encoder = self.gene_encoder.to(torch.cuda.current_device())
            self.gene_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            _encoder_lora_targets = ["Wqkv", "dense"]

        ######################################################################################################  </MAJOR CHANGE>

        if freeze_gene_encoder:
            for name, param in self.gene_encoder.named_parameters():
                param.requires_grad = False
            self.gene_encoder = self.gene_encoder.eval()
            self.gene_encoder.train = disabled_train
            logging.info("freeze gene encoder")
        else:
            encoder_lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=_encoder_lora_targets,
                lora_dropout=0.05,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.gene_encoder = get_peft_model(self.gene_encoder, encoder_lora_config)
            self.gene_encoder.print_trainable_parameters()
            self.gene_encoder.enable_input_require_grads()
            if encoder_type != "nt_v2":
                self.gene_encoder.gradient_checkpointing_enable()
            logging.info(f"gene encoder LoRA applied (r=16, {_encoder_lora_targets})")
        
        print('\n\n---->Loading LLAMA')

        # LLama Tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        # LLama Model
        if self.low_resource:
            print("Start Low Resource Mode")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto'
                # device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, config).model

            # Freeze LoRA params for all layers, then unfreeze only layers 5-10 and 32-39
            _active_layers = set(range(5, 11)) | set(range(32, 40))
            for name, param in self.llama_model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = False
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            try:
                                if int(parts[i + 1]) in _active_layers:
                                    param.requires_grad = True
                            except ValueError:
                                pass
            self.llama_model.print_trainable_parameters()
            self.llama_model.enable_input_require_grads()
            self.llama_model.gradient_checkpointing_enable()
            self.llama_model.config.use_cache = False

        # Linear layer to align the gene embeddings to the LLama token embedding space

        ######################################################################################################  </MAJOR CHANGE>
        # Detect encoder hidden size
        if encoder_type in ("esm2", "nt_v2"):
            encoder_hidden_size = self.gene_encoder.config.hidden_size  # 1280 (ESM-2) or 1024 (NT-v2)
        else:
            encoder_hidden_size = self.gene_encoder.embeddings.word_embeddings.weight.shape[1]  # 768 (DNABERT-2)
        if adaptor_type == "mlp":
            # Two-layer MLP adaptor: 768 → 2048 → 5120
            # Increases adaptor capacity without touching LLM LoRA weights.
            # Load a linear-adaptor checkpoint with strict=False — the linear weights
            # are ignored and only gene_encoder + llama LoRA weights are reused.
            self.hyena_llama_proj = nn.Sequential(
                nn.Linear(encoder_hidden_size, 2048),
                nn.GELU(),
                nn.Linear(2048, self.llama_model.config.hidden_size),
            )
        else:
            # Default: single linear projection (backward-compatible)
            self.hyena_llama_proj = nn.Linear(
                encoder_hidden_size, self.llama_model.config.hidden_size
            )

        # Learned attention pooling: scores each genomic chunk by informativeness.
        # Replaces naive mean pooling — model learns to upweight exonic chunks and
        # downweight intronic noise purely from the gene description training signal.
        self.chunk_attention = nn.Linear(encoder_hidden_size, 1)

        '''
        ######################################################################################################  HyenaDNA
        self.hyena_llama_proj = nn.Linear(
            self.gene_encoder.backbone.embeddings.word_embeddings.embedding_dim, self.llama_model.config.hidden_size
        )
        '''
        ######################################################################################################  </MAJOR CHANGE>

        if freeze_adaptor:
            for name, param in self.hyena_llama_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.chunk_attention.named_parameters():
                param.requires_grad = False
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        # ── Keyword-weighted loss ──────────────────────────────────────────────
        # Precompute per-vocab-token weight tensor (once at init, ~1 sec).
        # Tokens that look like gene symbols / biology keywords get alpha weight;
        # all other tokens get 1.0. Registered as a non-trainable buffer so it
        # moves to GPU automatically with the model.
        _STOPWORDS = {
            "the", "a", "an", "is", "it", "in", "of", "to", "and", "or",
            "by", "as", "at", "be", "are", "was", "for", "with", "its",
            "this", "that", "has", "have", "not", "from", "on", "but",
            "also", "which", "than", "into", "been", "can", "may", "do",
        }
        import re as _re
        vocab_size = self.llama_tokenizer.vocab_size
        w = torch.ones(vocab_size, dtype=torch.float32)
        for tok_id in range(vocab_size):
            tok_str = self.llama_tokenizer.convert_ids_to_tokens(tok_id) or ""
            # strip SentencePiece leading space (▁) for matching
            tok_clean = tok_str.lstrip("▁").strip()
            if (len(tok_clean) >= 2
                    and tok_clean.lower() not in _STOPWORDS
                    and sum(1 for c in tok_clean if c.isupper()) >= 2):
                w[tok_id] = keyword_loss_alpha
        self.register_buffer("token_loss_weights", w, persistent=False)
        logging.info(
            f"Token upweighting: {(w > 1).sum().item()} / {vocab_size} tokens "
            f"weighted at alpha={keyword_loss_alpha}"
        )
        # ──────────────────────────────────────────────────────────────────────

    def _encode_gene_esm2(self, seqs):
        """ESM-2 encoder path: takes protein amino acid sequences, returns (batch, 1, 1280)."""
        MAX_AA = 1022  # ESM-2 hard limit (model was trained with 1024 - 2 special tokens)
        all_embeds = []

        for seq in seqs:
            seq = seq[:MAX_AA]
            tokens = self.gene_tokenizer(
                seq, return_tensors='pt', max_length=MAX_AA,
                truncation=True, padding=False,
            )
            input_ids = tokens['input_ids'].to(self.gene_encoder.device)
            attention_mask = tokens['attention_mask'].to(self.gene_encoder.device)

            outputs = self.gene_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (1, seq_len, 1280)

            # Mean pool over valid (non-padding) positions, excluding CLS/EOS tokens
            mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
            embed = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, 1280)
            all_embeds.append(embed)

        gene_embeds = torch.stack(all_embeds, dim=0).squeeze(1).unsqueeze(1)  # (batch, 1, 1280)

        _proj_weight = (self.hyena_llama_proj.weight
                        if isinstance(self.hyena_llama_proj, nn.Linear)
                        else next(self.hyena_llama_proj.parameters()))
        if gene_embeds.dtype != _proj_weight.dtype:
            gene_embeds = gene_embeds.to(_proj_weight.dtype)

        inputs_llama = self.hyena_llama_proj(gene_embeds).to(gene_embeds.device)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(gene_embeds.device)
        return inputs_llama, atts_llama

    def _encode_gene_ntv2(self, seqs):
        """NT-v2 encoder path: takes DNA/mRNA sequences, returns (batch, 1, 1024).
        NT-v2 uses 6-mer tokenization — 5000bp fits in ~833 tokens (well under 2048 limit),
        so no chunking needed unlike DNABERT-2.
        """
        all_embeds = []
        for seq in seqs:
            seq = seq[:self.max_gene_length]
            seq = find_longest_orf(seq)

            tokens = self.gene_tokenizer(
                seq, return_tensors="pt", max_length=2048,
                truncation=True, padding=True, add_special_tokens=True,
            )
            input_ids      = tokens["input_ids"].to(self.gene_encoder.device)
            attention_mask = tokens["attention_mask"].to(self.gene_encoder.device)

            outputs = self.gene_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (1, seq_len, 1024)

            # Mean pool over non-padding tokens
            mask  = attention_mask.unsqueeze(-1).float()
            embed = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, 1024)
            all_embeds.append(embed)

        gene_embeds = torch.stack(all_embeds, dim=0).squeeze(1).unsqueeze(1)  # (batch, 1, 1024)

        _proj_weight = (self.hyena_llama_proj.weight
                        if isinstance(self.hyena_llama_proj, nn.Linear)
                        else next(self.hyena_llama_proj.parameters()))
        if gene_embeds.dtype != _proj_weight.dtype:
            gene_embeds = gene_embeds.to(_proj_weight.dtype)

        inputs_llama = self.hyena_llama_proj(gene_embeds).to(gene_embeds.device)
        atts_llama   = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(gene_embeds.device)
        return inputs_llama, atts_llama

    def encode_gene(self, seqs):
        '''
        Encode the input gene/protein sequence.
        Parameters:
            seqs            - Batch of sequences (mRNA for dnabert2, protein AA for esm2)
        Output:
            inputs_llama    - Encoded embedding projected to LLaMA token embedding space
            atts_llama      - Attention masks
        '''

        if self.encoder_type == "esm2":
            return self._encode_gene_esm2(seqs)
        if self.encoder_type == "nt_v2":
            return self._encode_gene_ntv2(seqs)

        '''
        ######################################################################################################  HyenaDNA
        batch_seqs = []
        for seq in seqs:
            s = seq[0] if isinstance(seq, list) else seq
            s = find_longest_orf(s)   # extract CDS if present, else keep full seq
            s = s[:self.max_gene_length]
            batch_seqs.append(s)

        batch_tokenizer_output = self.gene_tokenizer(
                        batch_seqs,
                        padding="longest",     # pad only to longest in batch, not max_gene_length
                        truncation=True,
                        max_length=self.max_gene_length,
                        return_tensors="pt"
                    )
        
        batch_tokens = batch_tokenizer_output["input_ids"]
        batch_tokens = batch_tokens.to(torch.cuda.current_device())

        # Extract the gene embeddings
        gene_embeds = self.gene_encoder(batch_tokens).to(batch_tokens.device)#self.gene_encoder(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33].to(batch_tokens.device)

        #Pooling the gene embeddings 
        #Output of the gene encoder is 160K. Pooling them in fixed intervals to get a smaller sequence of embeddings
        gene_embeddings_permute = gene_embeds.permute(0, 2, 1)
        gene_embeddings_permute = self.avg_pool(gene_embeddings_permute)
        gene_embeds = gene_embeddings_permute.permute(0, 2, 1)

        # input llama is of shape [B, len, 5120]
        if gene_embeds.dtype != self.hyena_llama_proj.weight.dtype:
            gene_embeds = gene_embeds.to(self.hyena_llama_proj.weight.dtype)

        #Alignment of gene embeddings to the LLama token embedding space
        inputs_llama = self.hyena_llama_proj(gene_embeds.squeeze(dim=2)).to(gene_embeds.device)

        # atts_llama is of shape [B, len]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(gene_embeds.device)
        
        #print(f'Size of inputs_llama: {inputs_llama.size()}')
        #print(f'Size of atts_llama: {atts_llama.size()}')

        return inputs_llama, atts_llama

        '''
        
        ######################################################################################################  DNABERT2
        '''
        Encode the input gene sequence using the DNABERT2
        Parameters:
            seqs            - Batch of gene sequences
        Output:
            inputs_llama    - Encoded gene embedding which is projected to the LLama Embedding space
            atts_llam       - Attention Masks of the input tokens
        '''

        # Encode each sequence in the batch using DNABERT-2 with learned attention pooling.
        # For each sequence: split into 512-char overlapping chunks → encode each chunk →
        # learned attention scores collapse all chunks into ONE gene embedding.
        # This replaces naive mean pooling, allowing the model to learn which genomic
        # regions (exons) carry functional information and ignore intronic noise.

        CHUNK_CHARS   = 2000  # ~400-500 BPE tokens (4 chars/token avg), truncation=True as safety net
        OVERLAP_CHARS = 10    # character overlap between consecutive chunks

        all_gene_embeds = []

        for seq in seqs:
            seq = seq[:self.max_gene_length]
            seq = find_longest_orf(seq)  # extract CDS if present, else keep full seq

            chunk_embeds = []
            for i in range(0, len(seq), CHUNK_CHARS):
                chunk = seq[max(0, i - OVERLAP_CHARS):i + CHUNK_CHARS]
                input_token = self.gene_tokenizer(
                    chunk, return_tensors='pt', max_length=512, truncation=True
                )["input_ids"].to(self.gene_encoder.device)

                hidden_states = self.gene_encoder(input_token)[0]  # (1, tok_len, 768)
                chunk_embed = torch.mean(hidden_states, dim=1)     # (1, 768)
                chunk_embeds.append(chunk_embed)

            # chunk_embeds: list of (1, 768) → stack to (num_chunks, 768)
            chunk_embeds = torch.cat(chunk_embeds, dim=0)  # (num_chunks, 768)

            # Learned attention pooling: score each chunk, softmax, weighted sum
            scores  = self.chunk_attention(chunk_embeds)           # (num_chunks, 1)
            weights = torch.softmax(scores, dim=0)                 # (num_chunks, 1)
            gene_embed = (chunk_embeds * weights).sum(dim=0)       # (768,)

            all_gene_embeds.append(gene_embed)

        # Stack batch → (batch, 768) → unsqueeze → (batch, 1, 768)
        gene_embeds = torch.stack(all_gene_embeds, dim=0).unsqueeze(1)

        # Cast to match projection weight dtype (works for both nn.Linear and nn.Sequential)
        _proj_weight = (self.hyena_llama_proj.weight
                        if isinstance(self.hyena_llama_proj, nn.Linear)
                        else next(self.hyena_llama_proj.parameters()))
        if gene_embeds.dtype != _proj_weight.dtype:
            gene_embeds = gene_embeds.to(_proj_weight.dtype)

        # Project to LLaMA embedding space: (batch, 1, 768) → (batch, 1, 5120)
        inputs_llama = self.hyena_llama_proj(gene_embeds).to(gene_embeds.device)

        # Attention mask: (batch, 1) — single gene token per sequence
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(gene_embeds.device)

        return inputs_llama, atts_llama
        

    def prompt_list_wrap(self, img_embeds, atts_img, prompt):
        '''
        Wrap the gene embeddings with the pre-gene sequence and post-gene sequence
        Input to the LLama - pre-gene + gene-sequence + post-gene
        pre-gene:   'Given the gene sequence'
        post-gene:  'Question'
        
        Parameters:
            gene_embeds         -   aligned encoded gene embeddings
            atts_img            -   attention map of the gene tokens
            prompt              -   entire prompt: pre-gene + <geneHere> + post-gene
        Output:
            wrapped_gene_embeds -   embeddings of the wrapped gene sequence - llama input
            wrapped_atts_img    -   attention maps of the wrapped gene sequence - llama input
        '''

        if prompt:

            p_before_lst = []
            p_after_lst = []

            for p in prompt:
                p_before, p_after = p.split('<geneHere>')
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)

            p_before_tokens_lst = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False,
                padding=True, truncation=True).to(img_embeds.device)

            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True,
                padding=True, truncation=True).to(img_embeds.device)
            
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens_lst.input_ids)

            #print("\n", p_before_embeds.shape, img_embeds.shape, p_after_embeds.shape, "\n")
            #print("Prompt: ", self.llama_tokenizer.decode(torch.cat([p_before_tokens_lst.input_ids, p_after_tokens_lst.input_ids], dim=1)[0], add_special_tokens=False))

            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            #wrapped_img_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        seqs = samples["seq"][0] # list of seq
        gene_embeds, atts = self.encode_gene(seqs)

        #print("\n", samples, "\n", samples["text_input"], "\n")
        
        #gene_embeds, atts = torch.rand((1,1,5120), dtype=torch.float64).to(torch.cuda.current_device()), torch.ones((1,1), dtype=torch.long).to(torch.cuda.current_device())

        img_embeds, atts_img = self.prompt_list_wrap(gene_embeds, atts, samples["prompt"])

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]
        
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(gene_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(gene_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        #print("Input Embeds Shape: ", inputs_embeds.shape, " Gene Embeds shape: ", gene_embeds.shape, "Summary part shape: ", to_regress_embeds.shape, " Targets: ", targets.shape, "Attention Mask shape: ", attention_mask.shape)
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        logits = outputs.logits

        # ── Keyword-weighted cross-entropy ─────────────────────────────────────
        # Replace mean CE with per-token weighted CE.
        # targets shape: [B, T]; logits shape: [B, T, V]
        # Shift: logits predict next token, so align with targets.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        import torch.nn.functional as _F
        ce_per_token = _F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction='none',
            ignore_index=-100,
        )  # shape: [B*(T-1)]
        flat_targets = shift_targets.view(-1)
        # look up weight for each target token; always float32 for numerical stability
        valid_mask = (flat_targets != -100)
        weights = torch.ones(flat_targets.shape[0], dtype=torch.float32, device=flat_targets.device)
        weights[valid_mask] = self.token_loss_weights[flat_targets[valid_mask]]
        loss = (ce_per_token.float() * weights).sum() / valid_mask.float().sum().clamp(min=1)
        # ──────────────────────────────────────────────────────────────────────

        #print("Loss: ", loss.item())
        
        #print("===========")
        #print("Output: ", self.llama_tokenizer.batch_decode(logits, skip_special_tokens=True), "\n")
        #print("===========")
        #print("Input: ", self.llama_tokenizer.batch_decode(to_regress_tokens.input_ids[0,:], skip_special_tokens=True))
       
        ''' 
        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds= torch.cat([bos_embeds, img_embeds, to_regress_embeds[:,:5,:]], dim=1),
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.9,
                length_penalty=1,
                temperature=float(0),
                output_hidden_states=False
            )
            output_token = outputs[0]

            print("Answer: ", self.llama_tokenizer.decode(to_regress_tokens.input_ids[0,:], add_special_tokens=False))
            print("Output Text: ", self.llama_tokenizer.decode(output_token, add_special_tokens=False))
            print("===========")        
        '''
        return {"loss": loss}

    @torch.no_grad()
    def generate_preview(self, samples, max_new_tokens=60, repetition_penalty=1.8, **kwargs):
        """
        Greedy-decode one sample from the batch for training monitoring.
        Uses a manual KV-cache loop to avoid reshape bugs in transformers.generate()
        with inputs_embeds.  Returns the generated string (no gradient, eval mode).
        """
        # Pick the requested sample index (default 0)
        idx = kwargs.get("idx", 0)
        seq_one = [samples["seq"][0][idx]]
        gene_embeds, _ = self.encode_gene(seq_one)   # (1, 1, 5120)

        device = gene_embeds.device

        # Reconstruct the same embedding layout as training / get_context_emb:
        #   [prefix text (BOS + "USER: [Gene ID]")] | [gene_embed] | [suffix text (" Q ASSISTANT:")]
        prompt_str = samples["prompt"][idx] if isinstance(samples["prompt"], (list, tuple)) \
                     else samples["prompt"]
        segs = prompt_str.split("<geneHere>")  # ["USER: [Gene ID]", " Q ASSISTANT:"]
        prefix_tokens = self.llama_tokenizer(
            segs[0], return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=64,
        )["input_ids"].to(device)
        suffix_tokens = self.llama_tokenizer(
            segs[1], return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=64,
        )["input_ids"].to(device)
        prefix_embeds = self.llama_model.model.embed_tokens(prefix_tokens)
        suffix_embeds = self.llama_model.model.embed_tokens(suffix_tokens)

        # [prefix] | gene (1 tok) | [suffix]  — matches training forward pass exactly
        inputs_embeds = torch.cat([prefix_embeds, gene_embeds, suffix_embeds], dim=1)

        eos_id  = self.llama_tokenizer.eos_token_id
        generated_ids = []

        with self.maybe_autocast():
            # Step 0: prime the KV cache
            out = self.llama_model(inputs_embeds=inputs_embeds, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            generated_ids.append(next_id.item())

            # Subsequent steps: feed one token at a time using KV cache
            for _ in range(max_new_tokens - 1):
                tok = next_id.item()
                if tok == eos_id:
                    break
                out = self.llama_model(input_ids=next_id, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]  # (1, vocab)
                if repetition_penalty != 1.0 and generated_ids:
                    for prev_id in set(generated_ids):
                        if logits[0, prev_id] > 0:
                            logits[0, prev_id] /= repetition_penalty
                        else:
                            logits[0, prev_id] *= repetition_penalty
                next_id = logits.argmax(dim=-1, keepdim=True)
                generated_ids.append(next_id.item())

        text = self.llama_tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Collapse embedded newlines / extra whitespace → single-line log output
        text = ' '.join(text.split())
        # Strip prompt-format leakage: anything from USER: onwards (new-turn indicator)
        if 'USER:' in text:
            text = text[:text.index('USER:')].strip()
        return text

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — REINFORCE helpers
    # WARNING: These methods are only used by protein_text_reinforce task.
    #          They are NOT called during Stage 1 or Stage 2 training.
    #          Only enable Stage 3 after Stage 1 + Stage 2 are complete.
    # ══════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate_sample(self, samples, idx=0, max_new_tokens=128, temperature=0.9):
        """
        Sample (not greedy) from the model for one item in the batch.
        Returns the generated token IDs as a list (used for log prob computation).
        Stage 3 / REINFORCE only.
        """
        seq_one = [samples["seq"][0][idx]]
        gene_embeds, _ = self.encode_gene(seq_one)
        device = gene_embeds.device

        prompt_str = samples["prompt"][idx] if isinstance(samples["prompt"], (list, tuple)) \
                     else samples["prompt"]
        segs = prompt_str.split("<geneHere>")
        prefix_tokens = self.llama_tokenizer(
            segs[0], return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=64,
        )["input_ids"].to(device)
        suffix_tokens = self.llama_tokenizer(
            segs[1], return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=64,
        )["input_ids"].to(device)
        prefix_embeds = self.llama_model.model.embed_tokens(prefix_tokens)
        suffix_embeds = self.llama_model.model.embed_tokens(suffix_tokens)
        inputs_embeds = torch.cat([prefix_embeds, gene_embeds, suffix_embeds], dim=1)

        eos_id = self.llama_tokenizer.eos_token_id
        generated_ids = []

        with self.maybe_autocast():
            out  = self.llama_model(inputs_embeds=inputs_embeds, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]  # (1, vocab)
            probs  = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample
            generated_ids.append(next_id.item())

            for _ in range(max_new_tokens - 1):
                if next_id.item() == eos_id:
                    break
                out  = self.llama_model(input_ids=next_id, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits  = out.logits[:, -1, :]
                probs   = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                generated_ids.append(next_id.item())

        return generated_ids  # list of int token IDs

    def compute_sequence_logprob(self, samples, generated_ids_batch):
        """
        Compute the mean log probability of each generated sequence given the
        gene embedding.  Gradients flow through this — used for REINFORCE update.

        samples:              standard training sample dict (gene seqs + prompts)
        generated_ids_batch:  list of lists of int token IDs (one per batch item)

        Returns: tensor of shape [B], one mean log-prob per sample.
        Stage 3 / REINFORCE only.
        """
        import torch.nn.functional as _F
        seqs = samples["seq"][0]
        gene_embeds, atts = self.encode_gene(seqs)
        img_embeds, atts_img = self.prompt_list_wrap(gene_embeds, atts, samples["prompt"])

        self.llama_tokenizer.padding_side = "right"

        # Decode generated IDs back to text to use the same tokenizer path as forward()
        gen_texts = [
            self.llama_tokenizer.decode(ids, skip_special_tokens=True) + self.end_sym
            for ids in generated_ids_batch
        ]
        to_regress_tokens = self.llama_tokenizer(
            gen_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(gene_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.full(
            [atts_img.shape[0], atts_img.shape[1] + 1], -100,
            dtype=torch.long, device=gene_embeds.device,
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=gene_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos   = atts_img[:, :1]
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds  = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=None,
            )

        logits = outputs.logits  # [B, T, V]
        shift_logits  = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        log_probs_all = _F.log_softmax(shift_logits.float(), dim=-1)  # [B, T-1, V]
        valid = (shift_targets != -100)  # [B, T-1]
        clamped = shift_targets.clamp(min=0)
        token_lp = log_probs_all.gather(-1, clamped.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        # Mean log prob per sample (normalise by sequence length)
        mean_lp = (token_lp * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp(min=1)
        return mean_lp  # [B]

    def compute_text_logprob(self, samples, texts):
        """
        Compute mean log prob of text strings given gene embeddings.
        Used for DPO loss — same forward pass as training but returns
        per-sample log probs instead of averaged CE loss.

        samples: dict with 'seq' and 'prompt' keys
        texts:   list of strings (one per batch item) — chosen or rejected
        Returns: tensor [B] of mean log probs (higher = model assigns more prob)
        """
        import torch.nn.functional as _F

        seqs = samples["seq"][0]
        gene_embeds, atts = self.encode_gene(seqs)
        img_embeds, atts_img = self.prompt_list_wrap(gene_embeds, atts, samples["prompt"])

        self.llama_tokenizer.padding_side = "right"
        text_tokens = self.llama_tokenizer(
            [t + self.end_sym for t in texts],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(gene_embeds.device)

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.full(
            [atts_img.shape[0], atts_img.shape[1] + 1], -100,
            dtype=torch.long, device=gene_embeds.device,
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=text_tokens.input_ids.dtype,
                         device=gene_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds  = self.llama_model.model.embed_tokens(bos)
        atts_bos    = atts_img[:, :1]
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        inputs_embeds  = torch.cat([bos_embeds, img_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, text_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=None,
            )

        logits = outputs.logits
        shift_logits  = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        log_probs_all = _F.log_softmax(shift_logits.float(), dim=-1)
        valid   = (shift_targets != -100)
        clamped = shift_targets.clamp(min=0)
        token_lp = log_probs_all.gather(-1, clamped.unsqueeze(-1)).squeeze(-1)
        mean_lp  = (token_lp * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp(min=1)
        return mean_lp  # [B]

    @classmethod
    def from_config(cls, cfg):
        '''
        Get the configuration parameters from the config file
        '''
        
        llama_model = cfg.get("llama_model")

        gene_model=cfg.get("gene_model")
        freeze_gene_encoder = cfg.get("freeze_gene_encoder", False)
        max_gene_length=cfg.get("max_gene_length", 5000)

        freeze_adaptor = cfg.get("freeze_adaptor", False)

        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)
        keyword_loss_alpha = cfg.get("keyword_loss_alpha", 3.0)
        adaptor_type = cfg.get("adaptor_type", "linear")
        encoder_type = cfg.get("encoder_type", "dnabert2")

        model = cls(
            freeze_gene_encoder=freeze_gene_encoder,
            gene_model=gene_model,
            max_gene_length=max_gene_length,
            freeze_adaptor=freeze_adaptor,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            embedding_agg=embedding_agg,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            keyword_loss_alpha=keyword_loss_alpha,
            adaptor_type=adaptor_type,
            encoder_type=encoder_type,
        )
        
        stage1_ckpt = cfg.get("stage1_ckpt", "")  # load weights of encoder and adaptor layer
        if stage1_ckpt:
            print("\n\n------>Load HyenaDNA and adaptor layer Checkpoint: {}".format(stage1_ckpt))
            ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
            msg = model.load_state_dict(ckpt['model'], strict=False)
        peft_ckpt = cfg.get("peft_ckpt", "")  # load weights of LoRA
        if peft_ckpt:
            print("\n\n-------> Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu", weights_only=False)
            msg = model.load_state_dict(ckpt['model'], strict=False)
            #print(msg)
        
        return model