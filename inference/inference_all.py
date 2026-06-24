import argparse
import os
import re
import random
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from collections import Counter

from genechat.common.config import Config
from genechat.common.registry import registry
from genechat.common.dist_utils import get_rank, init_distributed_mode
from genechat.common.conversation import Chat, CONV_VISION
import copy

from eval import get_simcse, get_simcse_llm_param
import json

# imports modules for registration
from genechat.datasets.builders import *
from genechat.models import *
from genechat.runners import *
from genechat.tasks import *



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.",
                        default='configs/genechat_eval.yaml')
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--output-name", type=str, default=None,
                        help="custom output filename (without extension). "
                             "Saved to result_mrna/<output-name>.json. "
                             "Defaults to eval_results_<cfg-name>.json in the config dir.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
print(args)
cfg = Config(args)
init_distributed_mode(cfg.run_cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

'''
count = 0
for name, parameter in model.named_parameters():
    count += 1
    print(name, end="\t")
    if count == 3:
        count = 0
        print()
print()
'''
chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_gene(seq, gene_id=0, name=None):
    chat_state = CONV_VISION.copy()
    img_list = []
    gene_embed, llm_message = chat.upload_gene(seq, chat_state, img_list, gene_id, name)
    return chat_state, img_list, gene_embed

def gradio_ask(user_message, chat_state, function=None):
    chat.ask(user_message, chat_state, function)
    return chat_state

def gradio_answer(chat_state, img_list, num_beams=1, temperature=1e-3, top_p = 0.9, save_embeds=False):
    #print("\n\n--", chat_state, "\n--\n")

    ####################################################################################################    CHANGE
    llm_message, _, loss = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              top_p = top_p,
                              repetition_penalty=2.5,
                              max_new_tokens=300,
                              max_length=1500,
                              save_embeds=save_embeds)
    return llm_message, chat_state, img_list, loss

def gradio_ppl(chat_state, img_list, predict_list):
    # print(chat_state)
    loss = chat.get_ppl(conv=chat_state,
                              img_list=img_list,
                              predict_list=predict_list)
    return loss

questions = [
    "Describe the biological function of this gene step by step, including its molecular role, cellular location, and disease relevance. ",
    "What is the primary molecular function of this gene product? Describe the biological pathway it participates in, its subcellular localization, and any known associations with human disease. ",
    "Provide a detailed description of this gene including: its biochemical activity or structural role, the cellular processes it regulates, where in the cell it is expressed or localized, and its clinical or disease significance. ",
    "Analyze this gene and describe the function of its encoded protein, including its enzymatic or signaling activity, the biological process it belongs to, its tissue or cellular expression pattern, and any disease phenotype linked to its mutation or dysregulation. ",
    "Describe this gene by explaining: (1) its molecular function at the protein level, (2) the biological pathway or process it is involved in, (3) its subcellular compartment, and (4) any relevance to genetic disorders or cancer. ",
    "What does the protein encoded by this gene do? Explain its role in molecular signaling, metabolism, or structural biology, describe where it acts within the cell, and summarize what is known about its involvement in disease. ",
]

def eval_func_text(qa_list, seq):
    start = time.time()
    func_text = []
    func_text_without_seq = []
    loss_list = []

    for item in qa_list:

        function = item['Summary']
        if 'Name' not in item:
            name = None
        else:
            name = item['Name']

        gene_id = item['Gene Id']
        seq = seqs.get(str(gene_id)) or seqs.get(gene_id)
        if seq is None:
            print(f"WARNING: gene {gene_id} not found in seqs — skipping")
            continue
        query = random.choice(questions)

        seq_str = seq[0] if isinstance(seq, list) else seq
        seq_src = "mRNA" if (use_mrna and str(gene_id) in seqs_mrna) else "DNA"
        max_seq_len = model_config.get("max_gene_length", 10000)
        if len(seq_str) > max_seq_len:
            seq_str = seq_str[:max_seq_len]

        user_message = query

        chat_state, img_list, gene_embeds = upload_gene(seq_str, gene_id, name)

        chat_state = gradio_ask(user_message, chat_state, None)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=8)
        llm_message = re.sub(r'\[(?:provided|supplied) by [^\]]+\]', '', llm_message).strip()

        loss_list.append(loss)
        entry = {"seq": seq, "query": query, "correct_func": function[:], "predict_func": llm_message}
        func_text.append(entry)

        entry_without_seq = {"query": query, "correct_func": function[:], "predict_func": llm_message}
        func_text_without_seq.append(entry_without_seq)

        print("Gene ID:", gene_id, f"[{seq_src}, len={len(seq_str)}]")
        print("Loss:", loss)
        print("Correct Function:", function[:])
        print(f"Predicted Function: {llm_message}")
        print('='*80)

    ppl = math.exp(sum(loss_list)/len(loss_list))
    print(ppl)
    end = time.time()
    print(end - start)
    print("******************")

    return func_text, func_text_without_seq

def eval_multi_round():
    func_text = []
    file_path = "data/multi_round/sample.json"
    qa_list = json.load(open(file_path))
    loss_list = []

    for item in qa_list:

        function = item['correct_func']
        seq = item['seq']
        query = item['query']

        if len(seq[0]) > 160000:
            seq[0] = seq[0][:159999]

        user_message = query
        chat_state, img_list, gene_embeds = upload_gene(seq)
        chat_state = gradio_ask(user_message, chat_state)

        #llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=4)
        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=1)
        # message_2 = "What specific antibacterial activity?"
        # message_2 = "Can you elaborate on the specific type of histone protein described, its unique properties, and its function in the regulation of DNA accessibility within cells?"
        message_2 = "What ligand can this protein bind to?"
        chat_state = gradio_ask(message_2, chat_state)

        llm_message_2, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=1)

        message_3 = "Which metal is this protein capable of binding?"
        chat_state = gradio_ask(message_3, chat_state)

        llm_message_3, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=3)

        loss_list.append(loss)
        entry = {"seq": seq, "query": query, "correct_func": function, "predict_func_1": llm_message, "query_2": message_2, "predict_func_2": llm_message_2, "query_3": message_3, "predict_func_3": llm_message_3}
        func_text.append(entry)

        print("seq:", seq)
        print("Correct Function:", function)
        print(f"Predicted Function 1: {llm_message}")
        print(f"Predicted Function 2: {llm_message_2}")
        print('='*80)

    with open("../data/multi_round/result_3.json", "w") as outfile:
        json.dump(func_text, outfile, indent=4)
    return func_text


def eval_LLM_params(qa_list, seq):
    start = time.time()
    func_text = []
    num_beams_list = [1, 2, 4, 8]
    # temperature_list = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
    l = len(num_beams_list)

    loss_list = [[] for i in range(l)] 
    ppl_list = [0 for i in range(l)] 
    # random_numbers = random.sample(list(range(len(qa_list))), k=NUM_TEST)

    for item in qa_list:
        # item = qa_list[random_numbers[i]]

        function = item['caption']
        gene_id = item['Gene Id']
        seq = seqs[gene_id]
        query = random.choice(questions)

        if len(seq[0]) > 160000:
            seq[0] = seq[0][:159999]

         # top_p: 0.9, 0.99 no difference 
        for i in range(l):
            # num_beams = 1
            # temperature = temperature_list[i]
            num_beams = num_beams_list[i]
            temperature = 1e-3

            user_message = query
            chat_state, img_list, gene_embeds = upload_gene(seq)
            chat_state = gradio_ask(user_message, chat_state)

            llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=num_beams, temperature=temperature)
            loss_list[i].append(loss)

            entry = {"gene_id": gene_id, "seq": seq, "query": query, "correct_func": function, "predict_func": llm_message, "num_beams": num_beams, "temperature": temperature, "ppl": loss}
            func_text.append(entry)

            print("Gene ID:", gene_id)
            print("Query:", query)
            print("Correct Function:", function)
            print(f"Predicted Function: {llm_message}")
    
    print(loss_list)
    for i in range(l):
        ppl_list[i] = sum(loss_list[i])/len(loss_list[i])

    print(ppl_list)

    end = time.time()
    print(end - start)
    print("******************")

    return func_text


q_map = {
    "Which organism does the gene belong to?":
    " Limit your answer to one or two words.",
    "What is the locus type of the gene?":
    " Limit your answer to one or two words.",
    "On which chromosome is the gene located?":
    " Limit your answer to one or two words.",
    "How many exons does the gene contain?":
    " Limit your answer to one or two words.",
    "What is the official symbol of the gene?":
    " Limit your answer to one or two words.",
    "What is the official full name of the gene?":
    " Limit your answer to one or two words."
}

def eval_kw(qa_list, seqs):
    start = time.time()
    
    func_text = []

    for item in qa_list:
        function = item['A']
        if ',' in function: 
            # if the answer contains multiple choices, skip
            continue
        gene_id = item['Gene Id']
        query = item['Q']
        query += q_map[query]

        seq = seqs[gene_id]
        
        '''
        if len(seq[0]) > 160000:
            seq[0] = seq[0][:159999]
        '''
        
        user_message = query
        chat_state, img_list, gene_embeds = upload_gene(seq[0])
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list)

        item['predict_func'] = llm_message
        func_text.append(item)
        print(item)

    end = time.time()
    print(end - start)
    print("******************")

    return func_text

'''
def tsne_one_seq(function, seq):

    if len(seq) > 600:
        seq = seq[:600]
    
    query_list = ["Co-chaperone involved in the maturation of iron-sulfur cluster-containing proteins. Seems to help targeting proteins to be folded toward HscA", random.choice(questions)]

    for query in query_list:
        user_message = query
        chat_state, img_list, gene_embeds = upload_gene(seq)
        print(gene_embeds.squeeze().detach().cpu().numpy().shape)
        np.save('/nfs_beijing_ai/mingjia_2023/proteinchat_glm/tsne/protein.npy', gene_embeds.squeeze().detach().cpu().numpy())

        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, save_embeds=True)

        entry = {"query": query, "correct_func": function, "predict_func": llm_message}
        func_text.append(entry)

        print("Query:", query)
        print("Correct Function:", function)
        print(f"Predicted Function: {llm_message}")
   
    return func_text

def tsne_multi_seq(prots):
    
    encoding_array = []
    for entry in prots:
        gene_id = entry['Gene Id']
        seq = entry['seq']
        tag = entry['Class']

        if len(seq[0]) > 160000:
            seq[0] = seq[0][:159999]

        chat_state, img_list, gene_embeds = upload_gene(seq)
        gene_embeds = torch.mean(gene_embeds, 1)
        
        encoding_array.append(gene_embeds.squeeze().detach().cpu().numpy())
    
    encoding_array = np.array(encoding_array)   
    print(encoding_array.shape)
    np.save('tsne/protein.npy', encoding_array)
'''

if  __name__ == "__main__":
    directory_name = "results"
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
        except Exception as e:
            print(f"An error occurred when creating results folder: {e}")

    # eval_ppl()

    # eval_multi_round()

    # result_dir = "results-glm/10-glm-scratch-llama2-kw/ckpt3"

    # for data_dir in ['test']: #'train', 
    #     seqs = json.load(open(f"data/{data_dir}_set/seq.json"))
    #     qa_list = json.load(open(f"data/{data_dir}_set/before_combine/subset/qa_kw.json"))
    #     scores = eval_kw(qa_list, seqs)
    #     with open("tmp.json", "w") as outfile:
    #         json.dump(scores, outfile, indent=4)
    # '/data2/gene_chat/exon_count/data/train_set/qa_summary_rule_unique_s.json'
    # eval func text & kw
    TEST_DIR = "/home/namdo/applications/data_hm/test_hm"
    seq_dna_path  = os.path.join(TEST_DIR, "seq.json")
    seq_mrna_path = os.path.join(TEST_DIR, "seq_mrna.json")
    seqs_dna = json.load(open(seq_dna_path))

    # Use mRNA sequences for mRNA model (seq_mrna builder), genomic DNA for standard model
    use_mrna = "seq_mrna" in cfg.config.get("datasets", {})
    if use_mrna and os.path.exists(seq_mrna_path):
        seqs_mrna = json.load(open(seq_mrna_path))
        # mRNA takes priority; fallback to genomic DNA for genes without mRNA
        seqs = dict(seqs_dna)
        seqs.update({k: [v] if isinstance(v, str) else v for k, v in seqs_mrna.items()})
        mrna_keys = set(str(k) for k in seqs_mrna.keys())
        dna_keys  = set(str(k) for k in seqs_dna.keys())
        print(f"Sequence loading summary:")
        print(f"  mRNA sequences loaded : {len(seqs_mrna):,} genes")
        print(f"  Genomic DNA loaded    : {len(seqs_dna):,} genes")
        print(f"  Total in seqs dict    : {len(seqs):,} genes")
        print(f"  mRNA-only genes       : {len(mrna_keys - dna_keys):,}")
        print(f"  DNA-only (fallback)   : {len(dna_keys - mrna_keys):,}")
        print(f"  Both mRNA+DNA         : {len(mrna_keys & dna_keys):,}")
        # Spot-check: verify a few gene IDs resolve correctly
        sample_ids = list(seqs_mrna.keys())[:3]
        print(f"  Spot-check mRNA genes : {sample_ids}")
        for sid in sample_ids:
            val = seqs.get(sid) or seqs.get(int(sid) if sid.isdigit() else sid)
            seq_str = val[0] if isinstance(val, list) else val
            src = "mRNA" if sid in mrna_keys else "DNA"
            print(f"    gene {sid}: {src}, len={len(seq_str) if seq_str else 'NOT FOUND'}")
    else:
        seqs = seqs_dna
        print(f"Using genomic DNA sequences ({len(seqs_dna):,} genes)")

    # Use cleaned rule summaries + UniProt for evaluation
    qa_files = []
    rule_clean = os.path.join(TEST_DIR, "qa_summary_rule_clean.json")
    uniprot_clean = os.path.join(TEST_DIR, "qa_summary_uniprot_clean.json")
    if os.path.exists(rule_clean):
        qa_files.append(("rule_clean", json.load(open(rule_clean))))
    if os.path.exists(uniprot_clean):
        qa_files.append(("uniprot", json.load(open(uniprot_clean))))

    simcse_path = "princeton-nlp/sup-simcse-roberta-large"
    if args.output_name:
        output_path = os.path.join("result_mrna", f"{args.output_name}.json")
        os.makedirs("result_mrna", exist_ok=True)
    else:
        output_path = os.path.join(os.path.dirname(args.cfg_path),
                                   f"eval_results_{os.path.basename(args.cfg_path).replace('.yaml','')}.json")

    for qa_name, qa_list in qa_files:
        print(f"\nEvaluating on {qa_name} ({len(qa_list)} entries)...")
        # Filter out empty summaries
        qa_list = [q for q in qa_list if q.get("Summary", "").strip()]
        func_text_total = []
        freq = 100
        for i in range(0, len(qa_list), freq):
            func_text, _ = eval_func_text(qa_list[i:i+freq], seqs)
            func_text_total.extend(func_text)
            print(f"  [{i+freq}/{len(qa_list)}] genes evaluated so far")

        # Compute SimCSE + BLEU once at the end (model loads only once)
        scores = get_simcse(simcse_path, func_text_total)
        print(f"\n{qa_name} final scores:", scores)

    with open(output_path, "w") as f:
        json.dump(func_text_total, f, indent=2)
    print(f"Results saved to {output_path}")
                
            #with open("results/esm.json", "a") as outfile:
            #    json.dump(scores, outfile, indent=4)
        
        #qa_list = json.load(open(f"/data2/gene_chat/exon_count/data/{data_dir}_set/qa_kw.json"))
        #scores = eval_kw(qa_list, seqs)
        #with open("results/esm.json", "a") as outfile:
        #    json.dump(scores, outfile, indent=4)


