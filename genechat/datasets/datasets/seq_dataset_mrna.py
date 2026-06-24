import os
import logging
from genechat.datasets.datasets.base_dataset import BaseDataset
import json
import random

questions = ["Tell me about this gene.",
             "Please provide a detailed description of the gene.",
             "Describe this gene."]


class SeqDatasetMrna(BaseDataset):
    def __init__(self, text_rule_path, seq_path,
                 seq_mrna_path=None, text_uniprot_path=None, text_extracted_path=None):
        """
        Single-task dataset: mRNA sequence → function description only.

        text_rule_path:      qa_summary_rule_clean.json   (NCBI full summaries, 1x)
        text_uniprot_path:   qa_summary_uniprot_clean.json (UniProt full summaries, 3x)
        text_extracted_path: qa_function_extracted.json   (targeted Q/A pairs, 1x)
        seq_mrna_path:       seq_mrna.json  {gene_id: mrna_sequence}
        seq_path:            seq.json       {gene_id: genomic_sequence}  (fallback)
        """
        self.rule      = json.load(open(text_rule_path, "r"))
        self.uniprot   = json.load(open(text_uniprot_path, "r")) \
                         if text_uniprot_path and os.path.exists(text_uniprot_path) else []
        self.extracted = json.load(open(text_extracted_path, "r")) \
                         if text_extracted_path and os.path.exists(text_extracted_path) else []

        # mRNA sequences (primary) + genomic DNA (fallback)
        self.sequence_mrna = json.load(open(seq_mrna_path, "r")) \
                             if seq_mrna_path and os.path.exists(seq_mrna_path) else {}
        self.sequence_dna  = json.load(open(seq_path, "r"))

        logging.info(f"SeqDatasetMrna: {len(self.sequence_mrna)} mRNA sequences, "
                     f"{len(self.sequence_dna)} genomic sequences (fallback)")

        self.len_rule      = len(self.rule)
        self.len_uniprot   = len(self.uniprot)
        self.len_extracted = len(self.extracted)

        # UniProt at 3x (higher quality), extracted at 1x
        self.split1 = self.len_rule
        self.split2 = self.split1 + 3 * self.len_uniprot
        self.split3 = self.split2 + self.len_extracted

        logging.info(f"SeqDatasetMrna: {self.len_rule} NCBI (1x) + "
                     f"{self.len_uniprot} UniProt (3x) + "
                     f"{self.len_extracted} extracted QA (1x) "
                     f"→ {self.split3} total training pairs")

    def __len__(self):
        return self.split3

    def _get_seq(self, gene_id):
        """Return [seq_str] — always a single-element list for consistent collation."""
        gid = str(gene_id)
        if gid in self.sequence_mrna:
            val = self.sequence_mrna[gid]
            return [val[0] if isinstance(val, list) else val]
        # Fallback: genomic DNA
        seq = self.sequence_dna[gid]
        seq_str = seq[0] if isinstance(seq, list) else seq
        return [seq_str]

    def __getitem__(self, index):
        if index < self.split1:
            entry   = self.rule[index]
            gene_id = entry["Gene Id"]
            answer  = entry["Summary"]
            question = random.choice(questions)
        elif index < self.split2:
            true_index = (index - self.split1) % self.len_uniprot
            entry   = self.uniprot[true_index]
            gene_id = entry["Gene Id"]
            answer  = entry["Summary"]
            question = random.choice(questions)
        else:
            true_index = (index - self.split2) % self.len_extracted
            entry   = self.extracted[true_index]
            gene_id = entry["Gene Id"]
            answer  = entry["A"]
            question = entry["Q"]   # use the specific extracted question

        prompt   = f"USER: [Gene {gene_id}]<geneHere> {question} ASSISTANT:"
        seq      = self._get_seq(gene_id)
        seq_type = "mRNA" if str(gene_id) in self.sequence_mrna else "DNA"

        return {
            "seq":        seq,
            "text_input": answer,
            "prompt":     prompt,
            "seq_type":   seq_type,
        }
