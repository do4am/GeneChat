import os
import sys
from genechat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random

questions = [
    "Tell me about this gene.",
    "Please provide a detailed description of the gene.",
    "Describe the biological function of this gene step by step, including its molecular role, cellular location, and disease relevance.",
    "What is the primary molecular function of this gene product? Describe the biological pathway it participates in, its subcellular localization, and any known associations with human disease.",
    "Provide a detailed description of this gene including: its biochemical activity or structural role, the cellular processes it regulates, where in the cell it is expressed or localized, and its clinical or disease significance.",
    "Analyze this gene and describe the function of its encoded protein, including its enzymatic or signaling activity, the biological process it belongs to, its tissue or cellular expression pattern, and any disease phenotype linked to its mutation or dysregulation.",
    "Describe this gene by explaining: (1) its molecular function at the protein level, (2) the biological pathway or process it is involved in, (3) its subcellular compartment, and (4) any relevance to genetic disorders or cancer.",
    "What does the protein encoded by this gene do? Explain its role in molecular signaling, metabolism, or structural biology, describe where it acts within the cell, and summarize what is known about its involvement in disease.",
]

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
class SeqDataset(BaseDataset):
    def __init__(self, kw_path, text_rule_path, text_manual_path, seq_path,
                 text_uniprot_path=None, text_regulatory_path=None):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.kw = json.load(open(kw_path, "r"))
        self.rule = json.load(open(text_rule_path, "r"))
        self.manual = json.load(open(text_manual_path, "r")) if text_manual_path and os.path.exists(text_manual_path) else []
        # UniProt Swiss-Prot descriptions (manually curated, higher quality)
        # Sampled at 3x rate over NCBI summaries to compensate for smaller count
        self.uniprot = json.load(open(text_uniprot_path, "r")) if text_uniprot_path and os.path.exists(text_uniprot_path) else []
        # Regulatory/ENCODE elements — sampled at 0.1x to avoid mode collapse
        # Model learns to recognize non-coding sequences and output correct response
        self.regulatory = json.load(open(text_regulatory_path, "r")) if text_regulatory_path and os.path.exists(text_regulatory_path) else []
        self.sequence = json.load(open(seq_path, "r"))

        self.rate = {'kw': 1, 'rule': 1, 'manual': 4, 'uniprot': 3, 'regulatory': 0.1}
        self.len_kw = len(self.kw)
        self.len_rule = len(self.rule)
        self.len_manual = len(self.manual)
        self.len_uniprot = len(self.uniprot)
        self.len_regulatory = len(self.regulatory)

        # regulatory uses 0.1x rate — take 10% of entries
        self.len_regulatory_eff = max(1, int(self.len_regulatory * self.rate['regulatory'])) if self.len_regulatory > 0 else 0

        self.split1 = self.rate['kw'] * self.len_kw
        self.split2 = self.split1 + self.rate['rule'] * self.len_rule
        self.split3 = self.split2 + self.rate['manual'] * self.len_manual
        self.split4 = self.split3 + self.rate['uniprot'] * self.len_uniprot
        self.split5 = self.split4 + self.len_regulatory_eff

        import logging
        if self.len_uniprot > 0:
            logging.info(f"SeqDataset: loaded {self.len_uniprot} UniProt descriptions (sampled 3x)")
        if self.len_regulatory > 0:
            logging.info(f"SeqDataset: loaded {self.len_regulatory} regulatory elements (sampled 0.1x = {self.len_regulatory_eff} effective)")

    def __len__(self):
        return self.split5

    def __getitem__(self, index):
        
        if index < self.split1: # sample kw 
            gene_id = self.kw[index]["Gene Id"]
            answer = self.kw[index]["A"]
            query = self.kw[index]['Q']
            query += q_map[query]
            prompt = f"USER: [Gene {gene_id}]<geneHere> {query} ASSISTANT:"
        elif index < self.split2: # sample rule based functionality
            true_index  = (index - self.split1) % self.len_rule
            gene_id = self.rule[true_index]["Gene Id"]
            answer = self.rule[true_index]["Summary"]

            '''
            ########################################################################################################################################## - CHANGE 
            if 'Name' in self.rule[true_index]:
                name = self.rule[true_index]["Name"]
                prompt = f"USER: [Gene {name}-{gene_id}]<geneHere> {random.choice(questions)} ASSISTANT:"
            else:
                prompt = f"USER: [Gene {gene_id}]<geneHere> {random.choice(questions)} ASSISTANT:"
            ########################################################################################################################################## - CHANGE 
            '''
            prompt = f"USER: [Gene {gene_id}]<geneHere> {random.choice(questions)} ASSISTANT:"
        elif index < self.split3: # sample manual annotated functionality
            true_index  = (index - self.split2) % self.len_manual
            gene_id = self.manual[true_index]["Gene Id"]
            answer = self.manual[true_index]["Summary"]
            prompt = f"USER: [Gene {gene_id}]<geneHere> {random.choice(questions)} ASSISTANT:"
        elif index < self.split4: # sample UniProt Swiss-Prot descriptions (highest quality)
            true_index  = (index - self.split3) % self.len_uniprot
            gene_id = self.uniprot[true_index]["Gene Id"]
            answer = self.uniprot[true_index]["Summary"]
            prompt = f"USER: [Gene {gene_id}]<geneHere> {random.choice(questions)} ASSISTANT:"
        else: # sample regulatory/ENCODE elements (low rate, teaches recognition)
            true_index  = (index - self.split4) % self.len_regulatory
            gene_id = self.regulatory[true_index]["Gene Id"]
            answer = self.regulatory[true_index]["Summary"]
            prompt = f"USER: [Gene {gene_id}]<geneHere> {random.choice(questions)} ASSISTANT:"

        seq = self.sequence[gene_id]

        if len(seq[0]) > 160000:
            seq[0] = seq[0][:159999]

        return {
            "seq": seq,
            "text_input": answer,
            "prompt": prompt
        }

    # stage1-Qformer
        # gene_id = self.annotation[index]["gene_id"]
        # seq = self.sequence[gene_id]
        # answer = self.annotation[index]["name"]

        # if len(seq) > 1024:
        #     seq = seq[:1024]

        # return {
        #     "seq": seq,
        #     "text_input": answer
        # }