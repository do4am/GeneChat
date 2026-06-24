import argparse
import os
os.environ['HF_HOME'] = '/home/namdo'
import re
import random
import time
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from genechat.common.config import Config
from genechat.common.registry import registry
from genechat.common.dist_utils import get_rank, init_distributed_mode
from genechat.common.conversation import Chat, CONV_VISION

from eval import get_simcse
import json

from genechat.datasets.builders import *
from genechat.models import *
from genechat.runners import *
from genechat.tasks import *


ASPECT_QUESTIONS = [
    "What is the primary molecular function of the protein encoded by this gene?",
    "What enzymatic activity, binding activity, or structural role does this protein carry out?",
    "Where in the cell is the protein encoded by this gene localized or expressed?",
    "In which tissues or cell types is this gene most highly expressed?",
    "What biological pathway or cellular process does this gene participate in?",
    "What other proteins does this gene product interact with or regulate?",
    "What post-translational modifications does this protein undergo, such as phosphorylation, ubiquitination, or glycosylation?",
    "What protein family or structural domain does this gene belong to?",
    "What human diseases or clinical phenotypes are associated with mutations or dysregulation of this gene?",
    "Is this gene evolutionarily conserved, and what is known about its homologs in other species?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aspect-based GeneChat Evaluation")
    parser.add_argument("--cfg-path", default="configs/genechat_eval_ntv2.yaml",
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def upload_gene(chat, seq, gene_id=0, name=None):
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_gene(seq, chat_state, img_list, gene_id, name)
    return chat_state, img_list


def ask_and_answer(chat, question, chat_state, img_list):
    chat.ask(question, chat_state)
    llm_message, _, loss = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1e-3,
        top_p=0.9,
        repetition_penalty=2.5,
        max_new_tokens=200,
        max_length=4096,
    )
    llm_message = re.sub(r'\[(?:provided|supplied) by [^\]]+\]', '', llm_message).strip()
    return llm_message, loss


def eval_aspect(qa_list, seqs, chat, use_mrna, seqs_mrna, max_gene_length):
    results = []

    for item in qa_list:
        reference = item['Summary']
        gene_id = item['Gene Id']
        name = item.get('Name', None)

        seq = seqs.get(str(gene_id)) or seqs.get(gene_id)
        if seq is None:
            print(f"WARNING: gene {gene_id} not found — skipping")
            continue

        seq_str = seq[0] if isinstance(seq, list) else seq
        seq_src = "mRNA" if (use_mrna and str(gene_id) in seqs_mrna) else "DNA"
        if len(seq_str) > max_gene_length:
            seq_str = seq_str[:max_gene_length]

        aspect_answers = []
        total_loss = 0.0
        for question in ASPECT_QUESTIONS:
            # Fresh chat state per question to prevent context explosion
            chat_state, img_list = upload_gene(chat, seq_str, gene_id, name)
            answer, loss = ask_and_answer(chat, question, chat_state, img_list)
            aspect_answers.append(answer)
            total_loss += loss

        combined_prediction = " ".join(aspect_answers)

        print(f"Gene ID: {gene_id} [{seq_src}, len={len(seq_str)}]")
        for q, a in zip(ASPECT_QUESTIONS, aspect_answers):
            print(f"  Q: {q}")
            print(f"  A: {a}")
        print(f"Reference: {reference}")
        print("=" * 80)

        results.append({
            "gene_id": gene_id,
            "correct_func": reference,
            "predict_func": combined_prediction,
            "aspect_answers": {q: a for q, a in zip(ASPECT_QUESTIONS, aspect_answers)},
            "avg_loss": total_loss / len(ASPECT_QUESTIONS),
        })

    return results


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{args.gpu_id}")
    chat = Chat(model, device=f"cuda:{args.gpu_id}")
    print("Model loaded.")

    TEST_DIR = "/home/namdo/applications/data_hm/test_hm"
    seq_dna_path  = os.path.join(TEST_DIR, "seq.json")
    seq_mrna_path = os.path.join(TEST_DIR, "seq_mrna.json")
    seqs_dna = json.load(open(seq_dna_path))

    use_mrna = "seq_mrna" in cfg.config.get("datasets", {})
    if use_mrna and os.path.exists(seq_mrna_path):
        seqs_mrna = json.load(open(seq_mrna_path))
        seqs = dict(seqs_dna)
        seqs.update({k: [v] if isinstance(v, str) else v for k, v in seqs_mrna.items()})
        print(f"Using mRNA sequences ({len(seqs_mrna):,} genes), DNA fallback ({len(seqs_dna):,} genes)")
    else:
        seqs_mrna = {}
        seqs = seqs_dna
        print(f"Using genomic DNA sequences ({len(seqs_dna):,} genes)")

    max_gene_length = model_config.get("max_gene_length", 5000)

    qa_files = []
    rule_clean   = os.path.join(TEST_DIR, "qa_summary_rule_clean.json")
    uniprot_clean = os.path.join(TEST_DIR, "qa_summary_uniprot_clean.json")
    if os.path.exists(rule_clean):
        qa_files.append(("rule_clean", json.load(open(rule_clean))))
    if os.path.exists(uniprot_clean):
        qa_files.append(("uniprot", json.load(open(uniprot_clean))))

    simcse_path = "princeton-nlp/sup-simcse-roberta-large"
    os.makedirs("result_mrna", exist_ok=True)
    if args.output_name:
        output_path = os.path.join("result_mrna", f"{args.output_name}.json")
    else:
        cfg_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
        output_path = os.path.join("result_mrna", f"eval_aspect_{cfg_name}.json")

    all_results = []
    for qa_name, qa_list in qa_files:
        qa_list = [q for q in qa_list if q.get("Summary", "").strip()]
        print(f"\nEvaluating on {qa_name} ({len(qa_list)} entries)...")
        results = eval_aspect(qa_list, seqs, chat, use_mrna, seqs_mrna, max_gene_length)
        all_results.extend(results)
        print(f"  {qa_name}: {len(results)} genes evaluated")

    scores = get_simcse(simcse_path, all_results)
    print("\nFinal scores:", scores[-1])

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")
