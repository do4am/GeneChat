import os
import logging
import warnings

from genechat.common.registry import registry
from genechat.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from genechat.datasets.datasets.seq_dataset_mrna import SeqDatasetMrna


@registry.register_builder("seq_mrna")
class SeqBuilderMrna(BaseDatasetBuilder):
    train_dataset_cls = SeqDatasetMrna
    eval_dataset_cls  = SeqDatasetMrna
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/seq/seq.yaml",
    }

    def build_datasets(self):
        logging.info("Building mRNA datasets...")

        build_info = self.config.build_info
        datasets   = dict()

        for split in ['train', 'valid']:
            if not build_info.get(split):
                continue
            storage_path = build_info.get(split).storage
            if not os.path.exists(storage_path):
                warnings.warn("storage path {} does not exist.".format(storage_path))
                continue

            is_train    = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls

            # Description files — prefer cleaned versions
            clean_rule_path = os.path.join(storage_path, 'qa_summary_rule_clean.json')
            rule_path       = clean_rule_path if os.path.exists(clean_rule_path) \
                              else os.path.join(storage_path, 'qa_summary_rule.json')
            clean_uniprot   = os.path.join(storage_path, 'qa_summary_uniprot_clean.json')
            uniprot_path    = clean_uniprot if os.path.exists(clean_uniprot) \
                              else os.path.join(storage_path, 'qa_summary_uniprot.json')

            # Sequence files — protein primary (ESM-2), mRNA secondary, genomic DNA fallback
            seq_protein_path = os.path.join(storage_path, 'seq_protein.json')
            seq_mrna_path    = os.path.join(storage_path, 'seq_mrna.json')
            seq_dna_path     = os.path.join(storage_path, 'seq.json')

            logging.info(f"[{split}] NCBI summaries: {rule_path}")
            logging.info(f"[{split}] UniProt summaries: {uniprot_path}")
            if os.path.exists(seq_protein_path):
                logging.info(f"[{split}] Using protein sequences (ESM-2 mode): {seq_protein_path}")
                seq_mrna_path = seq_protein_path  # reuse mrna slot — same format
            elif not os.path.exists(seq_mrna_path):
                logging.warning(f"[{split}] seq_mrna.json not found — using genomic DNA only")

            extracted_path = os.path.join(storage_path, 'qa_function_extracted.json')

            datasets[split] = dataset_cls(
                text_rule_path      = rule_path,
                seq_path            = seq_dna_path,
                seq_mrna_path       = seq_mrna_path    if os.path.exists(seq_mrna_path)    else None,
                text_uniprot_path   = uniprot_path     if os.path.exists(uniprot_path)     else None,
                text_extracted_path = extracted_path   if os.path.exists(extracted_path)   else None,
            )

        return datasets
