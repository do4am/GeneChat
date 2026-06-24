import os
import logging
import warnings

from genechat.common.registry import registry
from genechat.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from genechat.datasets.datasets.seq_dataset import SeqDataset

@registry.register_builder("seq")
class SeqBuilder(BaseDatasetBuilder):
    train_dataset_cls = SeqDataset
    eval_dataset_cls = SeqDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/seq/seq.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")

        build_info = self.config.build_info
        datasets = dict()

        for split in ['train', 'valid']:
            if not build_info.get(split):
                continue
            storage_path = build_info.get(split).storage
            if not os.path.exists(storage_path):
                warnings.warn("storage path {} does not exist.".format(storage_path))
                continue

            is_train = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            manual_path = os.path.join(storage_path, 'qa_text_manual.json')
            clean_rule_path = os.path.join(storage_path, 'qa_summary_rule_clean.json')
            rule_path = clean_rule_path if os.path.exists(clean_rule_path) else os.path.join(storage_path, 'qa_summary_rule.json')
            clean_uniprot = os.path.join(storage_path, 'qa_summary_uniprot_clean.json')
            uniprot_path  = clean_uniprot if os.path.exists(clean_uniprot) \
                            else os.path.join(storage_path, 'qa_summary_uniprot.json')
            logging.info("Using summary file: {}".format(rule_path))
            if os.path.exists(uniprot_path):
                logging.info("Using UniProt descriptions: {}".format(uniprot_path))
            regulatory_path = os.path.join(storage_path, 'qa_summary_regulatory.json')
            datasets[split] = dataset_cls(
                kw_path = os.path.join(storage_path, 'qa_kw.json'),
                text_rule_path = rule_path,
                text_manual_path = manual_path if os.path.exists(manual_path) else None,
                seq_path = os.path.join(storage_path, 'seq.json'),
                text_uniprot_path = uniprot_path if os.path.exists(uniprot_path) else None,
                text_regulatory_path = regulatory_path if os.path.exists(regulatory_path) else None,
            )

        return datasets

