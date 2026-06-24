import os
import logging
import warnings

from genechat.common.registry import registry
from genechat.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from genechat.datasets.datasets.dpo_dataset import DPODataset


@registry.register_builder("seq_dpo")
class DPOBuilder(BaseDatasetBuilder):
    train_dataset_cls = DPODataset
    eval_dataset_cls  = DPODataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/seq/seq.yaml"}

    def build_datasets(self):
        logging.info("Building DPO datasets...")
        build_info = self.config.build_info
        datasets   = {}

        for split in ["train"]:
            if not build_info.get(split):
                continue
            storage_path = build_info.get(split).storage
            if not os.path.exists(storage_path):
                warnings.warn(f"storage path {storage_path} does not exist.")
                continue

            pairs_path = os.path.join(storage_path, "dpo_pairs.json")
            if not os.path.exists(pairs_path):
                raise FileNotFoundError(
                    f"dpo_pairs.json not found at {pairs_path}. "
                    "Run build_dpo_pairs.py first."
                )

            logging.info(f"[{split}] DPO pairs: {pairs_path}")
            datasets[split] = DPODataset(pairs_path)

        return datasets
