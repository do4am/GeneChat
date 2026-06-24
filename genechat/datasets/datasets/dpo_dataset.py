import json
import random
from torch.utils.data import Dataset


class DPODataset(Dataset):
    """
    Dataset for DPO fine-tuning.
    Each item: (seq, query, chosen_text, rejected_text)
    Loaded from dpo_pairs.json built by build_dpo_pairs.py.
    """

    def __init__(self, pairs_path):
        with open(pairs_path) as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        seq = p["seq"]
        if isinstance(seq, list):
            seq = seq[0]
        return {
            "seq":      seq,
            "query":    p.get("query", "Tell me about this gene."),
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
        }

    @staticmethod
    def collater(batch):
        return {
            "seq":      [[b["seq"] for b in batch]],  # [B] seqs in one inner list, matching model convention
            "query":    [b["query"] for b in batch],
            "chosen":   [b["chosen"] for b in batch],
            "rejected": [b["rejected"] for b in batch],
        }
