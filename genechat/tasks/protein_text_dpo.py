"""
protein_text_dpo.py

DPO (Direct Preference Optimization) fine-tuning task for GeneChat.

Loss (simplified DPO without explicit reference model):
  L = -log σ( β * (log P(chosen|gene) - log P(rejected|gene)) )

This directly teaches the model:
  - increase probability of ground-truth descriptions
  - decrease probability of hallucinated descriptions
"""

import logging
import torch
import torch.nn.functional as F

from genechat.common.registry import registry
from genechat.tasks.base_task import BaseTask


@registry.register_task("protein_text_dpo")
class ProteinTextDPO(BaseTask):

    def __init__(self, dpo_beta=0.1):
        super().__init__()
        self.dpo_beta = dpo_beta  # controls strength of preference signal

    @classmethod
    def setup_task(cls, cfg):
        beta = cfg.run_cfg.get("dpo_beta", 0.1)
        return cls(dpo_beta=beta)

    def train_step(self, model, samples):
        """
        samples keys: seq, query, chosen, rejected
        """
        raw_model = model.module if hasattr(model, "module") else model

        chosen   = samples["chosen"]    # list[str], ground truth
        rejected = samples["rejected"]  # list[str], model hallucination

        # Empty string prompt — falsy, so prompt_list_wrap skips wrapping
        dpo_samples = {
            "seq":    samples["seq"],
            "prompt": "",
        }

        # Compute log probs for chosen and rejected under the current policy
        pi_chosen   = raw_model.compute_text_logprob(dpo_samples, chosen)    # [B]
        pi_rejected = raw_model.compute_text_logprob(dpo_samples, rejected)  # [B]

        # DPO loss: push chosen up, push rejected down
        loss = -F.logsigmoid(self.dpo_beta * (pi_chosen - pi_rejected)).mean()

        # Log metrics
        with torch.no_grad():
            chosen_reward   = self.dpo_beta * pi_chosen.mean().item()
            rejected_reward = self.dpo_beta * pi_rejected.mean().item()
            margin          = (pi_chosen - pi_rejected).mean().item()

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                f"DPO | chosen_lp={pi_chosen.mean():.3f} "
                f"rejected_lp={pi_rejected.mean():.3f} "
                f"margin={margin:.3f} loss={loss.item():.4f}"
            )

        # Attach metrics for the runner to log
        loss._dpo_metrics = {
            "dpo/loss":            loss.item(),
            "dpo/chosen_reward":   chosen_reward,
            "dpo/rejected_reward": rejected_reward,
            "dpo/margin":          margin,
        }

        return loss

    def valid_step(self, model, samples):
        return {}

    def after_evaluation(self, **kwargs):
        return {}
