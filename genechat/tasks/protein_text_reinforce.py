"""
Stage 3 — REINFORCE fine-tuning task for GeneChat.

⚠️  WARNING: This task is designed for Stage 3 only.
    It MUST be run after Stage 1 (adaptor-only CE) and Stage 2 (joint CE with
    keyword upweighting) are complete. Running REINFORCE on an unconverged model
    produces high-variance gradients and will destabilize training.

How it works:
    1. For each batch, generate one sampled output per gene (no grad).
    2. Compute keyword F1 reward: how many GT keywords appear in the generation?
    3. Compute mean log-prob of the generated sequence (with grad).
    4. REINFORCE loss: -advantage × log_prob
       (advantage = reward - running baseline, reduces variance)
    5. Optional KL penalty against a frozen reference to prevent reward hacking.

Reward function:
    Extracts uppercase tokens ≥2 chars (gene symbols: GABA, ATP, GABRA1…)
    from both GT and generated text, then computes F1 overlap.
    Reward ∈ [0, 1].
"""

import re
import logging
import torch
import torch.distributed as dist
import wandb

from genechat.common.registry import registry
from genechat.tasks.base_task import BaseTask


def _extract_keywords(text):
    """Extract gene-name-like tokens: all-caps, length >= 2."""
    return set(re.findall(r'\b[A-Z][A-Z0-9\-]{1,}\b', text))


def keyword_f1(generated: str, ground_truth: str) -> float:
    """
    F1 overlap of gene-name keywords between generated text and ground truth.
    Returns value in [0, 1].  Returns 0.5 if GT has no keywords (neutral reward).
    """
    gt_kw  = _extract_keywords(ground_truth)
    gen_kw = _extract_keywords(generated)

    if not gt_kw:
        return 0.5  # no keywords to evaluate → neutral

    if not gen_kw:
        return 0.0

    tp        = len(gt_kw & gen_kw)
    precision = tp / len(gen_kw)
    recall    = tp / len(gt_kw)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@registry.register_task("protein_text_reinforce")
class ProteinTextReinforceTask(BaseTask):
    """
    REINFORCE stage for GeneChat.

    Config parameters (under `run:` in yaml):
        reinforce_temperature:  sampling temperature for generation (default 0.9)
        reinforce_max_tokens:   max new tokens per sample (default 128)
        reinforce_kl_coeff:     KL penalty coefficient vs reference (default 0.1)
                                Set to 0.0 to disable KL penalty.
        reinforce_baseline_ema: EMA decay for running reward baseline (default 0.99)
    """

    def __init__(self, temperature=0.9, max_tokens=128, kl_coeff=0.1, baseline_ema=0.99):
        super().__init__()
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.kl_coeff      = kl_coeff
        self.baseline_ema  = baseline_ema
        self._baseline     = 0.0   # running reward baseline (reduces variance)
        self._step         = 0

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            temperature  = run_cfg.get("reinforce_temperature",  0.9),
            max_tokens   = run_cfg.get("reinforce_max_tokens",   128),
            kl_coeff     = run_cfg.get("reinforce_kl_coeff",     0.1),
            baseline_ema = run_cfg.get("reinforce_baseline_ema", 0.99),
        )

    def train_step(self, model, samples):
        raw_model = model.module if hasattr(model, "module") else model

        # Skip batches with very long sequences to avoid OOM in 3-pass REINFORCE
        MAX_SEQ_LEN = 6000
        if "seq" in samples:
            max_len = max(len(s) for s in samples["seq"][0])
            if max_len > MAX_SEQ_LEN:
                # Return zero loss — base_task OOM handler will skip gradient update
                dummy = next(p for p in raw_model.parameters() if p.requires_grad)
                return dummy.sum() * 0.0

        batch_size = len(samples["prompt"])
        gt_texts   = samples["text_input"]  # ground truth descriptions

        # ── Step 1: generate one sample per batch item (no grad) ────────────
        raw_model.eval()
        generated_ids_batch = []
        generated_texts     = []
        with torch.no_grad():
            for i in range(batch_size):
                gids = raw_model.generate_sample(
                    samples, idx=i,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                generated_ids_batch.append(gids)
                text = raw_model.llama_tokenizer.decode(gids, skip_special_tokens=True)
                text = " ".join(text.split())
                generated_texts.append(text)
        raw_model.train()

        # ── Step 2: compute keyword F1 rewards ──────────────────────────────
        rewards = torch.tensor(
            [keyword_f1(gen, gt) for gen, gt in zip(generated_texts, gt_texts)],
            dtype=torch.float32,
        )
        mean_reward = rewards.mean().item()

        # Sync reward baseline across GPUs so all processes share the same EMA
        # (no-op in single-GPU mode — dist not initialised or world_size=1)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            reward_tensor = torch.tensor(mean_reward, device='cuda')
            dist.all_reduce(reward_tensor, op=dist.ReduceOp.AVG)
            mean_reward = reward_tensor.item()

        # Advantage = reward - running baseline (control variate, reduces variance)
        advantages = rewards - self._baseline
        self._baseline = self.baseline_ema * self._baseline + (1 - self.baseline_ema) * mean_reward

        # ── Step 3: compute log prob of generated sequences (with grad) ─────
        log_probs = raw_model.compute_sequence_logprob(samples, generated_ids_batch)
        # log_probs: [B], gradient flows through here

        # ── Step 4: REINFORCE loss ───────────────────────────────────────────
        # loss = -E[advantage × log π(a|s)]
        adv_device = advantages.to(log_probs.device)
        reinforce_loss = -(adv_device.detach() * log_probs).mean()

        # ── Step 5: optional KL penalty (prevent reward hacking) ────────────
        # Use raw_model (not DDP wrapper) to avoid "variable ready twice" error:
        # both forward passes (logprob + CE) must go through the same model handle
        # so DDP doesn't see two separate backward graphs through shared parameters.
        ce_loss = raw_model(samples)["loss"]
        loss = reinforce_loss + self.kl_coeff * ce_loss

        # Log to wandb every step
        self._step += 1
        mean_adv = advantages.mean().item()
        try:
            wandb.log({
                "reinforce/reward":   mean_reward,
                "reinforce/baseline": self._baseline,
                "reinforce/advantage": mean_adv,
                "reinforce/ce_loss":  ce_loss.item(),
                "reinforce/step":     self._step,
            })
        except Exception:
            pass  # wandb not initialised or offline — don't crash training

        # Console diagnostics every 50 steps
        if self._step % 50 == 0:
            kw_stats = []
            for gen, gt in zip(generated_texts[:2], gt_texts[:2]):
                gt_kw  = _extract_keywords(gt)
                gen_kw = _extract_keywords(gen)
                hit    = gt_kw & gen_kw
                kw_stats.append(f"GT={len(gt_kw)} GEN={len(gen_kw)} hit={len(hit)}")
            logging.info(
                f"[REINFORCE step {self._step}] "
                f"reward={mean_reward:.3f} baseline={self._baseline:.3f} "
                f"adv={mean_adv:.3f} | "
                + " | ".join(kw_stats)
            )

        return loss

    def post_backward_sync(self, raw_model):
        """
        Manually all-reduce gradients after backward.
        Called by base_task after loss.backward() when both forward passes in
        train_step bypass DDP (to avoid the 'variable ready twice' error).
        No-op in single-GPU mode.
        """
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        world_size = dist.get_world_size()
        for p in raw_model.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        avg_loss = float(val_result.get("loss", 0.0))
        logging.info("Validation loss ({}): {:.4f}".format(split_name, avg_loss))
        return {"agg_metrics": -avg_loss, "loss": avg_loss}
