"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import time
import torch
import torch.distributed as dist
from genechat.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from genechat.common.logger import MetricLogger, SmoothedValue
from genechat.common.registry import registry
from genechat.datasets.data_utils import prepare_sample
import wandb

_gen_review_file = None   # lazily opened on first write

def _write_gen_review(line):
    """Append a line to log_gen_review.txt in the current output directory."""
    global _gen_review_file
    if _gen_review_file is None:
        out_dir = registry.get_path("output_dir") or "."
        path = os.path.join(out_dir, "log_gen_review.txt")
        _gen_review_file = open(path, "a", buffering=1)  # line-buffered
    _gen_review_file.write(line + "\n")

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):

        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)

        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        # print(f"datasets_config is {datasets_config}")

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            # print(datasets_config)
            # print(name)

            # dataset['train'].name = name
            # if 'sample_ratio' in dataset_config:
            #     dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset
            # print(datasets)
        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True, wandb=None):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50

        results = []
        iters_per_epoch = 1000

        # for samples in metric_logger.log_every(data_loader, print_freq, header):
        #     samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        for i in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
            iter_start = time.time()
            # print(f"[base_task] _train_inner_loop ITERATION start -----------------------------")
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            loss = self.valid_step(model=model, samples=samples)
            # print(samples, i, loss)

            metric_logger.update(loss=loss.item())
            results.append(loss.item())

            wandb.log({'loss': loss.item()})

        if is_dist_avail_and_initialized():
            dist.barrier()

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        # print(f"[base_task] _train_inner_loop -----------------------------{time.time()-start}")
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        wandb=None
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            wandb=wandb
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        wandb=None
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            wandb=wandb
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        wandb=None
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """

        start = time.time()
        # print(f"[base_task] _train_inner_loop -----------------------------")
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))


        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )

        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        # print("parameters types")
        # for param in model.module.named_parameters():
        #     print(param[0])
        #     print(param[1].size(), param[1].dtype)

        # input()
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            iter_start = time.time()
            # print(f"[base_task] _train_inner_loop ITERATION start -----------------------------")
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            # samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            end_samples = time.time()
            # print(f"[base_task] _train_inner_loop end_samples: {end_samples - iter_start}")

            # exit()
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            end_update_sample = time.time()
            # print(f"[base_task] _train_inner_loop end_update_sample: {end_update_sample - end_samples}")

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)


            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Tasks that bypass DDP in train_step (e.g. REINFORCE) manually
                # all-reduce their own gradients via this hook.
                if hasattr(self, 'post_backward_sync'):
                    _raw = model.module if hasattr(model, 'module') else model
                    self.post_backward_sync(_raw)

                # update gradients every accum_grad_iters iterations
                if (i + 1) % accum_grad_iters == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                wandb.log({'loss': loss.item()})

                # Periodically clear cache to prevent fragmentation buildup
                if (i + 1) % 1000 == 0:
                    torch.cuda.empty_cache()

                # Generation preview every 10 iters — catch mode collapse early
                _PREVIEW_FREQ = 100
                global_iter = (start_iters if start_iters is not None else inner_epoch * iters_per_epoch) + i + 1
                if (i + 1) % _PREVIEW_FREQ == 0 and is_main_process():
                    raw_model = model.module if hasattr(model, "module") else model
                    if hasattr(raw_model, "generate_preview") and \
                            "text_input" in samples and "prompt" in samples:
                        import random as _random
                        _idx = _random.randrange(len(samples["prompt"])) if isinstance(samples["prompt"], (list, tuple)) else 0
                        prompt0 = samples["prompt"][_idx]
                        target0 = samples["text_input"][_idx]
                        # Extract gene ID from prompt: "USER: [Gene XXXXX]<geneHere> Q ASSISTANT:"
                        import re as _re
                        _m = _re.search(r'\[Gene ([^\]]+)\]', prompt0)
                        gene_id_str = _m.group(1) if _m else "unknown"
                        seq_type = samples["seq_type"][_idx] if "seq_type" in samples else "?"
                        seq_len  = len(samples["seq"][0][_idx]) if "seq" in samples else "?"
                        # Extract the question (between <geneHere> and ASSISTANT:)
                        q_start = prompt0.find("<geneHere>") + len("<geneHere>")
                        q_end   = prompt0.find("ASSISTANT:")
                        question = prompt0[q_start:q_end].strip() if q_start > 10 and q_end > 0 else prompt0
                        try:
                            raw_model.eval()
                            predicted = raw_model.generate_preview(samples, max_new_tokens=150, idx=_idx)
                            raw_model.train()
                            msg = (
                                f"\n[global_iter {global_iter}] --- GENERATION PREVIEW ---"
                                f"\n  GeneID: {gene_id_str} [{seq_type}, len={seq_len}]"
                                f"\n  Q  : {question}"
                                f"\n  GT : {str(target0)}"
                                f"\n  GEN: {predicted}"
                            )
                            _write_gen_review(msg)
                        except Exception as _exc:
                            raw_model.train()
                            logging.warning(f"[global_iter {global_iter}] generate_preview failed: {_exc}")

            except torch.cuda.OutOfMemoryError:
                try:
                    seq_len = len(samples["seq"][0][0]) if "seq" in samples else "unknown"
                except Exception:
                    seq_len = "unknown"
                logging.warning("OOM at iter {}, seq_len={}, skipping batch.".format(i, seq_len))
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            # print(f"[base_task] _train_inner_loop ITERATION end -----------------------------{time.time()-iter_start}")

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        # print(f"[base_task] _train_inner_loop -----------------------------{time.time()-start}")
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
