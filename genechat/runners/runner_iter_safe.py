"""
Modified iteration-based runner with safer checkpoint saving:
1. Saves checkpoints less frequently
2. Auto-deletes old checkpoints to save disk space
3. Handles disk errors gracefully
"""

import datetime
import logging
import os
import time
import glob

import torch
import torch.distributed as dist
import webdataset as wds
from genechat.common.dist_utils import download_cached_file, is_main_process, main_process
from genechat.common.registry import registry
from genechat.common.utils import is_url
from genechat.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from genechat.runners.runner_base import RunnerBase
from torch.utils.data.dataset import ChainDataset


@registry.register_runner("runner_iter_safe")
class RunnerIterSafe(RunnerBase):
    """
    Safer iteration-based runner with:
    - Configurable checkpoint frequency
    - Auto-deletion of old checkpoints
    - Disk space checking before saving
    """

    def __init__(self, cfg, task, model, datasets, wandb, job_id):
        super().__init__(cfg, task, model, datasets, wandb, job_id)

        self.start_iters = 0
        self.max_iters = int(self.config.run_cfg.get("max_iters", -1))
        assert self.max_iters > 0, "max_iters must be greater than 0."

        self.iters_per_inner_epoch = int(
            self.config.run_cfg.get("iters_per_inner_epoch", -1)
        )
        assert self.iters_per_inner_epoch > 0, "iters_per_inner_epoch must be greater than 0."

        # New: configurable checkpoint frequency
        self.checkpoint_freq = int(self.config.run_cfg.get("checkpoint_freq", 1))
        logging.info(f"Saving checkpoints every {self.checkpoint_freq} inner epochs "
                    f"({self.checkpoint_freq * self.iters_per_inner_epoch} iterations)")

        # New: max checkpoints to keep
        self.max_checkpoints = int(self.config.run_cfg.get("max_checkpoints", 3))
        logging.info(f"Keeping maximum {self.max_checkpoints} checkpoints")

    @property
    def max_epoch(self):
        return int(self.max_iters / self.iters_per_inner_epoch)

    @property
    def cur_epoch(self):
        try:
            return self.train_loader.epoch
        except AttributeError:
            return 0

    def _progress(self, cur_iters):
        return "{}_iters={}".format(self.cur_epoch, cur_iters)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_iters = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        inner_epoch_counter = 0

        for start_iters in range(
            self.start_iters, self.max_iters, self.iters_per_inner_epoch
        ):
            end_iters = start_iters + self.iters_per_inner_epoch
            inner_epoch_counter += 1

            # training phase
            if not self.evaluate_only:
                logging.info(
                    "Start training, max_iters={}, in total {} inner epochs.".format(
                        self.max_iters, int(self.max_iters / self.iters_per_inner_epoch)
                    )
                )

                train_stats = self.train_iters(self.cur_epoch, start_iters, wandb=self.wandb)

                # Save checkpoint only every N inner epochs
                if inner_epoch_counter % self.checkpoint_freq == 0:
                    self._save_checkpoint(end_iters, is_best=False)
                else:
                    logging.info(f"Skipping checkpoint save (will save every {self.checkpoint_freq} epochs)")

                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=self._progress(end_iters)
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_iters, best_agg_metric = end_iters, agg_metrics
                                self._save_checkpoint(end_iters, is_best=True)

                            val_log.update({"best_iters": best_iters})
                            self.log_stats(val_log, split_name)

            else:
                # if no validation split is provided, save checkpoint based on frequency
                if not self.evaluate_only and inner_epoch_counter % self.checkpoint_freq == 0:
                    self._save_checkpoint(end_iters, is_best=False)

            if self.evaluate_only:
                break
            dist.barrier()

        # testing phase
        self.evaluate(cur_epoch=self.cur_epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def train_iters(self, epoch, start_iters, wandb=None):
        # train by iterations
        self.model.train()

        return self.task.train_iters(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_inner_epoch=self.iters_per_inner_epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
            wandb=wandb
        )

    def _check_disk_space(self, required_gb=5):
        """Check if there's enough disk space"""
        import shutil
        stat = shutil.disk_usage(self.output_dir)
        free_gb = stat.free / (1024**3)

        if free_gb < required_gb:
            logging.warning(f"Low disk space: {free_gb:.2f} GB free (need {required_gb} GB)")
            return False
        return True

    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints, keeping only the N most recent"""
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)

        # Don't delete "best" checkpoint
        checkpoints = [c for c in checkpoints if "best" not in c]

        if len(checkpoints) > self.max_checkpoints:
            # Sort by modification time (oldest first)
            checkpoints.sort(key=os.path.getmtime)

            # Delete oldest checkpoints
            to_delete = checkpoints[:len(checkpoints) - self.max_checkpoints]
            for ckpt_path in to_delete:
                try:
                    logging.info(f"Deleting old checkpoint: {ckpt_path}")
                    os.remove(ckpt_path)
                except Exception as e:
                    logging.warning(f"Failed to delete {ckpt_path}: {e}")

    @main_process
    def _save_checkpoint(self, cur_iters, is_best=False):
        """Save checkpoint with disk space checking and cleanup"""

        # Check disk space first
        if not self._check_disk_space(required_gb=5):
            logging.warning("Not enough disk space, attempting cleanup...")
            self._cleanup_old_checkpoints()

            if not self._check_disk_space(required_gb=5):
                logging.error("Still not enough disk space after cleanup! Skipping checkpoint save.")
                return

        save_obj = {
            "model": self.unwrap_dist_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "iters": cur_iters,
        }

        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_iters),
        )

        logging.info("Saving checkpoint at iters {} to {}.".format(cur_iters, save_to))

        try:
            torch.save(save_obj, save_to)
            logging.info("Checkpoint saved successfully")

            # Cleanup old checkpoints after successful save
            if not is_best:
                self._cleanup_old_checkpoints()

        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            logging.error("Attempting to free up space...")
            self._cleanup_old_checkpoints()

            # Try one more time
            try:
                torch.save(save_obj, save_to)
                logging.info("Checkpoint saved successfully on retry")
            except Exception as e2:
                logging.error(f"Failed to save checkpoint even after cleanup: {e2}")
                raise

    def _load_checkpoint(self, url_or_filename):
        """Resume from a checkpoint."""
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_iters = checkpoint["iters"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @property
    def dataloaders(self) -> dict:
        """Inherited from RunnerBase - same implementation"""
        if self._dataloaders is None:
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            if dataset_ratios is None:
                logging.info(
                    "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
                )

                datasets = reorg_datasets_by_split(self.datasets)
                self.datasets = concat_datasets(datasets)
            else:
                missing_keys = [k for k in dataset_ratios if k not in self.datasets]
                if len(missing_keys) > 0:
                    raise ValueError(
                        "Datasets with the following split names are not found: {}".format(
                            missing_keys
                        )
                    )

                unexpected_keys = [k for k in self.datasets if k not in dataset_ratios]
                if len(unexpected_keys) > 0:
                    raise ValueError(
                        "Datasets with the following split names are not expected: {}".format(
                            unexpected_keys
                        )
                    )

                dataset_ratios = [float(dataset_ratios[k]) for k in self.datasets]
                self.datasets = reorg_datasets_by_split(self.datasets)
                self.datasets = {
                    k: v[0] if len(v) == 1 else v for k, v in datasets.items()
                }

            # print dataset statistics
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )
                else:
                    try:
                        num_records = len(self.datasets[split_name])
                    except TypeError:
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            split_names = sorted(self.datasets.keys())
            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders
