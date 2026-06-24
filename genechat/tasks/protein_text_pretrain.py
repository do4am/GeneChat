"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from genechat.common.registry import registry
from genechat.tasks.base_task import BaseTask


@registry.register_task("protein_text_pretrain")
class ProteinTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        # val_result is {"loss": "2.345"} from base_task.evaluation()
        # agg_metrics must be higher = better; use negative loss
        avg_loss = float(val_result.get("loss", 0.0))
        import logging
        logging.info("Validation loss ({}): {:.4f}".format(split_name, avg_loss))
        return {"agg_metrics": -avg_loss, "loss": avg_loss}


