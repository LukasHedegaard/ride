import os

from pytorch_lightning.callbacks import Callback
from ray import tune


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            **{
                k: float(v)
                for k, v in trainer.callback_metrics.items()
                if v is not None
            }
        )


class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        path = tune.make_checkpoint_dir(trainer.global_step)
        trainer.save_checkpoint(os.path.join(path, "checkpoint"))
        tune.save_checkpoint(path)
