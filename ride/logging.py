import io
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
from matplotlib.figure import Figure
from PIL import Image
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    LoggerCollection,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)
from pytorch_lightning.utilities import rank_zero_only

from ride.utils.env import RUN_LOGS_PATH
from ride.utils.logging import getLogger, process_rank

logger = getLogger(__name__)
ExperimentLogger = Union[TensorBoardLogger, LoggerCollection, WandbLogger]
ExperimentLoggerCreator = Callable[[str], ExperimentLogger]


def singleton_experiment_logger() -> ExperimentLoggerCreator:
    _loggers = {}

    def experiment_logger(
        name: str = None,
        logging_backend: str = "tensorboard",
        Module=None,
        save_dir=RUN_LOGS_PATH,
    ) -> ExperimentLogger:
        nonlocal _loggers
        if logging_backend not in _loggers:
            if process_rank != 0:  # pragma: no cover
                _loggers[logging_backend] = pl.loggers.base.DummyLogger()
                _loggers[logging_backend].log_dir = None
                return _loggers[logging_backend]

            logging_backend = logging_backend.lower()
            if logging_backend == "tensorboard":
                _loggers[logging_backend] = TensorBoardLogger(
                    save_dir=save_dir, name=name
                )
            elif logging_backend == "wandb":
                _loggers[logging_backend] = WandbLogger(
                    save_dir=save_dir,
                    name=name,
                    project=(Module.__name__ if Module else None),
                )
                _loggers[logging_backend].log_dir = getattr(
                    _loggers[logging_backend].experiment._settings, "_sync_dir", None
                )
            else:
                logger.warn("No valid logger selected.")

        return _loggers[logging_backend]

    return experiment_logger


experiment_logger = singleton_experiment_logger()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def add_experiment_logger(
    prev_logger: LightningLoggerBase, new_logger: LightningLoggerBase
) -> LoggerCollection:
    # If no logger existed previously don't do anything
    if not prev_logger:
        return None

    if isinstance(prev_logger, LoggerCollection):
        return LoggerCollection([*prev_logger._logger_iterable, new_logger])
    else:
        return LoggerCollection([prev_logger, new_logger])


def log_figures(module: pl.LightningModule, d: Dict[str, Figure]):
    assert isinstance(module, pl.LightningModule)
    module_loggers = (
        module.logger if hasattr(module.logger, "__getitem__") else [module.loggers]
    )
    image_loggers = []
    for lgr in module_loggers:
        if type(lgr) == NeptuneLogger:
            # log_image(log_name, image, step=None)
            image_loggers.append(lgr.log_image)
        elif type(lgr) == TensorBoardLogger:
            # SummaryWriter.add_figure(self, tag, figure)
            image_loggers.append(lgr.experiment.add_figure)
        elif type(lgr) == WandbLogger:
            try:
                import wandb  # noqa: F401
            except ImportError:
                logger.error(
                    "Before using the WandbLogger, first install WandB using `pip install wandb`"
                )

            wandb_log = lgr.experiment.log

            def log_figure(tag, fig):
                im = wandb.Image(fig2img(fig), caption=tag)
                return wandb_log({tag: im})

            image_loggers.append(log_figure)
        elif type(lgr) == ResultsLogger:
            image_loggers.append(lgr.log_figure)

    if not image_loggers:
        logger.warn(
            f"Unable to log figures {d.keys()}: No compatible logger found among {module_loggers}"
        )
        return

    for k, v in d.items():
        for log in image_loggers:
            log(k, v)


class ResultsLogger(LightningLoggerBase):
    def __init__(self, prefix="test", save_to: str = None):
        super().__init__()
        self.results = {}
        self.prefix = prefix
        self.log_dir = save_to

    def _fix_name_perfix(self, s: str, replace="test/") -> str:
        if not self.prefix:
            return s

        if s.startswith(replace):
            return f"{self.prefix}/{s[5:]}"
        else:
            return f"{self.prefix}/{s}"

    @property
    def experiment(self):
        return None

    @rank_zero_only
    def log_hyperparams(self, params):
        ...  # Skip it: hparams are saved in main
        # if self.log_dir:
        #     dump_yaml(
        #         path=Path(self.log_dir) / f"{self.prefix}_hparams.yaml",
        #         data=params,
        #     )

    @rank_zero_only
    def log_metrics(self, metrics: Dict, step):
        self.results = {self._fix_name_perfix(k): float(v) for k, v in metrics.items()}
        ...  # Skip it: results are saved in main
        # if self.log_dir:
        #     dump_yaml(
        #         path=Path(self.log_dir) / f"{self.prefix}_results.yaml",
        #         data={k: float(v) for k, v in self.results.items()},
        #     )

    def log_figure(self, tag: str, fig: Figure):
        if self.log_dir:
            fig_path = str(Path(self.log_dir) / f"{self.prefix}_{tag}.png")
            logger.info(f"Saving confusion matrix to {fig_path}")
            fig.savefig(fig_path)

    @rank_zero_only
    def finalize(self, status):
        pass

    @property
    def save_dir(self) -> Optional[str]:
        return self.log_dir

    @property
    def name(self):
        return "ResultsLogger"

    @property
    def version(self):
        return "1"


StepOutputs = List[Dict[str, Any]]


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.

    Courtesy of Andrew Jong
    https://github.com/PyTorchLightning/pytorch-lightning/issues/2534
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0 and global_step != 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            pl.utilities.rank_zero_info(f"Saving model to {str(ckpt_path)}")
            trainer.save_checkpoint(ckpt_path)
