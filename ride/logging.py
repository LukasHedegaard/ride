import itertools
import os
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.figure import Figure
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    LoggerCollection,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ride.core import ClassificationDataset, Configs, RideMixin
from ride.utils.env import RUN_LOGS_PATH
from ride.utils.io import dump_yaml
from ride.utils.logging import getLogger, process_rank

logger = getLogger(__name__)
ExperimentLogger = Union[TensorBoardLogger, LoggerCollection, WandbLogger]
ExperimentLoggerCreator = Callable[[str], ExperimentLogger]


def singleton_experiment_logger() -> ExperimentLoggerCreator:
    _logger = None

    def experiment_logger(
        name: str = None,
        logging_backend: str = "tensorboard",
        Module=None,
        save_dir=RUN_LOGS_PATH,
    ) -> ExperimentLogger:
        nonlocal _logger
        if not _logger:
            if process_rank != 0:
                _logger = pl.loggers.base.DummyLogger()
                _logger.log_dir = None
                return _logger

            logging_backend = logging_backend.lower()
            if logging_backend == "tensorboard":
                _logger = TensorBoardLogger(save_dir=save_dir, name=name)
            elif logging_backend == "wandb":
                _logger = WandbLogger(
                    save_dir=save_dir,
                    name=name,
                    project=(Module.__name__ if Module else None),
                )
                _logger.log_dir = getattr(
                    _logger.experiment._settings, "_sync_dir", None
                )
            else:
                logger.warn("No valid logger selected.")

        return _logger

    return experiment_logger


experiment_logger = singleton_experiment_logger()


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
            # wandb.log({"chart": plt})
            image_loggers.append(lambda tag, fig: lgr.experiment.add_figure({tag: fig}))
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
        if self.log_dir:
            dump_yaml(
                path=Path(self.log_dir) / f"{self.prefix}_hparams.yaml",
                data=params,
            )

    @rank_zero_only
    def log_metrics(self, metrics: Dict, step):
        self.results = {self._fix_name_perfix(k): float(v) for k, v in metrics.items()}
        if self.log_dir:
            dump_yaml(
                path=Path(self.log_dir) / f"{self.prefix}_results.yaml",
                data={k: float(v) for k, v in self.results.items()},
            )

    def log_figure(self, tag: str, fig: Figure):
        if self.log_dir:
            fig.savefig(self.log_dir / f"{self.prefix}_{tag}.png")

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


@dataclass
class PredTargetPair:
    preds: Tensor
    targets: Tensor


StepOutputs = List[Dict[str, Any]]


class BaseValueLoggerMixin(RideMixin):
    """Abstract base class for value loggers"""

    def validate_attributes(self):
        for attribute in [
            "logger",  # from pytorch_lightning TensorboardLogger
        ]:
            attrgetter(attribute)(self)

    @staticmethod
    def collect_values_for_step(preds: Tensor, targets: Tensor) -> PredTargetPair:
        return PredTargetPair(preds.detach(), targets.detach())

    @staticmethod
    def collect_values_for_epoch(outputs: StepOutputs) -> PredTargetPair:
        preds, targets = [], []
        for x in outputs:
            if "values" in x:
                preds.append(x["values"].preds)
                targets.append(x["values"].targets)

        return PredTargetPair(torch.cat(preds), torch.cat(targets))

    def log_values(self, values: PredTargetPair):
        ...


class TensorboardClassificationPlotsMixin(RideMixin):
    """Logs plots for classification results in Tensorboard"""

    dataloader: ClassificationDataset
    logger: Union[TensorBoardLogger, LoggerCollection]

    def get_tb_logger(self) -> Optional[TensorBoardLogger]:
        if type(self.logger) == TensorBoardLogger:
            return self.logger  # type:ignore
        elif type(self.logger) == LoggerCollection:
            lc: LoggerCollection = self.logger  # type:ignore
            for lg in lc._logger_iterable:
                if type(self.logger) == TensorBoardLogger:
                    return lg  # type:ignore
        return None

    def validate_attributes(self):
        assert issubclass(self.dataloader.__class__, ClassificationDataset)
        assert issubclass(self.logger.__class__, TensorBoardLogger)
        assert issubclass(self.logger.experiment.__class__, SummaryWriter)

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="log_confusion_matrix",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="constant",
            description="Flag indicating whether a confusion matrix for test results should be logged to tensorboard",
        )
        c.add(
            name="log_figure_height",
            type=float,
            default=4.8,
            strategy="constant",
            description="Height of figures logged to Tensorboard",
        )
        c.add(
            name="log_figure_width",
            type=float,
            default=6.4,
            strategy="constant",
            description="Width of figures logged to Tensorboard",
        )
        return c

    def log_values(self, values: PredTargetPair):
        if not (self.hparams.log_confusion_matrix):
            return

        logger = self.get_tb_logger()
        if not logger:
            return

        num_classes = len(self.dataloader.classes)
        confusion_matrix = get_confusion_matrix(
            values.preds, values.targets, num_classes=num_classes
        )

        width: float = self.hparams.log_figure_width
        height: float = self.hparams.log_figure_height
        figsize = (width, height)

        if self.hparams.log_confusion_matrix:
            tag = "Confusion Matrix"
            figure = plot_confusion_matrix(
                confusion_matrix=confusion_matrix,
                num_classes=num_classes,
                class_names=self.dataloader.classes,
                figsize=figsize,
            )
            logger.experiment.add_figure(tag=tag, figure=figure)


def get_confusion_matrix(
    preds: Tensor, labels: Tensor, num_classes: int, normalize="true"
):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """
    from sklearn.metrics import confusion_matrix

    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)), normalize=normalize
    )
    return cmtx


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    num_classes: int,
    class_names: List[str] = None,
    figsize: Tuple[float, float] = None,
):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        confusion_matrix (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or not isinstance(class_names, list):
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = confusion_matrix.max() / 2.0
    for i, j in itertools.product(
        range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
    ):
        color = "white" if confusion_matrix[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(confusion_matrix[i, j], ".2f")
            if confusion_matrix[i, j] != 0
            else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


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
