from enum import Enum
from operator import attrgetter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from ptflops import get_model_complexity_info
from supers import supers
from torch import Tensor

from ride.core import Configs, RideMixin
from ride.utils.logging import getLogger
from ride.utils.utils import merge_dicts, name

MetricDict = Dict[str, Tensor]
StepOutputs = List[Dict[str, Tensor]]

logger = getLogger(__name__)


class OptimisationDirection(Enum):
    MIN = "min"
    MAX = "max"


class MetricMixin(RideMixin):
    """Abstract base class for Ride modules"""

    def __init_subclass__(cls):
        if not hasattr(cls, "_metrics"):
            logger.error(
                f"Subclasses of MetricMixin should define a `_metrics` classmethod, but none was found in {name(cls)}"
            )

    @classmethod
    def metrics(cls) -> Dict[str, str]:
        ms = merge_dicts(
            *[c._metrics() for c in cls.__bases__ if issubclass(c, MetricMixin)]
        )
        return ms

    @classmethod
    def metric_names(cls) -> List[str]:
        return list(sorted(cls.metrics().keys()))

    def metrics_step(self, *args, **kwargs) -> MetricDict:
        return {}  # pragma: no cover

    def metrics_epoch(
        self, outputs: StepOutputs, prefix: str = "", **kwargs
    ) -> MetricDict:
        return {}

    def collect_metrics(self, preds: Tensor, targets: Tensor) -> MetricDict:
        device = preds.device
        mdlist: List[MetricDict] = supers(self).metrics_step(preds, targets)  # type: ignore
        return {
            k: v.to(device=device) if hasattr(v, "to") else v
            for md in mdlist
            for k, v in md.items()
        }

    def collect_epoch_metrics(self, preds: Tensor, targets: Tensor) -> MetricDict:
        device = preds.device
        mdlist: List[MetricDict] = supers(self).metrics_epoch(preds, targets)  # type: ignore
        return {
            k: v.to(device=device) if hasattr(v, "to") else v
            for md in mdlist
            for k, v in md.items()
        }


class MeanAveragePrecisionMetric(MetricMixin):
    """Mean Average Precision (mAP) metric"""

    def validate_attributes(self):
        for attribute in ["hparams.loss", "classes"]:
            attrgetter(attribute)(self)

    @classmethod
    def _metrics(cls):
        return {"mAP": OptimisationDirection.MAX}

    def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
        map = torch.tensor(-1.0)
        try:
            map = compute_map(self)(preds, targets)
        except RuntimeError:  # pragma: no cover
            logger.error("Unable to compute mAP.")
        return {"mAP": map}

    def metrics_epoch(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
        map = torch.tensor(-1.0)
        try:
            map = compute_map(self)(preds, targets)
        except RuntimeError:  # pragma: no cover
            logger.error("Unable to compute mAP.")
        return {"mAP": map}


def compute_map(self):
    if "map_fn" not in vars():
        if "binary" in self.hparams.loss:
            map_fn = pl.metrics.classification.AveragePrecision(pos_label=1)
        else:
            map_fn = pl.metrics.classification.AveragePrecision(
                num_classes=len(self.classes)
            )
    return map_fn


def TopKAccuracyMetric(*Ks) -> MetricMixin:
    if not Ks:
        Ks = [1, 3, 5]

    for k in Ks:
        assert type(k) == int and k > 0

    class TopKAccuracyMetricClass(MetricMixin):
        """Top K accuracy metrics: top1acc, top3acc, top5acc"""

        @classmethod
        def _metrics(cls):
            return {f"top{k}acc": OptimisationDirection.MAX for k in Ks}

        def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
            ks = [k for k in Ks]
            accs = [torch.tensor(-1.0) for _ in ks]
            try:
                accs = topk_accuracies(preds, targets, ks)
            except RuntimeError:  # pragma: no cover
                logger.error("Unable to compute top-k accuracy.")
            return {f"top{k}acc": accs[i] for i, k in enumerate(ks)}

    return TopKAccuracyMetricClass


class FlopsMetric(MetricMixin):
    """Computes Floating Point Operations (FLOPs) for the model and adds it as metric"""

    @classmethod
    def _metrics(cls):
        return {"flops": OptimisationDirection.MIN}

    def on_init_end(self, *args, **kwargs):
        assert isinstance(self, torch.nn.Module)
        self.flops = flops(self)  # type: ignore

    def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
        return {"flops": torch.tensor(self.flops)}


class FlopsWeightedAccuracyMetric(FlopsMetric):
    """Computes acc * (flops / target_gflops) ** (-0.07)"""

    @classmethod
    def _metrics(cls):
        return {
            **{"flops_weighted_acc": OptimisationDirection.MAX},
            **FlopsMetric._metrics(),
        }

    def validate_attributes(self):
        for hparam in FlopsWeightedAccuracyMetric.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="target_gflops",
            type=float,
            default=2.0,
            strategy="constant",
            description="Target (Giga) Floating Point Operations per Second.",
        )
        return c

    def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
        acc = topk_accuracies(preds, targets, ks=[1])[0]
        return {
            **FlopsMetric.metrics_step(self, preds, targets, **kwargs),
            "flops_weighted_acc": acc
            * (self.flops * 1e-9 / self.hparams.target_gflops) ** (-0.07),
        }


def topks_correct(preds: Tensor, labels: Tensor, ks: List[int]) -> List[Tensor]:
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"

    max_k = int(preds.shape[-1])

    # Find the top max_k predictions for each sample
    _, top_max_k_inds = torch.topk(preds, max_k, dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct: List[Tensor] = [
        top_max_k_correct[: min(k, max_k), :].reshape(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds: Tensor, labels: Tensor, ks: List[int]):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) for x in num_topks_correct]


def topk_accuracies(preds: Tensor, labels: Tensor, ks: List[int]):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) for x in num_topks_correct]


def flops(model: torch.nn.Module):
    """Compute the Floating Point Operations per Second for the model"""
    return get_model_complexity_info(
        model,
        model.input_shape,
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True,
    )[0]


def params_count(model: torch.nn.Module):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()])


def make_confusion_matrix(  # noqa: C901
    preds,
    targets,
    num_classes,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=False,
    figsize=None,
    cmap="Blues",
    title=None,
    save_as=None,
):
    """
    Modified from https://github.com/DTrimarchi10/confusion_matrix
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    preds:         Predictions

    targets:       Targets

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    percent:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    """

    from pytorch_lightning.metrics.functional.confusion_matrix import confusion_matrix
    from seaborn import heatmap

    font_logger = getLogger("matplotlib.font_manager")
    font_logger.propagate = False

    if len(preds.shape) > 1:
        preds = preds.squeeze().argmax(-1)
    assert targets.shape and preds.shape and len(targets.shape) == 1
    cf = confusion_matrix(preds, targets, num_classes).cpu().numpy()

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks is False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    fig = plt.figure(figsize=figsize)
    heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    return fig
