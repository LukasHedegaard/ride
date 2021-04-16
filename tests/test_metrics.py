import logging

import pytorch_lightning as pl
import torch

from ride import metrics
from ride.core import RideModule

from .dummy_dataset import DummyClassificationDataLoader


def test_faulty_metric_warning(caplog):
    """Test that base class has sensible defaults"""

    caplog.clear()
    with caplog.at_level(logging.ERROR):

        class MyMetric(metrics.MetricMixin):
            ...

    assert caplog.messages == [
        "Subclasses of MetricMixin should define a `_metrics` classmethod, but none was found in MyMetric"
    ]


def test_MeanAveragePrecisionMetric():
    class DummyModule(
        RideModule, DummyClassificationDataLoader, metrics.MeanAveragePrecisionMetric
    ):
        def __init__(self, hparams):
            self.lin = torch.nn.Linear(self.input_shape[0], self.output_shape)
            self.loss = torch.nn.functional.mse_loss

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.lin(x))
            return x

    net = DummyModule()

    _, targets, _ = next(iter(net.train_dataloader()))
    targets = torch.nn.functional.one_hot(targets)
    targets = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
    )
    preds = torch.tensor([[0.1, 0.9] for _ in range(8)])

    pl_map = pl.metrics.classification.AveragePrecision(num_classes=len(net.classes))(
        preds, targets
    )

    assert DummyModule.metric_names() == ["loss", "mAP"]
    assert net.metrics_step(preds, targets)["mAP"] == pl_map
    assert net.metrics_epoch(preds, targets)["mAP"] == pl_map


def test_TopKAccuracyMetric():
    class DummyModule(
        RideModule, DummyClassificationDataLoader, metrics.TopKAccuracyMetric()
    ):
        def __init__(self, hparams):
            self.lin = torch.nn.Linear(self.input_shape[0], self.output_shape)
            self.loss = torch.nn.functional.mse_loss

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.lin(x))
            return x

    net = DummyModule()
    targets = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    preds = torch.tensor([[0.1, 0.9] for _ in range(8)])

    assert DummyModule.metric_names() == ["loss", "top1acc", "top3acc", "top5acc"]
    assert net.metrics_step(preds, targets)["top1acc"] == torch.tensor(0.5)
    assert net.metrics_step(preds, targets)["top3acc"] == torch.tensor(1.0)
    assert net.metrics_step(preds, targets)["top5acc"] == torch.tensor(1.0)


def test_FlopsWeightedAccuracyMetric():
    class DummyModule(
        RideModule, DummyClassificationDataLoader, metrics.FlopsWeightedAccuracyMetric
    ):
        def __init__(self, hparams):
            self.lin = torch.nn.Linear(self.input_shape[0], self.output_shape)
            self.loss = torch.nn.functional.mse_loss

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.lin(x))
            return x

    net = DummyModule()
    targets = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    preds = torch.tensor([[0.1, 0.9] for _ in range(8)])

    assert DummyModule.metric_names() == ["flops", "flops_weighted_acc", "loss"]
    assert net.metrics_step(preds, targets)["flops"] == torch.tensor(22.0)
    assert torch.tensor(0.0) < net.metrics_step(preds, targets)["flops_weighted_acc"]
