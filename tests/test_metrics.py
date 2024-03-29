import logging

import pytest
import torch
from torchmetrics.functional.classification import average_precision

from ride import metrics
from ride.core import RideModule
from ride.utils.utils import attributedict

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


# For unknown reasons, this test fails on the Github Actions build system
@pytest.mark.skip_cicd
def test_MeanAveragePrecisionMetric():
    class DummyModule(
        RideModule, metrics.MeanAveragePrecisionMetric, DummyClassificationDataLoader
    ):
        def __init__(self, hparams):
            self.lin = torch.nn.Linear(self.input_shape[0], self.output_shape)
            self.loss = torch.nn.functional.mse_loss

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.lin(x))
            return x

    targets = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
    )
    preds = torch.tensor([[0.1, 0.9] for _ in range(8)]).clone()

    pl_map = average_precision(preds, targets, num_classes=2)

    net = DummyModule()
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


def test_MetricSelector():
    class DummyModule(
        RideModule,
        DummyClassificationDataLoader,
        metrics.MetricSelector(
            {
                "one": metrics.TopKAccuracyMetric(1),
                "two": [
                    metrics.FlopsWeightedAccuracyMetric,
                    metrics.TopKAccuracyMetric(2),
                ],
            }
        ),
    ):
        def __init__(self, hparams):
            self.lin = torch.nn.Linear(self.input_shape[0], self.output_shape)
            self.loss = torch.nn.functional.mse_loss

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.lin(x))
            return x

    hparams = attributedict(DummyModule.configs().default_values())
    hparams.metric_selection = "two"
    net = DummyModule(hparams)
    targets = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    preds = torch.tensor([[0.1, 0.9] for _ in range(8)])

    assert set(DummyModule.metric_names()) == {
        "flops",
        "flops_weighted_acc",
        "loss",
        "top1acc",
        "top2acc",
    }
    step_output = net.metrics_step(preds, targets)
    assert net.metrics_step(preds, targets)["flops"] == torch.tensor(22.0)
    assert torch.tensor(0.0) < step_output["flops_weighted_acc"]
    assert torch.tensor(0.0) < step_output["top2acc"]
    assert "top1acc" not in step_output
