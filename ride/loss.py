from inspect import isclass
from typing import Any, Callable, Dict, List, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as TorchLossClass

from ride.metrics import MetricMixin
from ride.utils.logging import getLogger
from ride.utils.utils import camel_to_snake, name, some

MetricDict = Dict[str, torch.Tensor]
StepOutputs = List[Dict[str, torch.Tensor]]
AnyLoss = Union[TorchLossClass, Callable[[torch.Tensor, torch.Tensor], Any]]

logger = getLogger(__name__)


class LossMixin(MetricMixin):
    def validate_attributes(self):
        for attr in ["loss_fn", "loss_key"]:
            assert some(
                self, attr
            ), f"{name(self)} should define `{attr}` but none was found."

    def metrics_step(
        self, preds: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricDict:
        return {self.loss_key: self.loss_fn(preds, targets)}


def loss_from_class(cls: TorchLossClass, *args, **kwargs) -> LossMixin:
    assert issubclass(
        cls, TorchLossClass
    ), "Expected loss class from `torch.nn.Modules`."

    class LossFromClass(LossMixin):
        loss_key = camel_to_snake(name(cls))

        def __init__(self, *inner_args, **inner_kwargs):
            self._loss_module = cls(*args, **kwargs)

        def loss_fn(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self._loss_module(logits, target)

    return LossFromClass


def loss_from_fn(
    fn: Callable[[torch.Tensor, torch.Tensor], Any], *args, **kwargs
) -> LossMixin:
    assert callable(
        fn
    ), "Expected loss class from loss function from `torch.nn.functional`."

    class LossFromFn(LossMixin):
        loss_key = name(fn)
        loss_fn = fn

    return LossFromFn


def loss_from(
    class_or_fn: AnyLoss,
    *args,
    **kwargs,
) -> LossMixin:
    if isclass(class_or_fn) and issubclass(class_or_fn, TorchLossClass):
        return loss_from_class(class_or_fn, *args, **kwargs)
    elif callable(class_or_fn):
        return loss_from_fn(class_or_fn, *args, **kwargs)
    else:
        raise ValueError(
            "Expected either loss class from `torch.nn.Modules` or loss function from `torch.nn.functional`."
        )


LossMixin.from_class = loss_from_class
LossMixin.from_fn = loss_from_fn
Loss = loss_from


class CrossEntropyLoss(LossMixin):
    loss_key = "cross_entropy"

    def loss_fn(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)


class MseLoss(LossMixin):
    loss_key = "mse"

    def loss_fn(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(logits, target)
