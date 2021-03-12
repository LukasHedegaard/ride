# from ride.profile import Profileable
import inspect
from abc import ABC
from typing import Any, List, Sequence, Union

import pytorch_lightning as pl
from corider import Configs as _Configs
from supers import supers

from ride.utils.logging import getLogger
from ride.utils.utils import (
    AttributeDict,
    AttributeDictOrDict,
    merge_attributedicts,
    missing_or_not_in_other,
    name,
    attributedict,
    some,
)

logger = getLogger(__name__)


DataShape = Union[int, Sequence[int], Sequence[Sequence[int]]]


class Configs(_Configs):
    @staticmethod
    def collect(cls: "RideModule") -> "Configs":
        c: Configs = sum(supers(cls).configs())  # type: ignore
        return c

    def default_values(self):
        return attributedict({k: v.default for k, v in self.values.items()})


def _init_subsubclass(cls):
    orig_init = cls.__init__

    def init(self, hparams: AttributeDictOrDict, *args, **kwargs):
        super(cls, self).__init__(hparams, *args, **kwargs)
        apply_init_args(orig_init, self, self.hparams, *args, **kwargs)

    cls.__init__ = init


def _init_subclass(cls):
    # Validate inheritance order
    assert (
        cls.__bases__[0] == RideModule or cls.__bases__[0].__bases__[0] == RideModule
    ), """RideModule must come first in inheritance order, e.g.:
    class YourModule(RideModule, OtherMixin):
        ..."""

    if cls.__bases__[0] is not RideModule:
        _init_subsubclass(cls)
        return

    if not cls.__bases__[-1] == pl.LightningModule:
        cls.__bases__ = (*cls.__bases__, pl.LightningModule)

    # Warn if there is no forward
    if missing_or_not_in_other(
        cls, pl.LightningModule, {"forward"}, must_be_callable=True
    ):
        # if not (some(cls, "forward") and callable(cls.forward)):
        logger.warning(
            f"No `forward` function found in {name(cls)}. Did you forget to define it?"
        )

    # Ensure lifecycle
    lifecycle_steps = {
        "training_step",
        "validation_step",
        "test_step",
    }
    missing_lifecycle_steps = missing_or_not_in_other(
        cls, pl.LightningModule, lifecycle_steps
    )
    if missing_lifecycle_steps == lifecycle_steps:
        logger.info(
            f"No lifecycle steps {missing_lifecycle_steps} found in {name(cls)}. Adding ClassificationLifecycle with CrossEntropyLoss automatically."
        )
        # Import here to break cyclical import
        from ride.lifecycle import TrainValTestLifecycle

        cls.__bases__ = (
            *cls.__bases__[:-1],
            TrainValTestLifecycle,
            *cls.__bases__[-1:],
        )
    elif missing_lifecycle_steps:
        for n in missing_lifecycle_steps:
            logger.warning(
                f"No `{n}` function found in {name(cls)}. Did you forget to define it?"
            )

    # Warn if there is no dataset
    dataset_steps = {"train_dataloader", "val_dataloader", "test_dataloader"}
    dataset_steps_steps = missing_or_not_in_other(
        cls, pl.LightningModule, dataset_steps
    )
    for n in dataset_steps_steps:
        logger.warning(
            f"No `{n}` function found in {name(cls)}. Did you forget to define it?"
        )

    # Monkeypatch derived module init
    orig_init = cls.__init__

    def init(self, hparams: AttributeDictOrDict = {}, *args, **kwargs):
        pl.LightningModule.__init__(self)
        self.hparams = merge_attributedicts(self.configs().default_values(), hparams)
        sself = (
            self
            if type(self).__bases__[0] == RideModule
            else supers(self)._superclasses[0]
        )
        supers(sself)[1:-1].__init__(self.hparams)
        apply_init_args(orig_init, self, self.hparams, *args, **kwargs)
        supers(sself)._init_end(self.hparams, *args, **kwargs)
        supers(self).validate_attributes()

    cls.__init__ = init

    # Monkeypatch derived module configs
    @staticmethod
    def configs():
        return Configs.collect(cls)

    cls.configs = configs


def apply_init_args(fn, self, hparams, *args, **kwargs):
    spec = inspect.getfullargspec(fn)
    valid_kwargs = (
        kwargs
        if spec.varkw == "kwargs"
        else {k: v for k, v in kwargs.items() if k in spec.args}
    )
    if len(spec.args) == 1:
        return fn(self)
    else:
        return fn(self, hparams, *args, **valid_kwargs)


class RideModule:
    def __init_subclass__(cls):
        _init_subclass(cls)

    @property
    def hparams(self) -> AttributeDict:
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @hparams.setter
    def hparams(self, hp: Union[dict, AttributeDict, Any]):
        # Overload the version in pytorch_lightning core to omit DeprecationWarning
        self._hparams = attributedict(hp)


class RideMixin(ABC):
    def __init__(self, hparams: AttributeDict):
        ...

    def _init_end(self, hparams: AttributeDict, *args, **kwargs):
        ...

    def validate_attributes(self):
        ...


class DatasetMixin(RideMixin):
    input_shape: DataShape
    output_shape: DataShape

    def validate_attributes(self):
        assert type(getattr(self, "input_shape", None)) in {
            int,
            list,
            tuple,
        }, "Ride Dataset should define `input_shape` but none was found."
        assert type(getattr(self, "output_shape", None)) in {
            int,
            list,
            tuple,
        }, "Ride Dataset should define `output_shape` but none was found."

        for n in DatasetMixin.configs().names:
            assert some(
                self, f"hparams.{n}"
            ), "`self.hparams.{n}` not found in Dataset. Did you forget to include its `configs`?"

        for n in {"train_dataloader", "val_dataloader", "test_dataloader"}:
            assert some(
                self, n
            ), f"Ride Dataset should define `{n}` but none was found."

    @staticmethod
    def configs() -> Configs:
        return Configs().add(
            name="batch_size",
            type=int,
            default=16,
            strategy="constant",
            description="Batch size for dataset.",
        )


class ClassificationDataset(DatasetMixin):
    classes: List[str]

    def validate_attributes(self):
        DatasetMixin.validate_attributes(self)
        assert type(getattr(self, "classes", None)) in {
            list,
        }, "Ride ClassificationDataset should define `classes` but none was found."
