# from ride.profile import Profileable
import inspect
from abc import ABC
from typing import Any, List, Sequence, Union

import pytorch_lightning as pl
from corider import Configs as _Configs
from supers import supers
from torch.utils.data import DataLoader

from ride.utils.logging import getLogger
from ride.utils.utils import (
    AttributeDict,
    DictLike,
    attributedict,
    merge_attributedicts,
    missing_or_not_in_other,
    name,
    some,
)

logger = getLogger(__name__)


DataShape = Union[int, Sequence[int], Sequence[Sequence[int]]]


class Configs(_Configs):
    @staticmethod
    def collect(cls: "RideModule") -> "Configs":
        c: Configs = sum([c.configs() for c in cls.__bases__ if hasattr(c, "configs")])  # type: ignore
        return c

    def default_values(self):
        return attributedict({k: v.default for k, v in self.values.items()})


def _init_subsubclass(cls):
    orig_init = cls.__init__

    def init(self, hparams: DictLike, *args, **kwargs):
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
        from ride.lifecycle import Lifecycle

        cls.__bases__ = (
            *cls.__bases__[:-1],
            Lifecycle,
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

    def init(self, hparams: DictLike = {}, *args, **kwargs):
        pl.LightningModule.__init__(self)
        self.hparams = merge_attributedicts(self.configs().default_values(), hparams)
        sself = (
            self
            if type(self).__bases__[0] == RideModule
            else supers(self)._superclasses[0]
        )
        supers(sself)[1:-1].__init__(self.hparams)
        apply_init_args(orig_init, self, self.hparams, *args, **kwargs)
        supers(sself).on_init_end(self.hparams, *args, **kwargs)
        supers(self).validate_attributes()

    cls.__init__ = init

    # Monkeypatch derived module configs
    orig_configs = getattr(cls, "configs", None)

    @staticmethod
    def configs():
        c = Configs.collect(cls)
        if orig_configs:
            c += orig_configs()
        return c

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

    @classmethod
    def with_dataset(cls, ds: "Dataset"):
        # @staticmethod
        # def configs():
        #     return ds.configs() + cls.configs()

        DerivedRideModule = type(
            f"{name(cls)}With{name(ds)}", cls.__bases__, dict(cls.__dict__)
        )

        new_bases = [b for b in cls.__bases__ if not issubclass(b, Dataset)]
        old_dataset = [b for b in cls.__bases__ if issubclass(b, Dataset)]
        assert len(old_dataset) <= 1, "`RideModule` should only have one `Dataset`"
        if old_dataset and issubclass(old_dataset[0], ClassificationDataset):
            assert issubclass(
                ds, ClassificationDataset
            ), "A `ClassificationDataset` should be replaced by a `ClassificationDataset`"
        new_bases.insert(-1, ds)
        DerivedRideModule.__bases__ = tuple(new_bases)

        return DerivedRideModule


class RideMixin(ABC):
    def __init__(self, hparams: AttributeDict, *args, **kwargs):
        ...

    def on_init_end(self, hparams: AttributeDict, *args, **kwargs):
        ...

    def validate_attributes(self):
        ...


class Dataset(RideMixin):
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

        for n in Dataset.configs().names:
            assert some(
                self, f"hparams.{n}"
            ), "`self.hparams.{n}` not found in Dataset. Did you forget to include its `configs`?"

        for n in {"train_dataloader", "val_dataloader", "test_dataloader"}:
            assert some(
                self, n
            ), f"Ride Dataset should define `{n}` but none was found."

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="batch_size",
            type=int,
            default=16,
            strategy="constant",
            description="Batch size for dataset.",
        )
        c.add(
            name="num_workers",
            type=int,
            default=0,
            strategy="constant",
            description="Number of workers in dataloader.",
        )
        return c


class ClassificationDataset(Dataset):
    classes: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def validate_attributes(self):
        Dataset.validate_attributes(self)
        assert type(getattr(self, "classes", None)) in {
            list,
        }, "Ride ClassificationDataset should define `classes` but none was found."

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        assert some(
            self, "datamodule.train_dataloader"
        ), f"{name(self)} should either have a `self.datamodule: pl.LightningDataModule` or overload the `train_dataloader` function."
        return self.datamodule.train_dataloader

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        assert some(
            self, "datamodule.val_dataloader"
        ), f"{name(self)} should either have a `self.datamodule: pl.LightningDataModule` or overload the `val_dataloader` function."
        return self.datamodule.val_dataloader

    def test_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        assert some(
            self, "datamodule.test_dataloader"
        ), f"{name(self)} should either have a `self.datamodule: pl.LightningDataModule` or overload the `test_dataloader` function."
        return self.datamodule.test_dataloader
