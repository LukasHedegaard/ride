""" Dummy Dataset
Modified from
https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/datamodules/dummy_dataset.py#L5-L42
"""

from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ride.core import Configs, RideClassificationDataset, RideDataset
from ride.utils.utils import some


class MeanVarDataset(Dataset):
    def __init__(self, input_shape: Sequence[int], num_samples=10000):
        super().__init__()
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.output_shape = (2,)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.rand(self.input_shape)
        y = torch.stack(
            [
                x.mean(),
                x.var(),
            ]
        )
        sample = [x, y, idx // 2]
        return sample


class DummyRegressionDataLoader(RideDataset):
    @staticmethod
    def configs() -> Configs:
        c = Configs.collect(DummyRegressionDataLoader)
        c.add(
            name="input_shape",
            type=int,
            default=10,
            strategy="constant",
            description="Input shape for data.",
        )
        return c

    def validate_attributes(self):
        RideDataset.validate_attributes(self)
        for attr in DummyRegressionDataLoader.configs().names:
            assert some(
                self, f"hparams.{attr}"
            ), f"ClassificationLifecycle should define `hparams.{attr}` but none was found."

    def __init__(self, hparams):
        num_workers = 1
        self.input_shape = (hparams.input_shape,)
        self.output_shape = 2

        self._train_dataloader = DataLoader(
            MeanVarDataset(self.input_shape, num_samples=65),
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=num_workers > 1,
        )
        self._val_dataloader = DataLoader(
            MeanVarDataset(self.input_shape, num_samples=4),
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
        )
        self._test_dataloader = DataLoader(
            MeanVarDataset(self.input_shape, num_samples=10),
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        return self._test_dataloader


class OverOneDataset(Dataset):
    def __init__(self, input_shape: Sequence[int], num_samples=10000):
        super().__init__()
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.output_shape = (2,)
        self.classes = ["0", "1"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_shape)
        m = x.mean()
        y = (m > 0).to(torch.long)
        sample = [x, y, idx // 2]
        return sample


class DummyClassificationDataLoader(RideClassificationDataset):
    @staticmethod
    def configs() -> Configs:
        c = Configs.collect(DummyClassificationDataLoader)
        c.add(
            name="input_shape",
            type=int,
            default=10,
            strategy="constant",
            description="Input shape for data.",
        )
        return c

    def validate_attributes(self):
        RideClassificationDataset.validate_attributes(self)
        for attr in DummyRegressionDataLoader.configs().names:
            assert some(
                self, f"hparams.{attr}"
            ), f"ClassificationLifecycle should define `hparams.{attr}` but none was found."

    def __init__(self, hparams):
        num_workers = 1
        self.input_shape = (hparams.input_shape,)
        self.output_shape = 2
        self.classes = ["fee", "fei"]

        self._train_dataloader = DataLoader(
            OverOneDataset(self.input_shape, num_samples=65),
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=num_workers > 1,
        )
        self._val_dataloader = DataLoader(
            OverOneDataset(self.input_shape, num_samples=4),
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
        )
        self._test_dataloader = DataLoader(
            OverOneDataset(self.input_shape, num_samples=10),
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        return self._test_dataloader
