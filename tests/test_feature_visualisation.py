import logging
from typing import Tuple

import pytest
import torch

from ride import Configs, Main
from ride.core import RideModule
from ride.feature_visualisation import FeatureVisualisable
from ride.optimizers import SgdOptimizer
from ride.utils.utils import AttributeDict

from .dummy_dataset import DummyClassificationDataLoader, DummyRegressionDataLoader


class VisDummyModule(
    RideModule, FeatureVisualisable, DummyClassificationDataLoader, SgdOptimizer
):
    def __init__(self, hparams):
        self.conv = torch.nn.Conv2d(3, 5, (3, 3))
        self.lin = torch.nn.Linear(45, self.output_shape)

    def forward(self, x):
        x = torch.relu(self.conv(x.reshape(-1, 3, 5, 5)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.lin(x))
        return x

    @staticmethod
    def configs():
        c = Configs()
        c.add(
            name="input_shape",
            type=int,
            default=3 * 5 * 5,
            strategy="constant",
            description="Input shape for data.",
        )
        return c


@pytest.fixture()  # scope="module"
def main_and_args() -> Tuple[Main, AttributeDict]:
    m = Main(VisDummyModule)
    parser = m.argparse(run=False)
    args, _ = parser.parse_known_args()
    args.max_epochs = 1
    args.gpus = 0
    args.checkpoint_callback = True
    args.optimization_metric = "loss"
    args.id = "automated_test"
    args.test_ensemble = 0
    args.checkpoint_every_n_steps = 0
    args.monitor_lr = 0
    args.auto_lr_find = 0
    args.auto_scale_batch_size = 0
    args.num_workers = 1
    args.batch_size = 4

    return m, args


class VisDummyModule2(
    RideModule, FeatureVisualisable, DummyRegressionDataLoader, SgdOptimizer
):
    def __init__(self, hparams):
        self.l1 = torch.nn.Linear(self.input_shape[0], self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, self.output_shape)
        self.loss = torch.nn.functional.mse_loss

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    @staticmethod
    def configs():
        c = Configs()
        c.add(
            name="hidden_dim",
            type=int,
            default=128,
            strategy="choice",
            choices=[128, 256, 512, 1024],
            description="Number of hiden units.",
        )
        return c


@pytest.fixture()  # scope="module"
def main_and_args2() -> Tuple[Main, AttributeDict]:
    m = Main(VisDummyModule2)
    parser = m.argparse(run=False)
    args, _ = parser.parse_known_args()
    args.max_epochs = 1
    args.gpus = 0
    args.checkpoint_callback = True
    args.optimization_metric = "loss"
    args.id = "automated_test"
    args.test_ensemble = 0
    args.checkpoint_every_n_steps = 0
    args.monitor_lr = 0
    args.auto_lr_find = 0
    args.auto_scale_batch_size = 0
    args.num_workers = 1
    args.batch_size = 4

    return m, args


class TestFeatureExtraction:
    def test_2d_feat_pca(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        m, args = main_and_args
        args.test = True
        args.extract_features_after_layer = "conv"
        args.visualise_features = "pca"

        with caplog.at_level(logging.DEBUG):
            m.main(args)

        for check in [
            "features/test/conv.npy",
            "features/test/conv_pca.npy",
            "test_conv_pca.png",
        ]:
            assert any([check in msg for msg in caplog.messages])

    def test_1d_feat_tse(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        m, args = main_and_args
        args.test = True
        args.extract_features_after_layer = "lin"
        args.visualise_features = "tsne"

        with caplog.at_level(logging.DEBUG):
            m.main(args)

        for check in [
            "features/test/lin.npy",
            "features/test/lin_tsne.npy",
            "test_lin_tsne.png",
        ]:
            assert any([check in msg for msg in caplog.messages])

    def test_1d_feat_umap(self, caplog, main_and_args2: Tuple[Main, AttributeDict]):
        m, args = main_and_args2
        args.test = True
        args.extract_features_after_layer = "l2"
        args.visualise_features = "umap"

        with caplog.at_level(logging.DEBUG):
            m.main(args)

        for check in [
            "features/test/l2.npy",
            "features/test/l2_umap.npy",
            "test_l2_umap.png",
        ]:
            assert any([check in msg for msg in caplog.messages])
