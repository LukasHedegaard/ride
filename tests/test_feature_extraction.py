from collections import OrderedDict
from typing import Tuple

import pytest
import torch

from ride import Configs, Main
from ride.core import RideModule

# from ride.feature_extraction import FeatureExtractable
from ride.optimizers import SgdOptimizer
from ride.utils.utils import AttributeDict

from .dummy_dataset import DummyRegressionDataLoader


class ExDummyModule(
    RideModule,
    DummyRegressionDataLoader,
    SgdOptimizer,
    # FeatureExtractable, # Not needed: FeatureVisualisable is part of core and inherits the functionality
):
    def __init__(self, hparams):
        self.l1 = torch.nn.Linear(self.input_shape[0], self.hparams.hidden_dim)
        self.seq1 = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
        )
        self.seq2 = torch.nn.Sequential(
            OrderedDict(
                [("l3", torch.nn.Linear(self.hparams.hidden_dim, self.output_shape))]
            )
        )
        self.loss = torch.nn.functional.mse_loss

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.seq1(x))
        x = torch.relu(self.seq2(x))
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
def main_and_args() -> Tuple[Main, AttributeDict]:
    m = Main(ExDummyModule)
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
    def test_bad_name(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        m, args = main_and_args
        args.test = True
        args.extract_features_after_layer = "badname"

        with pytest.raises(AssertionError) as e:
            m.main(args)

        assert e.value.args[0] == (
            "Invalid `extract_features_after_layer` (badname). "
            "Available layers are: ['l1', 'seq1', 'seq1.0', 'seq2', 'seq2.l3']"
        )

    def test_shallow(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        m, args = main_and_args
        args.test = True
        args.extract_features_after_layer = "l1"

        m.main(args)

        assert len(m.runner.trainer.model.extracted_features) > 1

    def test_deep(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        m, args = main_and_args
        args.test = True

        args.extract_features_after_layer = "seq1.0"
        m.main(args)

        assert len(m.runner.trainer.model.extracted_features) > 1

        args.extract_features_after_layer = "seq2.l3"
        m.main(args)
        assert len(m.runner.trainer.model.extracted_features) > 1
