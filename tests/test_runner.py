import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import pytest
import pytorch_lightning as pl
import torch

from ride.core import RideModule
from ride.logging import experiment_logger
from ride.optimizers import SgdOptimizer
from ride.runner import Runner
from ride.utils.utils import AttributeDict

from .dummy_dataset import DummyDataLoader

pl.seed_everything(42)


class DummyModule(RideModule, DummyDataLoader, SgdOptimizer):
    def __init__(self, hparams):
        self.pred_layer = torch.nn.Linear(
            self.input_shape[0],  # from DummyDataLoader
            self.output_shape,  # from DummyDataLoader
        )
        self.loss = torch.nn.functional.mse_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_layer(x)


@pytest.fixture()  # scope="module"
def runner_and_args() -> Tuple[Runner, AttributeDict]:
    r = Runner(DummyModule)
    parser = r.Module.configs().add_argparse_args(ArgumentParser())
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

    return r, args


class TestRunner:
    def test_init(self):
        class NotARideModule:
            meaningoflife = 42

        with pytest.raises(AssertionError):
            r = Runner(NotARideModule)  # type:ignore

        r = Runner(DummyModule)
        assert r.Module is not None

    def test_train(self, runner_and_args: Tuple[Runner, AttributeDict]):
        """Test that training works"""
        runner, args = runner_and_args
        runner.train(args)

    def test_train_auto_scale_batch_size(
        self, runner_and_args: Tuple[Runner, AttributeDict]
    ):
        runner, args = runner_and_args
        args.auto_scale_batch_size = "power"
        runner.train(args)

    def test_train_auto_lr_find(self, runner_and_args: Tuple[Runner, AttributeDict]):
        runner, args = runner_and_args
        args.auto_lr_find = True
        runner.train(args)

    def test_validate(self, runner_and_args: Tuple[Runner, AttributeDict]):
        """Test that evaluation using validation set works
        Implicitely tests Runner.evaluate
        """
        runner, args = runner_and_args
        result = runner.validate(args)
        assert result["val/loss"]

    def test_test(self, runner_and_args: Tuple[Runner, AttributeDict]):
        """Test that evaluation using test set works
        Implicitely tests Runner.evaluate
        """
        runner, args = runner_and_args
        result = runner.test(args)
        assert result["test/loss"]

    def test_train_and_val(self, runner_and_args: Tuple[Runner, AttributeDict]):
        """Test that training and evaluation using validation set works"""
        runner, args = runner_and_args
        runner.train_and_val(args)

    def test_train_and_val_static(self, runner_and_args: Tuple[Runner, AttributeDict]):
        """Test that static version of training and evaluation using validation set works"""
        _, args = runner_and_args
        Runner.static_train_and_val(DummyModule, args)

    def test_profile_model(self, runner_and_args: Tuple[Runner, AttributeDict]):
        """Test that profiling model work"""
        runner, args = runner_and_args
        runner.profile_model(args, max_wait_seconds=1)

    def test_profile_dataset(self, runner_and_args: Tuple[Runner, AttributeDict]):
        """Test that profiling model work"""
        runner, args = runner_and_args

        # If dataset doesn't have a profile function, profiling won't work
        with pytest.raises(AssertionError):
            runner.profile_dataset(args)

        # TODO: test with profileable dataset

    # TODO
    # def test_find_learning_rate(self, runner_and_args: Tuple[Runner, AttributeDict]):
    #     """Test that automatic learning rate search works
    #     """
    #     runner, _ = runner_and_args
    #     runner.find_learning_rate()

    # TODO
    # def test_find_batch_size(self, runner_and_args: Tuple[Runner, AttributeDict]):
    #     """Test that automatic batch size search works
    #     """
    #     runner, _ = runner_and_args
    #     runner.find_batch_size()

    def test_teardown(self, runner_and_args: Tuple[Runner, AttributeDict]):
        runner, args = runner_and_args
        d = Path(experiment_logger(args.id).log_dir).parent
        shutil.rmtree(d)
        assert not d.exists()
