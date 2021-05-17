import platform
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import pytest
import pytorch_lightning as pl
import torch

from ride.core import Configs, RideModule
from ride.logging import experiment_logger
from ride.optimizers import SgdOptimizer
from ride.runner import Runner
from ride.utils.utils import AttributeDict

from .dummy_dataset import DummyRegressionDataLoader

pl.seed_everything(42)


class DummyModule(RideModule, DummyRegressionDataLoader, SgdOptimizer):
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
    args.logging_backend = "tensorboard"

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

    @pytest.mark.skip
    def test_train_auto_scale_batch_size(
        self, runner_and_args: Tuple[Runner, AttributeDict]
    ):
        runner, args = runner_and_args
        args.auto_scale_batch_size = "power"
        runner.train(args)

    @pytest.mark.skip
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

    def test_teardown(self, runner_and_args: Tuple[Runner, AttributeDict]):
        runner, args = runner_and_args
        d = Path(experiment_logger(args.id).log_dir).parent

        try:
            # May throw error on Windows:
            # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process
            shutil.rmtree(d)
        except Exception:
            pass

        if platform.system() != "Windows":
            assert not d.exists()
