import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch

from ride.core import RideModule
from ride.finetune import Finetunable
from ride.optimizers import AdamWOneCycleOptimizer
from ride.utils.checkpoints import get_latest_checkpoint
from ride.utils.io import dump_json, dump_yaml, is_nonempty_file
from ride.utils.utils import AttributeDict, attributedict, temporary_parameter

# from ride.finetune import Finetunable
from .dummy_dataset import DummyRegressionDataLoader

from ride import Main, Configs  # noqa: F401  # isort:skip


class DummyModule(
    RideModule, Finetunable, DummyRegressionDataLoader, AdamWOneCycleOptimizer
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


def default_main_and_args() -> Tuple[Main, AttributeDict]:
    m = Main(DummyModule)
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


@pytest.fixture()  # scope="module"
def main_and_args() -> Tuple[Main, AttributeDict]:
    return default_main_and_args()


class TestMain:
    def test_help(self, capsys, caplog):
        """Test that command help works"""

        # Nothing is logged
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            Main(DummyModule).argparse([])

        assert caplog.messages == [] or (
            len(caplog.messages) == 1 and "Missing log" in caplog.messages[0]
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        # Help is printed
        caplog.clear()
        with pytest.raises(SystemExit), caplog.at_level(logging.WARNING):
            Main(DummyModule).argparse(["--help"])

        # Help message was neither logged nor in stderr
        assert len(caplog.messages) == 0
        captured = capsys.readouterr()
        assert captured.err == ""

        # Help is in stdout
        help_msg = captured.out
        assert len(help_msg) > 0

        # Flow args
        for msg in [
            "--hparamsearch",
            "--train",
            "--test",
            "--profile_model",
        ]:
            assert msg in help_msg

        # General args
        for msg in ["--id", "--logging_backend", "--optimization_metric"]:
            assert msg in help_msg

        # Pytorch Lightning args
        for msg in [
            "--logger",
            "--gpus",
            "--accumulate_grad_batches",
            "--max_epochs",
            "--limit_train_batches",
            "--precision",
            "--resume_from_checkpoint",
            "--benchmark",
            "--auto_lr_find",
            "--auto_scale_batch_size",
        ]:
            assert msg in help_msg

        # Module args
        for msg in ["--loss", "--learning_rate", "--batch_size", "--hidden_dim"]:
            assert msg in help_msg

    def test_default_id(self):
        """Test that a default id is given"""
        m = Main(DummyModule)
        parser = m.argparse(run=False)
        args, _ = parser.parse_known_args()
        assert args.id == "unnamed"

    def test_main_arg_types(self, main_and_args: Tuple[Main, AttributeDict]):
        """Test if passing both dict and AttributeDict work for main"""
        m, args = main_and_args
        m.main(args)  # Namespace
        m.main(attributedict(args))  # AttributeDict
        m.main(vars(args))  # Dict

    def test_train_val_test_combination(
        self, caplog, main_and_args: Tuple[Main, AttributeDict]
    ):
        """Test that training, validation, and test works"""
        m, args = main_and_args
        args.train = True
        args.validate = True
        args.test = True
        args.checkpoint_every_n_steps = 10

        # Trainer args
        args.limit_val_batches = 2
        args.limit_test_batches = 2

        with caplog.at_level(logging.INFO):
            m.main(args)

        for check in [
            "Running training",
            "val/loss",
            "saving model to",
            "Running evaluation on validation set",
            "val/epoch",
            "Running evaluation on test set",
            "test/loss",
        ]:
            assert any([check in msg for msg in caplog.messages])

        save_lines = [m for m in caplog.messages if "Saving" in m]

        # Check that result files exist
        hparams_path = Path(save_lines[0].split(" ")[2])
        assert is_nonempty_file(hparams_path)

        val_result_path = Path(save_lines[-2].split(" ")[2])
        assert is_nonempty_file(val_result_path)

        test_result_path = Path(save_lines[-1].split(" ")[2])
        assert is_nonempty_file(test_result_path)

        # Check that trainer args were passed
        assert m.runner.trainer.limit_val_batches == 2
        assert m.runner.trainer.limit_test_batches == 2

    def test_test_ensemble(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        """Test ensemble works"""
        m, args = main_and_args
        args.test = True
        args.test_ensemble = True

        with caplog.at_level(logging.INFO):
            m.main(args)

        for check in [
            "Running evaluation on test set using ensemble testing",
            "test/epoch",
        ]:
            assert any([check in msg for msg in caplog.messages])

    def test_from_hparams_file(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        """Test loading of hparams"""
        m, args = main_and_args
        args.validate = True

        for suffix, dump, hidden_dim in [
            ("yaml", dump_yaml, 256),
            ("json", dump_json, 512),
        ]:
            # Prep file
            hparams_path = Path(os.getcwd()) / f"test_dummy_module_hparams.{suffix}"
            dump(
                hparams_path,
                {"hidden_dim": hidden_dim, "batch_size": 4, "learning_rate": 0.01},
            )

            # Test
            caplog.clear()
            args.from_hparams_file = str(hparams_path)
            # args.train = True  # Triggers auto_scale_lr
            args.batch_size = 6
            with temporary_parameter(
                sys, "argv", [*sys.argv, "--batch_size", "6"]
            ), caplog.at_level(logging.INFO):
                m.main(args)

            len(caplog.messages) > 0
            for check in ["Scaling learning_rate from 0.01 to 0.015"]:
                any([check in msg for msg in caplog.messages])

            assert m.runner.trainer.model.hparams.hidden_dim == hidden_dim
            assert m.runner.trainer.model.hparams.batch_size == 6  # from command line
            assert m.runner.trainer.model.hparams.learning_rate == 0.015  # auto scaled

            # Cleanup
            hparams_path.unlink()

    def test_resume_from_checkpoint(self, main_and_args: Tuple[Main, AttributeDict]):
        """Test that resuming from checkpoint works with inferred checkpoint path works"""
        m, args = main_and_args
        args.train = True

        m.main(args)

        # From specific file (as usual in PyTorch Lightning)
        args.resume_from_checkpoint = str(get_latest_checkpoint(m.log_dir))
        assert ".ckpt" in args.resume_from_checkpoint
        m.main(args)  # Doesn't run because max_epochs was reached in checkpoint

        # Automatically pick lastest checkpoint in directory
        args.resume_from_checkpoint = m.log_dir
        assert ".ckpt" not in args.resume_from_checkpoint
        args.max_epochs = 2
        m.main(args)

    def test_profile_model(self, caplog, main_and_args: Tuple[Main, AttributeDict]):
        """Test that profiling model works"""
        m, args = main_and_args
        args.profile_model = True
        args.profile_model_num_runs = 10

        caplog.clear()

        # Nothing is logged
        with caplog.at_level(logging.INFO):
            m.main(args)

        assert len(caplog.messages) > 0

        for check in [
            "Results",
            "flops",
            "machine",
            "samples_per_second",
            "num_runs",
        ]:
            assert any([check in msg for msg in caplog.messages])

        # Check that results are saved and path is printed
        model_profile_path = Path(caplog.messages[-1].split(" ")[2])
        assert is_nonempty_file(model_profile_path)

    def test_hparamsearch(self, main_and_args: Tuple[Main, AttributeDict]):
        """Test that execution of hparamsearch and retreival of best hparams works"""

        # Prepare hparamspace file
        hparams_space_path = Path(os.getcwd()) / "test_dummy_module_hparamspace.yaml"
        hparams_space = {
            "hidden_dim": {
                "type": "int",
                "strategy": "choice",
                "choices": [128, 256, 512, 1024],
            }
        }
        dump_yaml(hparams_space_path, hparams_space)

        # Prep args
        m, args = main_and_args
        args.hparamsearch = True
        args.max_epochs = 1
        args.trials = 2
        args.from_hparam_space_file = str(hparams_space_path)

        # Test
        m.main(args)

        # Clean up
        hparams_space_path.unlink()
