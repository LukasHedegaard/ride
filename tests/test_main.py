from ride import Main  # noqa: F401  # isort:skip
import torch

from typing import Tuple
import pytest
from ride.core import RideModule
from ride.main import Main
from ride.utils.utils import AttributeDict, attributedict
from ride.optimizers import AdamWOneCycleOptimizer
import logging
from .dummy_dataset import DummyDataLoader


class DummyModule(RideModule, DummyDataLoader, AdamWOneCycleOptimizer):
    def __init__(self, hparams):
        self.pred_layer = torch.nn.Linear(
            self.input_shape[0],  # from DummyDataLoader
            self.output_shape,  # from DummyDataLoader
        )
        self.loss = torch.nn.functional.mse_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_layer(x)


@pytest.fixture()  # scope="module"
def main_and_args() -> Tuple[Main, AttributeDict]:
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


class TestMain:
    def test_help(self, capsys, caplog):
        """Test that command help works"""

        caplog.clear()

        # Nothing is logged
        with caplog.at_level(logging.WARNING):
            Main(DummyModule).argparse([])

        assert len(caplog.messages) == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        # Help is printed
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
            "--profile_dataset",
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
        for msg in ["--loss", "--learning_rate", "--batch_size"]:
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

        with caplog.at_level(logging.INFO):
            m.main(args)

        for check in [
            "Running training",
            "val/loss",
            "saving model to",
            "Running evaluation on validation set",
            "val/epoch",
            "Running evaluation on test set",
            "test/epoch",
        ]:
            assert any([check in msg for msg in caplog.messages])


# def test_hparamsearch():
#     """Test that execution of hparamsearch and retreival of best hparams works"""
#     assert False


# def test_from_hparams_file():
#     """Test loading of hparams"""
#     assert False


# def test_profile_dataset():
#     """Test that profiling dataset works"""
#     assert False


# def test_profile_model():
#     """Test that profiling dataset works"""
#     assert False
