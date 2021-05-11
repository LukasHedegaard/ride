import logging
import pickle
import shutil
from pathlib import Path
from typing import Tuple

import pytest
import torch

from ride.core import RideModule
from ride.finetune import Finetunable
from ride.optimizers import AdamWCyclicLrOptimizer
from ride.utils.checkpoints import get_latest_checkpoint
from ride.utils.utils import AttributeDict

# from ride.finetune import Finetunable
from .dummy_dataset import DummyRegressionDataLoader

from ride import Main, Configs  # noqa: F401  # isort:skip


class DummyModule(
    RideModule, Finetunable, DummyRegressionDataLoader, AdamWCyclicLrOptimizer
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


class TestFinetuning:
    def test_finetune_from_weights(
        self, caplog, main_and_args: Tuple[Main, AttributeDict]
    ):
        """
        Test finetune_from_weights pickle and pyth suffixes
        """
        m, args = main_and_args
        args.train = True
        args.discriminative_lr_fraction = 0.5

        # Create a run to start with
        m.main(args)

        # Create checkpoints
        ckpt_cp = get_latest_checkpoint(m.log_dir)

        pth_cp = Path(m.log_dir) / "pretrained.pth"
        torch.save(m.runner.best_model.state_dict(), str(pth_cp))

        pth_cp_2 = Path(m.log_dir) / "pretrained_2.pth"
        torch.save(m.runner.best_model, str(pth_cp_2))

        pkl_cp = Path(m.log_dir) / "pretrained.pkl"
        with open(pkl_cp, "wb") as f:
            pickle.dump(m.runner.best_model.state_dict(), f)

        # Finetune from checkpoint
        for cp in [ckpt_cp, pth_cp, pth_cp_2, pkl_cp]:
            # for cp in [pth_cp]:
            m, args = default_main_and_args()  # Need to make new main
            args.finetune_from_weights = str(cp)
            args.test = True

            caplog.clear()
            with caplog.at_level(logging.INFO):
                m.main(args)

            assert len(caplog.messages) > 0
            assert any(
                [
                    f"Loading model weights from {str(cp)}" in msg
                    for msg in caplog.messages
                ]
            )

        # Clean up
        ckpt_cp.unlink()
        pth_cp.unlink()
        pkl_cp.unlink()

    def test_complex_finetuning(
        self, caplog, main_and_args: Tuple[Main, AttributeDict]
    ):
        """
        Test complex fine-tuning setup, including
        - finetune_from_weights
        - discriminative_lr_fraction
        - gradual_unfreeze
        """
        m, args = main_and_args
        args.train = True
        args.discriminative_lr_fraction = 0.5

        # Create a run to start with
        m.main(args)

        latest_checkpoint = get_latest_checkpoint(m.log_dir)
        assert latest_checkpoint.exists()
        checkpoint = Path("test_complex_finetuning.ckpt")
        shutil.copy(latest_checkpoint, checkpoint)

        # Finetune from checkpoint
        m, args = default_main_and_args()  # Need to make new main
        args.finetune_from_weights = str(checkpoint)
        args.unfreeze_layers_initial = 1
        args.unfreeze_epoch_step = 1
        args.unfreeze_from_epoch = 0
        args.train = True

        # One epoch, one unfrozen
        args.max_epochs = 1

        caplog.clear()
        with caplog.at_level(logging.INFO):
            m.main(args)

        assert len(caplog.messages) > 0
        assert any(["Unfreezing 1 layer(s)" in msg for msg in caplog.messages])
        assert not any(["Unfreezing 2 layer(s)" in msg for msg in caplog.messages])

        assert m.runner.trainer.model.l1.weight.requires_grad is False
        assert m.runner.trainer.model.l1.bias.requires_grad is False
        assert m.runner.trainer.model.l2.weight.requires_grad is True
        assert m.runner.trainer.model.l2.bias.requires_grad is True

        # Two epochs, both unfrozen
        m, args = default_main_and_args()  # Need to make new main
        args.finetune_from_weights = str(checkpoint)
        args.unfreeze_layers_initial = 1
        args.unfreeze_epoch_step = 1
        args.unfreeze_from_epoch = 0
        args.train = True
        args.max_epochs = 2
        # TODO: test with max_epochs > num layers

        caplog.clear()
        with caplog.at_level(logging.INFO):
            m.main(args)

        assert any(["Unfreezing 1 layer(s)" in msg for msg in caplog.messages])
        assert any(["Unfreezing 2 layer(s)" in msg for msg in caplog.messages])
        assert m.runner.trainer.model.l1.weight.requires_grad is True
        assert m.runner.trainer.model.l1.bias.requires_grad is True
        assert m.runner.trainer.model.l2.weight.requires_grad is True
        assert m.runner.trainer.model.l2.bias.requires_grad is True

        checkpoint.unlink()
