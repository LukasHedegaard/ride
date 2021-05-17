import logging
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from ride.core import AttributeDict, RideMixin, RideModule
from ride.optimizers import SgdOptimizer
from ride.utils.utils import DictLike

from .dummy_dataset import DummyRegressionDataLoader

pl.seed_everything(42)


class Mixin1(RideMixin):
    def __init__(self, hparams: AttributeDict):
        self.hparams.msg.append("Mixin1.__init__")

    def on_init_end(self, hparams, *args, **kwargs):
        self.hparams.msg.append("Mixin1.on_init_end")


class Mixin2(RideMixin):
    def __init__(self, hparams: AttributeDict):
        self.hparams.msg.append("Mixin2.__init__")

    def on_init_end(self, hparams, *args, **kwargs):
        self.hparams.msg.append("Mixin2.on_init_end")


class InitOrderModule(
    RideModule,
    Mixin1,
    Mixin2,
):
    def __init__(self, hparams: DictLike):
        self.hparams.msg.append("InitOrderModule.__init__")
        self.input_shape = (1,)
        self.output_shape = (1,)


def test_init_order():

    module = InitOrderModule({"msg": []})

    assert module.hparams.msg == [
        "Mixin1.__init__",
        "Mixin2.__init__",
        "InitOrderModule.__init__",
        "Mixin1.on_init_end",
        "Mixin2.on_init_end",
    ]


class DummyModule(
    RideModule,
    SgdOptimizer,
    DummyRegressionDataLoader,
):
    # Not needed:
    # @staticmethod
    # def configs() -> Configs:
    #     return Configs.collect(DummyModule)

    def __init__(self):
        self.lin = torch.nn.Linear(
            self.input_shape[0],  # from DummyRegressionDataLoader
            self.output_shape,  # from DummyRegressionDataLoader
        )
        # Alternative way of specifying loss:
        self.loss = torch.nn.functional.mse_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_init_only_self():
    """Test if modules initialises if"""
    DummyModule()


def test_training_step_added_automatically():
    """Test if modules adds lifecycle automatically"""

    parser = DummyModule.configs().add_argparse_args(ArgumentParser())
    args, _ = parser.parse_known_args()
    args = apply_standard_args(args)
    module = DummyModule(args)
    assert module.loss.__name__ == "mse_loss"
    batch_size = 2
    x = torch.ones((batch_size, *module.input_shape), dtype=torch.float)
    y = torch.ones((batch_size, module.output_shape), dtype=torch.float)
    step_output = module.training_step((x, y), batch_idx=0)

    assert type(step_output["loss"]) is torch.Tensor
    assert step_output["pred"].shape == y.shape
    assert torch.equal(step_output["target"], y)


def apply_standard_args(args):
    args.max_epochs = 3
    args.gpus = 0
    args.checkpoint_callback = True
    args.performance_metric = "loss"
    args.id = "automated_test"
    args.test_ensemble = 0
    args.loss = "mse_loss"
    return args


def test_module_with_no_forward_warns(caplog):
    """Test if modules initialises when hparams is also given"""
    caplog.clear()
    with caplog.at_level(logging.WARNING):

        class DummyModuleNoForward(
            RideModule,
            SgdOptimizer,
            DummyRegressionDataLoader,
        ):
            def __init__(self):
                self.lin = torch.nn.Linear(
                    self.input_shape[0],  # from DummyRegressionDataLoader
                    self.output_shape,  # from DummyRegressionDataLoader
                )

            # Missing on purpose:
            # def forward(self, x: torch.Tensor) -> torch.Tensor:
            #     return self.lin(x)

    assert len(caplog.messages) == 1
    assert "forward" in caplog.text
