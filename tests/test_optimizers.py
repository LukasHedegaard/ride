from argparse import ArgumentParser

import torch

from ride import optimizers
from ride.core import Configs, RideModule

from .dummy_dataset import DummyRegressionDataLoader


def apply_standard_args(args):
    args.max_epochs = 3
    args.gpus = 0
    args.checkpoint_callback = True
    args.performance_metric = "loss"
    args.id = "automated_test"
    args.test_ensemble = 0
    args.loss = "mse_loss"
    return args


def dummy_module_with_optimimzer(Optimizer):
    class DummyModule(RideModule, DummyRegressionDataLoader, Optimizer):
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

    return DummyModule


def test_optimizers():
    for Optimizer in [
        optimizers.SgdOptimizer,
        optimizers.AdamWOptimizer,
        optimizers.SgdReduceLrOnPlateauOptimizer,
        optimizers.AdamWReduceLrOnPlateauOptimizer,
        optimizers.SgdOneCycleOptimizer,
        optimizers.SgdCyclicLrOptimizer,
        optimizers.AdamWCyclicLrOptimizer,
        optimizers.AdamWOneCycleOptimizer,
    ]:
        DummyModule = dummy_module_with_optimimzer(Optimizer)
        parser = DummyModule.configs().add_argparse_args(ArgumentParser())
        args, _ = parser.parse_known_args()
        args = apply_standard_args(args)
        module = DummyModule(args)

        ret = module.configure_optimizers()
        if type(ret) == tuple:
            [optimizer], [scheduler] = ret
            assert issubclass(type(optimizer), torch.optim.Optimizer)
            assert hasattr(scheduler, "step")
        elif type(ret) == dict:
            optimizer = ret["optimizer"]
            scheduler = ret["lr_scheduler"]
            monitor = ret["monitor"]
            assert issubclass(type(optimizer), torch.optim.Optimizer)
            assert hasattr(scheduler, "step")
            assert type(monitor) == str
        else:
            assert issubclass(type(ret), torch.optim.Optimizer)

        assert ret is not None
