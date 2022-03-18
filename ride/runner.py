import os
from argparse import Namespace
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, List, Type

import torch
from pytorch_benchmark.benchmark import benchmark as pb_benchmark
from pytorch_benchmark.benchmark import warm_up as pb_warm_up
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.parsing import AttributeDict

from ride.core import RideDataset, RideModule
from ride.logging import (
    ExperimentLoggerCreator,
    ResultsLogger,
    add_experiment_logger,
    experiment_logger,
)
from ride.utils.logging import getLogger, process_rank
from ride.utils.utils import attributedict, flatten_dict, some_callable

EvalutationResults = Dict[str, float]

logger = getLogger(__name__)


def is_runnable(cls):
    assert issubclass(cls, RideModule)
    assert issubclass(cls, RideDataset)
    assert issubclass(cls, LightningModule)
    return True


class Runner:
    trained_model: RideModule

    def __init__(
        self,
        Module: Type[RideModule],
    ):
        assert is_runnable(Module)
        self.Module = Module

    def train(
        self,
        args: AttributeDict,
        trainer_callbacks: List[Callable] = [],
        tune_checkpoint_dir: str = None,
        experiment_logger: ExperimentLoggerCreator = experiment_logger,
    ) -> RideModule:
        args = attributedict(args)

        # Support for checkpoint loading in Tune
        # c.f. https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html#adding-checkpoints-to-the-pytorch-lightning-module
        if tune_checkpoint_dir:
            tune_ckpt = pl_load(
                os.path.join(tune_checkpoint_dir, "checkpoint"),
                map_location=lambda storage, loc: storage,
            )
            model = self.Module._load_model_state(tune_ckpt, hparams=args)

        else:
            model = self.Module(hparams=args)

        # Handle distributed backend modes in Lightning
        if not getattr(args, "distributed_backend", None):
            if model.hparams.num_gpus > 1:
                args.distributed_backend = "ddp"
            else:
                args.distributed_backend = None

        # Ensure callbacks are not stacked if multiple calls are made
        trainer_callbacks = trainer_callbacks.copy()

        # Prepare logger and callbacks
        trainer_callbacks.append(
            ModelCheckpoint(
                save_top_k=1,
                verbose=True,
                monitor=f"val/{args.optimization_metric}",  # Comment out when using pl.EvalResult
                mode=self.Module.metrics()[args.optimization_metric].value,
                save_last=True,
                every_n_train_steps=args.checkpoint_every_n_steps or None,
            )
        )
        logger.info(
            f"✅ Checkpointing on val/{args.optimization_metric} with optimisation direction {self.Module.metrics()[args.optimization_metric].value}"
        )
        if args.monitor_lr:
            trainer_callbacks.append(LearningRateMonitor(logging_interval="step"))

        _experiment_logger = experiment_logger(args.id, args.logging_backend)

        self.trainer = Trainer.from_argparse_args(
            Namespace(**args),
            logger=_experiment_logger,
            callbacks=trainer_callbacks,
        )
        _experiment_logger.log_hyperparams(dict(**model.hparams))

        # Load epoch state for Tune checkpoint
        if tune_checkpoint_dir:
            self.trainer.current_epoch = tune_ckpt["epoch"]  # type:ignore

        # Run hparam routines in Lightning
        if args.auto_scale_batch_size:
            # with temporary_parameter(self.trainer, "auto_lr_find", 0):
            self.trainer.tune(model)

        if args.auto_lr_find:
            lr_finder = self.trainer.tuner.lr_find(model, min_lr=1e-8, max_lr=1e-1)
            lr_suggestion = lr_finder.suggestion()
            logger.info(f"Suggested learning rate is {lr_suggestion}")
            model.hparams.learning_rate = lr_suggestion

            # Plot suggestion
            if process_rank == 0:
                fig = lr_finder.plot(suggest=True)
                lr_fig_path = Path(_experiment_logger.log_dir) / "lr_find.png"
                logger.info(f"Saving plot of learning rate sweep to {lr_fig_path}")
                lr_fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(lr_fig_path)

        self.trainer.fit(model)

        if Path(self.trainer.checkpoint_callback.best_model_path).is_file():
            self.best_model = self.Module.load_from_checkpoint(
                checkpoint_path=self.trainer.checkpoint_callback.best_model_path,
                hparams=model.hparams,
            )

        return model

    def evaluate(self, args: AttributeDict, mode="val") -> EvalutationResults:
        assert mode in {"val", "test"}
        args = attributedict(args)

        # Init model
        if hasattr(self, "best_model"):
            model = self.best_model
        else:
            model = self.Module(hparams=args)

        dataloaders = {"val": model.val_dataloader, "test": model.test_dataloader}

        # Init trainer
        base_logger = experiment_logger(args.id, args.logging_backend)
        results_logger = ResultsLogger(prefix=mode, save_to=base_logger.log_dir)
        logger = add_experiment_logger(base_logger, results_logger)
        if hasattr(self, "trainer"):
            self.trainer.logger = logger
        else:
            self.trainer = Trainer.from_argparse_args(Namespace(**args), logger=logger)

        base_logger.log_hyperparams(dict(**model.hparams))
        # Run eval
        self.trainer.test(model, dataloaders[mode]())

        results = results_logger.results

        # It seems that `trainer.test` actively sets the model dataloader; revert it
        model.val_dataloader = dataloaders["val"]
        model.test_dataloader = dataloaders["test"]

        return results

    def validate(self, args: AttributeDict) -> EvalutationResults:
        return self.evaluate(args, "val")

    def test(self, args: AttributeDict) -> EvalutationResults:
        return self.evaluate(args, "test")

    def train_and_val(
        self,
        args: AttributeDict,
        trainer_callbacks: List[Callable] = [],
        tune_checkpoint_dir: str = None,
        experiment_logger: ExperimentLoggerCreator = experiment_logger,
    ) -> EvalutationResults:
        self.train(args, trainer_callbacks, tune_checkpoint_dir, experiment_logger)
        return self.evaluate(args, mode="val")

    @staticmethod
    def static_train_and_val(
        Module: Type[RideModule],
        args: AttributeDict,
        trainer_callbacks: List[Callable] = [],
        tune_checkpoint_dir: str = None,
        experiment_logger: ExperimentLoggerCreator = experiment_logger,
    ) -> EvalutationResults:
        return Runner(Module).train_and_val(
            args, trainer_callbacks, tune_checkpoint_dir, experiment_logger
        )

    def profile_model(self, args: AttributeDict, num_runs: int = 100) -> Dict[str, Any]:
        if hasattr(self, "trained_model"):
            model = self.trained_model
        else:
            model = self.Module(hparams=args)

        sample = torch.randn(
            getattr(model.hparams, "batch_size", 1), *model.input_shape
        )

        results = pb_benchmark(
            model,
            sample,
            num_runs=num_runs,
            warm_up_fn=model.warm_up if some_callable(model, "warm_up") else pb_warm_up,
        )

        elogger = experiment_logger(args.id, args.logging_backend)
        elogger.log_hyperparams(dict(**model.hparams))
        elogger.log_metrics(
            {
                k: v
                for k, v in flatten_dict(results, sep="__").items()
                if isinstance(v, Number)
            }
        )

        return results

    def find_learning_rate(self):
        raise NotImplementedError()

    def find_batch_size(self):
        raise NotImplementedError()
