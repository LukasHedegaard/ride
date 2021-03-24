.. role:: hidden
    :class: hidden-section

.. testsetup:: *

    from ride.main import Main
    from ride.core import RideModule

.. _main:

Main
====

The :class:`~ride.Main` class wraps a :class:`~ride.RideModule` to supply a fully functional command-line interface which includes

- Training ("--train")
- Evaluation on validation set ("--validate")
- Evaluation on test set ("--test")
- Logger integration ("--logging_backend")
- Hyperparameter search ("--hparamsearch")
- Hyperparameter file loading ("--from_hparams_file")
- Profiling of model timing, flops, and params ("--profile_model")
- Checkpointing
- Checkpoint loading ("--resume_from_checkpoint")


Example
-------

All it takes to get a working CLI is to add the following to the bottom of a file:

.. code-block:: python

    # my_ride_module.py
    
    import numpy as np
    from ride import RideModule, TopKAccuracyMetric
    from .examples.mnist_dataset import MnistDataset

    class MyRideModule(RideModule, TopKAccuracyMetric(1,3), MnistDataset):
        def __init__(self, hparams):
            # `self.input_shape` and `self.output_shape` were injected via `MnistDataset`
            self.lin = torch.nn.Linear(np.prod(self.input_shape), self.output_shape)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.lin(x))
            return x

    ride.Main(MyRideModule).argparse()  # <-- Add this

and executing from the command line:

.. code-block:: bash

    >> python my_ride_module.py --train --test --max_epochs 1 --id my_first_run

    lightning: Global seed set to 123
    ride: Running on host d40049
    ride: ⭐️ View project repository at https://github.com/username/ride/tree/hash
    ride: Run data is saved locally at /Users/username/project_folder/logs/run_logs/my_first_run/version_0
    ride: Logging using Tensorboard
    ride: 🚀 Running training
    ride: Checkpointing on val/loss with optimisation direction min
    lightning: GPU available: False, used: False
    lightning: TPU available: None, using: 0 TPU cores
    lightning: 
    | Name | Type   | Params
    --------------------------------
    0 | l1   | Linear | 100 K 
    1 | l2   | Linear | 1.3 K 
    --------------------------------
    101 K     Trainable params
    0         Non-trainable params
    101 K     Total params
    0.407     Total estimated model params size (MB)
    Epoch 0: 100%|███████████████| 3751/3751 [00:16<00:00, 225.44it/s, loss=0.762, v_num=0, step_train/loss=0.899]
    lightning: Epoch 0, global step 3437: val/loss reached 0.90666 (best 0.90666), saving model to "/Users/username/project_folder/logs/run_logs/my_first_run/version_0/checkpoints/epoch=0-step=3437.ckpt" as top 1                                                 
    Epoch 1: 100%|███████████████| 3751/3751 [00:17<00:00, 210.52it/s, loss=0.581, v_num=1, step_train/loss=0.0221]
    lightning: Epoch 1, global step 3437: val/loss reached 0.61922 (best 0.61922), saving model to "/Users/username/project_folder/logs/run_logs/my_first_run/version_0/checkpoints/epoch=1-step=6875.ckpt" as top 1                                                 
    lightning: Saving latest checkpoint...
    ride: 🚀 Running evaluation on test set
    Testing: 100%|███████████████| 625/625 [00:01<00:00, 432.69it/s]
    --------------------------------------------------------------------------------
    ride: Results:
    test/epoch: 0.000000000
    test/loss: 0.889312625
    test/top1acc: 0.739199996
    test/top3acc: 0.883000016

    ride: Saving /Users/username/project_folder/ride/logs/my_first_run/version_0/evaluation/test_results.yaml


Help
-------

The best way to explore all the options available is to run the "--help"

.. code-block:: bash

    >> python my_ride_module.py --help

    ...

    Flow:
    Commands that control the top-level flow of the programme.

    --hparamsearch        Run hyperparameter search. The best hyperparameters
                            will be used for subsequent lifecycle methods
    --train               Run model training
    --validate            Run model evaluation on validation set
    --test                Run model evaluation on test set
    --profile_model       Profile the model

    General:
    Settings that apply to the programme in general.

    --id ID               Identifier for the run. If not specified, the current
                            timestamp will be used (Default: 202101011337)
    --seed SEED           Global random seed (Default: 123)
    --logging_backend {tensorboard,wandb}
                            Type of experiment logger (Default: tensorboard)
    ...

    Pytorch Lightning:
    Settings inherited from the pytorch_lightning.Trainer
    ...
    --gpus GPUS           number of gpus to train on (int) or which GPUs to
                            train on (list or str) applied per node
    ...

    Hparamsearch:
    Settings associated with hyperparameter optimisation
    ...

    Module:
    Settings associated with the Module
    --loss {mse_loss,l1_loss,nll_loss,cross_entropy,binary_cross_entropy,...}
                            Loss function used during optimisation. 
                            (Default: cross_entropy)
    --batch_size BATCH_SIZE
                            Dataloader batch size. (Default: 64)
    --num_workers NUM_WORKERS
                            Number of CPU workers to use for dataloading.
                            (Default: 10)
    --learning_rate LEARNING_RATE
                            Learning rate. (Default: 0.1)
    --weight_decay WEIGHT_DECAY
                            Weight decay. (Default: 1e-05)
    --momentum MOMENTUM   Momentum. (Default: 0.9)
    ...