<div align="center">
  <img src="docs/images/ride_logo.svg" width="350"><br>
</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/LukasHedegaard/ride/branch/main/graph/badge.svg?token=SJ59JOWNAC)](https://codecov.io/gh/LukasHedegaard/ride)


Training wheels, side rails, and helicopter parent for your Deep Learning projects in [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

```bash
pip install ride
```

## Zero-boilerplate AI research
Though [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) helps remove a lot of boilerplate code, writing and testing Deep Learning models still includes setting up
- __Trainer__
- __Checkpointing__
- __Metrics__
- __Train-val-test step methods__
- __Finetuning schemes__
- __Hyperparameter search__
- __Main function__
- __Command-line interface__ 
- __...__

This project is an audacious attempt at disposing of the remaining boiletplate by providing good battle-tested defaults with minimal coding.

Everything you find here is highly opinionated and was first and foremost an attempt at generalising personal research boiler-plate. 
On the other hand, it might be just right, and if not, it's highly extendable and forkable.
Suggestions and pull requests are always welcome!



## Programming model
Did you ever take a peek to the source code of the [LightningModule](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/lightning.py)?
This core class of the Pytorch Lightning library makes heavy use of _Mixins_ and _multiple inheritance_ to group functionalities and "inject" them in the LightningModule. 

In `ride` we build up our modules the same way, _mixing in_ functionality by inheriting from multiple base classes in our Model definition.


## Model definition
Below, we have the __complete__ code for a simple classifier on the MNIST dataset:
```python
# simple_classifier.py
import torch
import ride
import numpy as np
from .examples import MnistDataset


class SimpleClassifier(
    ride.RideModule,
    ride.Lifecycle, 
    ride.SgdOneCycleOptimizer, 
    ride.TopKAccuracyMetric(1,3),
    MnistDataset,
):
    def __init__(self, hparams):
        # `self.input_shape` and `self.output_shape` were injected via `ride.MnistDataset`
        self.l1 = torch.nn.Linear(np.prod(self.input_shape), self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, self.output_shape)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    @staticmethod
    def configs():
        c = ride.Configs()
        c.add(
            name="hidden_dim",
            type=int,
            default=128,
            strategy="choice",
            choices=[128, 256, 512, 1024],
            description="Number of hiden units.",
        )
        return c


if __name__ == "__main__":
    ride.Main(SimpleClassifier).argparse()

```
That's it! So what's going on, and aren't we missing a bunch of code?

All of the usual boiler-plate code has been _mixed in_ using multiple inheritance:
- `RideModule` is a base-module which includes `pl.LightningModule` and makes some behind-the-scenes python-magic work. For instance, it modifies your `__init__` function to automatically initiate all the mixins correctly.
- `ClassificationLifecycle` mixes in `training_step`, `validation_step`, and `test_step` alongside a `loss_fn` with [cross-entropy](https://pytorch.org/docs/stable/nn.functional.html#cross-entropy).
- `SgdOneCycleOptimizer` mixes in the `configure_optimizers` function with SGD and [OneCycleLR scheduler](https://pytorch.org/vision/0.8/datasets.html#torchvision.datasets.MNIST).
- `MnistDataset` mixes in `train_dataloader`, `val_dataloader`, and `test_dataloader` functions for the [MNIST dataset](https://github.com/LukasHedegaard/co-rider). Dataset mixins always provide `input_shape` and `output_shape` attributes, which are handy for defining the networking structure as seen in `__init__`.
- `TopKAccuracyMetric` adds top1acc and top3acc metrics, which can be used for checkpointing and benchmarking.

In addition to inheriting lifecycle functions etc., the mixins also add `configs` to your module (powered by [co-rider](https://github.com/LukasHedegaard/co-rider)). 
These define all of the configurable (hyper)parameters including their
- _type_ 
- _default_ value
- _description_ in plain text (reflected in command-line interface),
- _space_ defines accepted input range
- _strategy_ specifies how hyperparameter-search tackles the parameter. 

Configs specific to the SimpleClassifier can be added by overloading the `configs` methods as shown in the example.

The final piece of sorcery is the `Main` class, which adds a complete command-line interface.

## Command-line interface 💻

Let's check out the command-line interface:
```shell
$ python simple_classifier.py --help
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
  --hidden_dim HIDDEN_DIM {128, 256, 512, 1024}
                        Number of hiden units. (Defualt: 128)
  ...
```

Whew, there's a lot going on there (a bunch was even omitted ...)! 

First, there are flags for controlling the programme flow (e.g. whether to run hparamsearch or training), then some general parameters (id, seed, etc.), all the parameters from Pytorch Lightning, hparamsearch-related arguments, and finally the Module-specific arguments, which we defined in or mixed into the `SimpleClassifier`.

### Training and testing
```shell
$ python simple_classifier.py --train --test --learning_rate 0.01 --hidden_dim 256
```

### Hyperparameter optimization
If we want to perform __hyperparameter optimisation__  across four gpus, we can run:
```shell
$ python simple_classifier.py --hparamsearch --gpus 4
```
Curretly, we use [Ray Tune](https://docs.ray.io/en/master/tune.html) and the [ASHA](https://arxiv.org/abs/1810.05934) algorithm under the hood.


### Model profiling
You can check the __timing__ and __FLOPs__ of the model with:
```shell
$ python simple_classifier.py --profile_model
```


## Environment
Per default, `ride` projects are oriented around the current working directory and will save logs in the `~/logs folders`, and cache to `~/.cache`.

This behaviour can be overloaded by changing of the following environment variables (defaults noted):
```bash
ROOT_PATH="~/"
CACHE_PATH=".cache"
DATASETS_PATH="datasets"  # Dir relative to ROOT_PATH
LOGS_PATH="logs"          # Dir relative to ROOT_PATH
RUN_LOGS_PATH="run_logs"  # Dir relative to LOGS_PATH
TUNE_LOGS_PATH="tune_logs"# Dir relative to LOGS_PATH
LOG_LEVEL="INFO"          # One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
```


## Documentation
Coming up. 
For now, a look to the source code should get you there.

<!-- ## Compatibility with Pytorch Lightning
The framework is built on top of Pytorch Lightning, and we strive to be fully compatible with the newest versions.
That said, Pytorch Lightning is still evolving rapidly, so things may occationally break on our side.
 -->



## Bibtex
If you end up using `ride` for your research and feel like citing it, here's a BibTex:

```bibtex
@article{hedegaard2021ride,
  title={Ride},
  author={Lukas Hedegaard},
  journal={GitHub. Note: https://github.com/LukasHedegaard/ride},
  year={2021}
}
```