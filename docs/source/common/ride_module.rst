.. role:: hidden
    :class: hidden-section

.. testsetup:: *

    from pytorch_lightning.core.lightning import LightningModule
    from ride.core import RideModule
    import torch
    from .examples import MnistDataset

.. _ride_module:

RideModule
===============
The :class:`~ride.RideModule` works in conjunction with the :class:`~pytorch_lightning.LightningModule`, to add functionality to a plain :class:`~torch.nn.Module`.
While :class:`~LightningModule` adds a bunch of structural code, that integrates with the :class:`~pytorch_lightning.Trainer`, the :class:`~ride.RideModule` provides good defaults for

- Train loop - :meth:`training_step`
- Validation loop - :meth:`validation_step`
- Test loop - :meth:`test_step`
- Optimizers - :meth:`configure_optimizers`

The only things left to be defined are

- Initialisation - :meth:`__init__`.
- Network forward pass - :meth:`forward`.
- :doc:`Dataset <../common/ride_datasets>`

The following thus constitutes a fully functional Neural Network module, which (when integrated with :class:`ride.Main`) provides full functionality for training, testing, hyperparameters search, profiling , etc., via a command line interface.

.. code-block:: python

    from ride import RideModule
    from .examples.mnist_dataset import MnistDataset

    class MyRideModule(RideModule, MnistDataset):
        def __init__(self, hparams):
            hidden_dim = 128
            # `self.input_shape` and `self.output_shape` were injected via `MnistDataset`
            self.l1 = torch.nn.Linear(np.prod(self.input_shape), hidden_dim)
            self.l2 = torch.nn.Linear(hidden_dim, self.output_shape)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.l1(x))
            x = torch.relu(self.l2(x))
            return x


Configs
-------
Out of the box, a wide selection parameters are integrated into `self.hparams` through :class:`ride.Main`. 
These include all the :class:`pytorch_lightning.Trainer` options, as well as configs in :meth:`ride.lifecycle.Lifecycle.configs`, the selected optimizer (default: :meth:`ride.optimizers.SgdOptimizer.configs`).

User-defined hyperparameters, which are reflected `self.hparams`, the command line interface, and hyperparameter serach space (by selection of `choices` and `strategy`), are easily defined by defining a `configs` method :class:`MyRideModule`:

.. code-block:: python

    @staticmethod
    def configs() -> ride.Configs:
        c = ride.Configs()
        c.add(
            name="hidden_dim",
            type=int,
            default=128,
            strategy="choice",
            choices=[128, 256, 512, 1024],
            description="Number of hidden units.",
        )
        return c

The configs package is also available seperately in the `Co-Rider package <https://github.com/LukasHedegaard/co-rider>`_.


Advanced behavior overloading
-----------------------------

Lifecycle methods
^^^^^^^^^^^^^^^^^

Naturally, the :meth:`training_step`, :meth:`validation_step`, and :meth:`test_step` can still be overloaded if complex computational schemes are required. 
In that case, ending the function with :meth:`common_step` will ensure that loss computation and collection of metrics still works as expected:

.. code-block:: python

    def training_step(self, batch, batch_idx=None):
        x, target = batch
        pred = self.forward(x)  # replace with complex interaction
        return self.common_step(pred, target, prefix="train/", log=True)


Loss
^^^^

By default, :class:`~ride.RideModule` automatically integrates the loss functions in :class:`torch.nn.functional` (set by command line using the "--loss" flag).
If other options are needed, one can define the :meth:`self.loss` in the module.

.. code-block:: python

    def loss(self, pred, target):
        return my_exotic_loss(pred, target)


Optimizer
^^^^^^^^^

The :class:`~ride.SgdOptimizer` is added automatically if no other :class:`~ride.optimizer.Optimizer` is found and :meth:`configure_optimizers` is not manually defined.
Other optimizers can thus be specified by using either Mixins:

.. code-block:: python

    class MyModel(
        ride.RideModule,
        ride.AdamWOneCycleOptimizer
    ):
        def __init__(self, hparams):
            ...

or function overloading:

.. code-block:: python

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

While the specifying parent Mixins automatically adds :meth:`ride.AdamWOneCycleOptimizer.configs` and hparams, the function overloading approach must be supplemented with a :meth:`configs` methods in order to reflect the parameter in the command line tool and hyperparameter search space.

.. code-block:: python

    @staticmethod
    def configs() -> ride.Configs:
        c = ride.Configs()
        c.add(
            name="learning_rate",
            type=float,
            default=0.1,
            choices=(1e-6, 1),
            strategy="loguniform",
            description="Learning rate.",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer



:doc:`Next <../common/ride_datasets>`, we'll see how to specify dataset.