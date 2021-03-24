.. role:: hidden
    :class: hidden-section

.. testsetup:: *

    from ride.main import Main
    from ride.core import RideModule, RideDataset, RideClassificationDataset
    import pytorch_lightning as pl

.. _datasets:

Datasets
========

In `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_, datasets can be integrated by overloading dataloader functions in the :class:`~pl.LightningModule`:

- :func:`train_dataloader`
- :func:`val_dataloader`
- :func:`test_dataloader`

This is exactly what a :class:`~RideDataset` does. 
In addition, it adds :code:`num_workers` and :code:`batch_size` :code:`configs`  as well as :code:`self.input_shape` and :code:`self.output_shape` tuples (which are very handy for computing layer shapes).

For classification dataset, the :class:`~RideClassificationDataset` expects a list of class-names defined in :code:`self.classes` and provides a :code:`self.num_classes` attribute.
:code:`self.classes` are then used plotting, e.g. if "--test_confusion_matrix True" is specified in the CLI.

In order to define a :class:`~RideDataset`, one can either define the :func:`train_dataloader`, :func:`val_dataloader`, :func:`test_dataloader` and functions or assign a :class:`~pl.LightningDataModule` to :code:`self.datamodule` as seen here:
  
.. code-block:: python

    from ride.core import AttributeDict, RideClassificationDataset, Configs
    from ride.utils.env import DATASETS_PATH
    import pl_bolts

    class MnistDataset(RideClassificationDataset):

        @staticmethod
        def configs():
            c = Configs.collect(MnistDataset)
            c.add(
                name="val_split",
                type=int,
                default=5000,
                strategy="constant",
                description="Number samples from train dataset used for val split.",
            )
            c.add(
                name="normalize",
                type=int,
                default=1,
                choices=[0, 1],
                strategy="constant",
                description="Whether to normalize dataset.",
            )
            return c

        def __init__(self, hparams: AttributeDict):
            self.datamodule = pl_bolts.datamodules.MNISTDataModule(
                data_dir=DATASETS_PATH,
                val_split=self.hparams.val_split,
                num_workers=self.hparams.num_workers,
                normalize=self.hparams.normalize,
                batch_size=self.hparams.batch_size,
                seed=42,
                shuffle=True,
                pin_memory=self.hparams.num_workers > 1,
                drop_last=False,
            )
            self.output_shape = 10
            self.classes = list(range(10))
            self.input_shape = self.datamodule.dims


Changing dataset
----------------

Though the dataset is specified at module definition, we can change the dataset using :meth:`~RideModule.with_dataset`.
This is especially handy for experiments using a single module over multiple datasets:

.. code-block:: python

    MyRideModuleWithMnistDataset = MyRideModule.with_dataset(MnistDataset)
    MyRideModuleWithCifar10Dataset = MyRideModule.with_dataset(Cifar10Dataset)
    ...


:doc:`Next <../common/main>`, we'll cover how the :class:`~ride.RideModule` integrates with :class:`~ride.Main`.