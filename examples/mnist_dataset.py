from ride import getLogger
from ride.core import AttributeDict, Configs, RideClassificationDataset
from ride.utils.env import DATASETS_PATH

logger = getLogger(__name__)

try:
    import pl_bolts  # noqa: F401
    import torchvision  # noqa: F401
except ImportError:
    logger.error(
        "To run the `mnist_dataset.py` example, first install its dependencies: "
        "`pip install pytorch-lightning-bolts torchvision`"
    )


class MnistDataset(RideClassificationDataset):
    """
    Example Mnist Dataset
    Modified from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_datamodule.py
    """

    @staticmethod
    def configs():
        c = Configs.collect(MnistDataset)
        c.add(
            name="val_split",
            type=int,
            default=5000,
            strategy="constant",
            description="Number samples from train dataset used for validation split.",
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
