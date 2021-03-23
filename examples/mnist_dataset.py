from ride.core import AttributeDict, ClassificationDataset, Configs
from ride.utils.env import DATASETS_PATH
from ride.utils.logging import getLogger

logger = getLogger(__name__)


class MnistDataset(ClassificationDataset):
    """
    Example Mnist Dataset
    Modified from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist_datamodule.py
    """

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
        import pl_bolts

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
