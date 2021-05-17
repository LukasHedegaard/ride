import numpy as np
import torch

import ride
from examples.mnist_dataset import MnistDataset


class SimpleClassifier(
    ride.RideModule,
    ride.SgdOneCycleOptimizer,
    ride.TopKAccuracyMetric(1, 3),
    MnistDataset,
):
    def __init__(self, hparams):
        # `self.input_shape` and `self.output_shape` were injected via `MnistDataset`
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
            description="Number of hidden units.",
        )
        return c


if __name__ == "__main__":
    ride.Main(SimpleClassifier).argparse()
