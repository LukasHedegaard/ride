import torch

import ride


class SimpleClassifier(
    ride.RideModule,
    ride.Lifecycle,
    ride.SgdOneCycleOptimizer,
    ride.MnistDataset,
    ride.TopKAccuracyMetric(1, 3),
):
    def __init__(self, hparams):
        # Injected via `ride.MnistDataset`
        height, width = self.input_shape
        num_classes = self.output_shape

        self.l1 = torch.nn.Linear(height * width, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    @staticmethod
    def configs():
        c = ride.Configs.collect(SimpleClassifier)
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
