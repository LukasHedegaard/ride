from pathlib import Path

import pytest

from examples.simple_classifier import SimpleClassifier

from .dummy_dataset import DummyClassificationDataLoader

from ride import Main  # noqa: F401  # isort:skip


# This test has a dependency on MNIST downloads, which are unstable
@pytest.mark.skip_cicd
def test_simple_classifier():
    m = Main(SimpleClassifier)
    parser = m.argparse(run=False)
    args, _ = parser.parse_known_args()
    args.train = True
    args.test = True
    args.max_epochs = 1
    args.limit_train_batches = 100
    args.limit_val_batches = 10
    args.limit_test_batches = 10
    args.test_confusion_matrix = True
    args.logging_backend = "wandb"
    m.main(args)

    assert "loss" in m.runner.trainer.model.metrics()
    assert "top1acc" in m.runner.trainer.model.metrics()
    assert "top3acc" in m.runner.trainer.model.metrics()
    assert (Path(m.log_dir) / "figures" / "test" / "confusion_matrix.png").is_file()


def test_simpler_classifier():
    # Serves as integration test with
    # feature extraction and multiple concurrent visualisations
    SimplerClassifier = SimpleClassifier.with_dataset(DummyClassificationDataLoader)
    m = Main(SimplerClassifier)
    parser = m.argparse(run=False)
    args, _ = parser.parse_known_args()
    args.batch_size = 10
    args.train = True
    args.test = True
    args.max_epochs = 1
    args.limit_train_batches = 100
    args.limit_val_batches = 10
    args.limit_test_batches = 10
    args.test_confusion_matrix = True
    args.visualise_features = "pca"
    args.extract_features_after_layer = "l1"
    args.loss = "cross_entropy"
    m.main(args)

    assert "loss" in m.runner.trainer.model.metrics()
    assert "top1acc" in m.runner.trainer.model.metrics()
    assert "top3acc" in m.runner.trainer.model.metrics()
    assert (Path(m.log_dir) / "figures" / "test" / "l1_pca.png").is_file()
    assert (Path(m.log_dir) / "figures" / "test" / "confusion_matrix.png").is_file()
