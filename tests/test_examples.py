from ride import Main  # noqa: F401  # isort:skip
from examples.simple_classifier import SimpleClassifier


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
    m.main(args)
