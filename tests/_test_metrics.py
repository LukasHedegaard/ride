import pytest  # noqa: F401

from ride import metrics  # noqa: F401


def test_BaseMetricMixin():
    """Test that base class has sensible defaults"""
    assert False


class TestTopKAccuracyMetric:
    def test_init(self):
        """Test intialisation"""
        assert False

    def test_validate_attributes(self):
        """Test attribute validation"""
        assert False

    def test_metrics_step(self):
        """Test metrics step"""
        assert False

    def test_metrics_epoch(self):
        """Test metrics step"""
        assert False


class TestFlopsMetricMixin:
    def test_init(self):
        """Test intialisation"""
        assert False


class FlopsWeightedAccuracyMetric:
    def test_init(self):
        """Test intialisation"""
        assert False

    def test_validate_attributes(self):
        """Test attribute validation"""
        assert False

    def test_metrics_step(self):
        """Test metrics step"""
        assert False

    def test_metrics_epoch(self):
        """Test metrics step"""
        assert False


def test_topk_accuracies():
    """Test topk_accuracies"""
    assert False


def test_accuracy():
    """Test accuracy"""
    assert False


def test_topk_errors():
    """Test topk_errors"""
    assert False


def test_gflops():
    """Test computation of gflops"""
    assert False


def test_activations():
    """Test computation of activations"""
    assert False


def test_params_count():
    """Test computation of the number of parameters"""
    assert False
