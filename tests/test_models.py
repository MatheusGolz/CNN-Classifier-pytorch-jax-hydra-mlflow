"""Fast checks that need no dataset download and no GPU.

Run with:  pytest
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import Split, iterate_batches, prepare_splits  # noqa: E402
from jax_model import JaxTrainer  # noqa: E402
from torch_model import TorchTrainer  # noqa: E402

SWEEP = [(blocks, channels) for blocks in (1, 2, 3) for channels in (16, 32, 64)]


def model_cfg(num_blocks: int, base_channels: int) -> SimpleNamespace:
    return SimpleNamespace(num_blocks=num_blocks, base_channels=base_channels,
                           hidden_units=128, num_classes=10)


def fake_split(n: int = 64, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    return Split(rng.normal(size=(n, 32, 32, 3)).astype(np.float32),
                 rng.integers(0, 10, n).astype(np.int32))


@pytest.mark.parametrize("num_blocks,base_channels", SWEEP)
def test_same_parameter_count_in_both_frameworks(num_blocks, base_channels):
    """If the counts match, JAX builds the same architecture as PyTorch."""
    cfg = model_cfg(num_blocks, base_channels)
    assert TorchTrainer(cfg, lr=1e-3, seed=0).num_params == JaxTrainer(cfg, lr=1e-3, seed=0).num_params


@pytest.mark.parametrize("trainer_class", [TorchTrainer, JaxTrainer])
def test_training_step_and_prediction_shapes(trainer_class):
    """One short epoch runs, returns sensible numbers, and predicts one class per image."""
    split = fake_split()
    trainer = trainer_class(model_cfg(2, 16), lr=1e-3, seed=0)

    loss, acc = trainer.train_epoch(split, batch_size=16, rng=np.random.default_rng(0))
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0

    loss, acc = trainer.evaluate(split, batch_size=16)
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0

    predictions = trainer.predict(split, batch_size=16)
    assert predictions.shape == (len(split),)
    assert predictions.min() >= 0 and predictions.max() < 10


def test_batches_are_identical_for_the_same_seed():
    """Both frameworks get exactly the same batches when given the same seed."""
    split = fake_split(n=100)
    first = list(iterate_batches(split, 32, np.random.default_rng(7), drop_last=True))
    second = list(iterate_batches(split, 32, np.random.default_rng(7), drop_last=True))
    assert len(first) == 3  # 100 images -> 3 full batches of 32, the last 4 are dropped
    for (x1, y1), (x2, y2) in zip(first, second):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


def test_prepare_splits_sizes_and_normalisation():
    rng = np.random.default_rng(0)
    x_train = rng.integers(0, 256, (500, 32, 32, 3), dtype=np.uint8)
    x_test = rng.integers(0, 256, (100, 32, 32, 3), dtype=np.uint8)
    y_train = rng.integers(0, 10, 500).astype(np.int32)
    y_test = rng.integers(0, 10, 100).astype(np.int32)

    data = prepare_splits(x_train, y_train, x_test, y_test, val_size=50, split_seed=0)

    assert (len(data.train), len(data.val), len(data.test)) == (450, 50, 100)
    assert data.train.x.dtype == np.float32
    # Normalised with training statistics: training data has mean ~0 and std ~1 per channel.
    np.testing.assert_allclose(data.train.x.mean(axis=(0, 1, 2)), 0, atol=1e-4)
    np.testing.assert_allclose(data.train.x.std(axis=(0, 1, 2)), 1, atol=1e-3)
