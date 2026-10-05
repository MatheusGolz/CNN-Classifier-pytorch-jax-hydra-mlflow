"""Load CIFAR-10 once and serve the same NumPy batches to PyTorch and JAX.

Neither framework needs its own data pipeline: torchvision downloads the
dataset, we turn it into plain NumPy arrays, and both training loops read
batches from `iterate_batches`. With the same seed, PyTorch and JAX see exactly
the same images in exactly the same order, which keeps the comparison fair.

Arrays use the channels-last layout (N, H, W, C) that JAX/Flax expects.
The PyTorch trainer transposes each batch to channels-first (N, C, H, W).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass
class Split:
    """One part of the dataset: images and their integer labels."""

    x: np.ndarray  # float32, shape (N, 32, 32, 3), normalised
    y: np.ndarray  # int32, shape (N,), values 0..9

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class CIFAR10:
    """The three splits plus the statistics used to normalise them."""

    train: Split
    val: Split
    test: Split
    mean: np.ndarray  # per-channel mean of the training images (0..1 scale)
    std: np.ndarray   # per-channel standard deviation


def load_raw_cifar10(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download CIFAR-10 (first time only) and return raw uint8 arrays.

    torchvision's CIFAR10 class already stores the images as a NumPy array of
    shape (N, 32, 32, 3), so no image decoding is needed.
    """
    from torchvision.datasets import CIFAR10 as TorchvisionCIFAR10

    data_dir = Path(data_dir)
    train = TorchvisionCIFAR10(root=data_dir, train=True, download=True)
    test = TorchvisionCIFAR10(root=data_dir, train=False, download=True)
    return (
        train.data, np.asarray(train.targets, dtype=np.int32),
        test.data, np.asarray(test.targets, dtype=np.int32),
    )


def prepare_splits(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    val_size: int, split_seed: int,
) -> CIFAR10:
    """Split off a validation set and normalise all images.

    The mean and standard deviation come from the training split only, so no
    information from the validation or test images leaks into training.
    """
    # Shuffle the 50k training images with a fixed seed, then hold out val_size.
    order = np.random.default_rng(split_seed).permutation(len(y_train))
    val_idx, train_idx = order[:val_size], order[val_size:]

    # Scale pixels from 0..255 to 0..1, then compute per-channel statistics.
    train_images = x_train[train_idx].astype(np.float32) / 255.0
    mean = train_images.mean(axis=(0, 1, 2))
    std = train_images.std(axis=(0, 1, 2))

    def normalise(images: np.ndarray) -> np.ndarray:
        return ((images.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)

    return CIFAR10(
        train=Split(normalise(x_train[train_idx]), y_train[train_idx]),
        val=Split(normalise(x_train[val_idx]), y_train[val_idx]),
        test=Split(normalise(x_test), y_test),
        mean=mean,
        std=std,
    )


def load_cifar10(data_dir: str | Path, val_size: int = 5000, split_seed: int = 0) -> CIFAR10:
    """Download (if needed), split and normalise CIFAR-10."""
    return prepare_splits(*load_raw_cifar10(data_dir), val_size=val_size, split_seed=split_seed)


def iterate_batches(
    split: Split,
    batch_size: int,
    rng: np.random.Generator | None = None,
    drop_last: bool = False,
    max_batches: int | None = None,
):
    """Yield (images, labels) batches as NumPy arrays.

    rng:         if given, the data is shuffled with it (use for training);
                 if None, batches come in the original order (use for evaluation).
    drop_last:   skip a final smaller batch. Used for training so every JAX step
                 has the same input shape and is compiled only once.
    max_batches: stop early; only used for quick smoke tests.
    """
    n = len(split)
    indices = rng.permutation(n) if rng is not None else np.arange(n)
    stop = n - n % batch_size if drop_last else n

    for batch_number, start in enumerate(range(0, stop, batch_size)):
        if max_batches is not None and batch_number >= max_batches:
            break
        batch_idx = indices[start:start + batch_size]
        yield split.x[batch_idx], split.y[batch_idx]
