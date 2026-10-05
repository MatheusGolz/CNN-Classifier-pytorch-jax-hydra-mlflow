"""Shallow CNN and its training loop in PyTorch.

This file and jax_model.py are written to mirror each other: the same model,
the same trainer methods, in the same order. Reading them side by side shows
how the two frameworks differ.
"""

import numpy as np
import torch
from torch import nn

from data import Split, iterate_batches


class ShallowCNN(nn.Module):
    """[Conv 3x3 -> ReLU -> MaxPool 2x2] x num_blocks, then a small dense head."""

    def __init__(
        self,
        num_blocks: int,
        base_channels: int,
        hidden_units: int,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 32,
    ):
        super().__init__()

        # Build the convolutional blocks in a loop, so depth is just a number
        # in the config. Each block doubles the channels and halves the image size.
        layers = []
        channels_in = in_channels
        for i in range(num_blocks):
            channels_out = base_channels * 2**i
            layers += [
                nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),  # padding=1 keeps 32x32
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),                                    # 32x32 -> 16x16
            ]
            channels_in = channels_out
        self.features = nn.Sequential(*layers)

        # After num_blocks poolings the image is image_size / 2**num_blocks pixels wide.
        final_size = image_size // 2**num_blocks
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels_in * final_size * final_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes),  # raw scores (logits); softmax is inside the loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> int:
    """Total number of trainable weights and biases."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TorchTrainer:
    """Owns the model and optimizer, and runs training and evaluation epochs."""

    def __init__(self, model_cfg, lr: float, seed: int):
        torch.manual_seed(seed)  # makes weight initialisation reproducible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ShallowCNN(
            num_blocks=model_cfg.num_blocks,
            base_channels=model_cfg.base_channels,
            hidden_units=model_cfg.hidden_units,
            num_classes=model_cfg.num_classes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name() if self.device.type == "cuda" else "cpu"

    @property
    def num_params(self) -> int:
        return count_params(self.model)

    def _to_device(self, images: np.ndarray, labels: np.ndarray):
        """NumPy (N, H, W, C) batch -> PyTorch (N, C, H, W) tensors on the GPU."""
        x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(self.device)
        y = torch.from_numpy(labels).long().to(self.device)
        return x, y

    def train_epoch(self, split: Split, batch_size: int, rng: np.random.Generator,
                    max_batches: int | None = None) -> tuple[float, float]:
        """One pass over the training data. Returns (mean loss, accuracy)."""
        self.model.train()
        # Keep running totals on the GPU and read them once at the end. Calling
        # .item() every step would force the CPU to wait for the GPU each time.
        total_loss = torch.zeros((), device=self.device)
        total_correct = torch.zeros((), device=self.device)
        total_seen = 0

        for images, labels in iterate_batches(split, batch_size, rng, drop_last=True,
                                              max_batches=max_batches):
            x, y = self._to_device(images, labels)

            logits = self.model(x)              # forward pass
            loss = self.loss_fn(logits, y)
            self.optimizer.zero_grad()          # clear gradients from the last step
            loss.backward()                     # backward pass: compute gradients
            self.optimizer.step()               # update the weights

            total_loss += loss.detach() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum()
            total_seen += len(y)

        return total_loss.item() / total_seen, total_correct.item() / total_seen

    @torch.no_grad()  # no gradients needed for evaluation: faster, less memory
    def evaluate(self, split: Split, batch_size: int) -> tuple[float, float]:
        """Loss and accuracy on a validation or test split."""
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        total_correct = torch.zeros((), device=self.device)

        for images, labels in iterate_batches(split, batch_size):
            x, y = self._to_device(images, labels)
            logits = self.model(x)
            total_loss += self.loss_fn(logits, y) * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum()

        return total_loss.item() / len(split), total_correct.item() / len(split)

    @torch.no_grad()
    def predict(self, split: Split, batch_size: int) -> np.ndarray:
        """Predicted class for every image in the split."""
        self.model.eval()
        predictions = []
        for images, labels in iterate_batches(split, batch_size):
            x, _ = self._to_device(images, labels)
            predictions.append(self.model(x).argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions)

    def save(self, path: str) -> None:
        """Save the trained weights (a 'state dict' maps layer names to tensors)."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load weights saved by save() into this trainer's model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def release_memory(self) -> None:
        """Free the GPU memory, so a JAX run in the same process has room.

        Call this only once training is finished: the model is deleted.
        """
        del self.model, self.optimizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
