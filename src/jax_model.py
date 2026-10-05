"""The same shallow CNN and training loop in JAX, using Flax NNX and Optax.

This mirrors torch_model.py method for method. The main differences to notice:

* Layout: Flax convolutions expect channels-last images (N, H, W, C), which is
  what data.py produces, so no transpose is needed here.
* Randomness: there is no global seed. Initial weights come from an explicit
  random key (nnx.Rngs).
* Gradients: instead of loss.backward(), we transform the loss function with
  nnx.value_and_grad, which returns both the loss and its gradients.
* Compilation: nnx.jit compiles the training step with XLA the first time it
  runs. That first call is slow; every later call is fast. This is why the
  first epoch is timed separately when comparing speed.
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from data import Split, iterate_batches


class ShallowCNN(nnx.Module):
    """[Conv 3x3 -> ReLU -> MaxPool 2x2] x num_blocks, then a small dense head."""

    def __init__(
        self,
        num_blocks: int,
        base_channels: int,
        hidden_units: int,
        num_classes: int,
        *,
        rngs: nnx.Rngs,  # source of random keys for initialising the weights
        in_channels: int = 3,
        image_size: int = 32,
    ):
        # Same loop as the PyTorch version. Flax needs nnx.List (not a plain
        # Python list) so it can find the layers' parameters inside it.
        convs = []
        channels_in = in_channels
        for i in range(num_blocks):
            channels_out = base_channels * 2**i
            convs.append(nnx.Conv(channels_in, channels_out, kernel_size=(3, 3),
                                  padding="SAME", rngs=rngs))  # "SAME" keeps 32x32
            channels_in = channels_out
        self.convs = nnx.List(convs)

        final_size = image_size // 2**num_blocks
        self.dense1 = nnx.Linear(channels_in * final_size * final_size, hidden_units, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_units, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for conv in self.convs:
            x = nnx.relu(conv(x))
            # strides must be given explicitly: Flax's default stride is 1, not the window size.
            x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)   # flatten everything except the batch dimension
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)           # logits


def count_params(model: nnx.Module) -> int:
    """Total number of trainable weights and biases."""
    params = nnx.state(model, nnx.Param)
    return sum(leaf.size for leaf in jax.tree.leaves(params))


# --- Compiled steps --------------------------------------------------------
# These are plain functions rather than methods so nnx.jit can compile them.
# nnx.jit understands Flax modules: changes made to `model` and `optimizer`
# inside the function (the weight update) are kept after it returns.

@nnx.jit
def train_step(model: ShallowCNN, optimizer: nnx.Optimizer, images, labels):
    """One optimisation step. Returns (mean loss, number of correct predictions)."""

    def loss_fn(model):
        logits = model(images)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits  # logits are returned as "auxiliary" output for the accuracy

    # Forward pass and gradients in one call (the JAX equivalent of loss.backward()).
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)  # the equivalent of optimizer.step()

    correct = (logits.argmax(axis=-1) == labels).sum()
    return loss, correct


@nnx.jit
def eval_step(model: ShallowCNN, images, labels):
    """Summed loss and number of correct predictions for one batch."""
    logits = model(images)
    loss_sum = optax.softmax_cross_entropy_with_integer_labels(logits, labels).sum()
    correct = (logits.argmax(axis=-1) == labels).sum()
    return loss_sum, correct


@nnx.jit
def predict_step(model: ShallowCNN, images):
    return model(images).argmax(axis=-1)


class JaxTrainer:
    """Owns the model and optimizer, and runs training and evaluation epochs."""

    def __init__(self, model_cfg, lr: float, seed: int):
        self.model = ShallowCNN(
            num_blocks=model_cfg.num_blocks,
            base_channels=model_cfg.base_channels,
            hidden_units=model_cfg.hidden_units,
            num_classes=model_cfg.num_classes,
            rngs=nnx.Rngs(seed),  # makes weight initialisation reproducible
        )
        # wrt=nnx.Param: the optimizer updates the trainable parameters only.
        self.optimizer = nnx.Optimizer(self.model, optax.adam(lr), wrt=nnx.Param)

        # cached_partial binds the model and optimizer once, which removes most
        # of the Python overhead nnx.jit would otherwise add on every call.
        self._train_step = nnx.cached_partial(train_step, self.model, self.optimizer)
        self._eval_step = nnx.cached_partial(eval_step, self.model)
        self._predict_step = nnx.cached_partial(predict_step, self.model)

    @property
    def device_name(self) -> str:
        device = jax.devices()[0]
        return device.device_kind if device.platform == "gpu" else device.platform

    @property
    def num_params(self) -> int:
        return count_params(self.model)

    def train_epoch(self, split: Split, batch_size: int, rng: np.random.Generator,
                    max_batches: int | None = None) -> tuple[float, float]:
        """One pass over the training data. Returns (mean loss, accuracy)."""
        # As in PyTorch, totals stay on the device and are read once at the end:
        # JAX runs asynchronously, and float() makes Python wait for the result.
        total_loss = jnp.zeros(())
        total_correct = jnp.zeros((), dtype=jnp.int32)
        total_seen = 0

        for images, labels in iterate_batches(split, batch_size, rng, drop_last=True,
                                              max_batches=max_batches):
            loss, correct = self._train_step(images, labels)  # NumPy arrays are moved to the GPU automatically
            total_loss += loss * len(labels)
            total_correct += correct
            total_seen += len(labels)

        return float(total_loss) / total_seen, int(total_correct) / total_seen

    def evaluate(self, split: Split, batch_size: int) -> tuple[float, float]:
        """Loss and accuracy on a validation or test split."""
        total_loss = jnp.zeros(())
        total_correct = jnp.zeros((), dtype=jnp.int32)

        # Note: a final smaller batch has a new shape, so it is compiled once more.
        for images, labels in iterate_batches(split, batch_size):
            loss_sum, correct = self._eval_step(images, labels)
            total_loss += loss_sum
            total_correct += correct

        return float(total_loss) / len(split), int(total_correct) / len(split)

    def predict(self, split: Split, batch_size: int) -> np.ndarray:
        """Predicted class for every image in the split."""
        predictions = [np.asarray(self._predict_step(images))
                       for images, _ in iterate_batches(split, batch_size)]
        return np.concatenate(predictions)

    def save(self, path: str) -> None:
        """Save the trained weights as a nested dict of NumPy arrays."""
        params = nnx.state(self.model, nnx.Param).to_pure_dict()
        with open(path, "wb") as f:
            pickle.dump(jax.tree.map(np.asarray, params), f)

    def load(self, path: str) -> None:
        """Load weights saved by save() into this trainer's model."""
        with open(path, "rb") as f:
            params = pickle.load(f)
        state = nnx.state(self.model, nnx.Param)
        state.replace_by_pure_dict(params)  # put the saved arrays into the model's structure
        nnx.update(self.model, state)       # and write them into the model in place

    def release_memory(self) -> None:
        """Nothing to do: JAX frees device arrays when they are no longer used."""
