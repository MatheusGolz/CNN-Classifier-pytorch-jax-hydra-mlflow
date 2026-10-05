"""Train one shallow CNN on CIFAR-10 and log the run to MLflow.

Hydra builds the configuration from conf/ and the command line, and calls
main() once per run (or once per combination, when sweeping with -m).

Examples, run from the repository root:
    python src/train.py                                   # one PyTorch run with the defaults
    python src/train.py framework=jax train.epochs=5      # one JAX run, 5 epochs
    python src/train.py -m +experiment=sweep              # stage 1: the 9-run PyTorch sweep
    python src/train.py -m +experiment=compare model.num_blocks=2 model.base_channels=64
                                                          # stage 2: winner in both frameworks
"""

import os

# JAX normally reserves 75% of GPU memory the first time it runs. A sweep that
# mixes PyTorch and JAX runs in one process needs to share the GPU, so we ask
# JAX to allocate memory as needed instead. This must happen before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging
import tempfile
import time
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")  # draw plots to files, without needing a display
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from data import CLASS_NAMES, CIFAR10, load_cifar10
from results import SEQUENTIAL_BLUE

log = logging.getLogger(__name__)

# During a sweep Hydra calls main() many times in the same process.
# Keep the dataset in memory so it is loaded and normalised only once.
_DATA_CACHE: dict[tuple, CIFAR10] = {}


def get_data(cfg: DictConfig) -> CIFAR10:
    key = (cfg.data.dir, cfg.data.val_size, cfg.data.split_seed)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_cifar10(
            to_absolute_path(cfg.data.dir), cfg.data.val_size, cfg.data.split_seed
        )
    return _DATA_CACHE[key]


def make_trainer(cfg: DictConfig):
    """Pick the PyTorch or JAX implementation from the config.

    The imports happen here, so a PyTorch run never imports JAX and vice versa.
    """
    if cfg.framework.name == "torch":
        from torch_model import TorchTrainer
        return TorchTrainer(cfg.model, lr=cfg.train.lr, seed=cfg.seed)
    if cfg.framework.name == "jax":
        from jax_model import JaxTrainer
        return JaxTrainer(cfg.model, lr=cfg.train.lr, seed=cfg.seed)
    raise ValueError(f"Unknown framework: {cfg.framework.name}")


def flatten(config: dict, prefix: str = "") -> dict:
    """{'model': {'num_blocks': 2}} -> {'model.num_blocks': 2}, the form MLflow expects."""
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def tracking_uri(uri: str) -> str:
    """Turn 'sqlite:///mlflow.db' into an absolute path, so it works from any folder."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + to_absolute_path(uri[len(prefix):])
    return uri


def confusion_matrix_figure(labels: np.ndarray, predictions: np.ndarray, title: str):
    """Rows: true class. Columns: predicted class. Each row sums to 1."""
    counts = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
    np.add.at(counts, (labels, predictions), 1)
    fractions = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(fractions, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)  # light = rare, dark = common
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Fraction of the true class")
    fig.tight_layout()
    return fig


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    data = get_data(cfg)
    trainer = make_trainer(cfg)
    m = cfg.model
    run_name = f"{cfg.framework.name}-blocks{m.num_blocks}-ch{m.base_channels}-seed{cfg.seed}"
    log.info("Run %s on %s, %s parameters", run_name, trainer.device_name, f"{trainer.num_params:,}")

    mlflow.set_tracking_uri(tracking_uri(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name=run_name):
        # 1. What was run: every config value, plus a few descriptive tags.
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))
        mlflow.set_tags({"framework": cfg.framework.name, "device": trainer.device_name})
        mlflow.log_metric("num_params", trainer.num_params)

        # 2. Training. The same seed gives the same batch order in both frameworks.
        batch_rng = np.random.default_rng(cfg.seed)
        epoch_times, val_accs = [], []

        for epoch in range(1, cfg.train.epochs + 1):
            start = time.perf_counter()
            train_loss, train_acc = trainer.train_epoch(
                data.train, cfg.train.batch_size, batch_rng, cfg.train.max_train_batches
            )
            epoch_times.append(time.perf_counter() - start)  # training time only, not validation

            val_loss, val_acc = trainer.evaluate(data.val, cfg.train.batch_size)
            val_accs.append(val_acc)

            # Metrics logged with a step number become curves in the MLflow UI.
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "epoch_time_s": epoch_times[-1],
            }, step=epoch)
            log.info("epoch %2d | train loss %.3f acc %.3f | val loss %.3f acc %.3f | %.1fs",
                     epoch, train_loss, train_acc, val_loss, val_acc, epoch_times[-1])

        # 3. One-number summaries, used to rank and compare runs.
        # JAX compiles during epoch 1, so the "mean" excludes it whenever possible.
        later_epochs = epoch_times[1:] or epoch_times
        mlflow.log_metrics({
            "final_val_acc": val_accs[-1],
            "best_val_acc": max(val_accs),
            "first_epoch_time_s": epoch_times[0],
            "mean_epoch_time_s": float(np.mean(later_epochs)),
            "total_train_time_s": float(np.sum(epoch_times)),
        })

        # 4. Test set: only for the final comparison runs (train.eval_test=true).
        if cfg.train.eval_test:
            test_loss, test_acc = trainer.evaluate(data.test, cfg.train.batch_size)
            mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
            log.info("test loss %.3f acc %.3f", test_loss, test_acc)

            predictions = trainer.predict(data.test, cfg.train.batch_size)
            fig = confusion_matrix_figure(data.test.y, predictions,
                                          f"{run_name}: test accuracy {test_acc:.1%}")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

        # 5. Files: the trained weights and the exact config Hydra used.
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / ("model.pt" if cfg.framework.name == "torch" else "model_params.pkl")
            trainer.save(str(weights))
            mlflow.log_artifact(str(weights), artifact_path="model")
        mlflow.log_artifacts(str(Path(HydraConfig.get().runtime.output_dir) / ".hydra"),
                             artifact_path="hydra_config")

    trainer.release_memory()
    return val_accs[-1]


if __name__ == "__main__":
    main()
