"""Read finished runs back from MLflow into pandas, and plot them.

Shared by select_best.py, compare.py and notebooks/results.ipynb, so the
scripts and the notebook always compute the same numbers.
"""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
SWEEP_EXPERIMENT = "cifar10-torch-sweep"
COMPARE_EXPERIMENT = "cifar10-framework-comparison"

# Chart colours. Each framework keeps the same colour in every chart.
FRAMEWORK_COLORS = {"torch": "#2a78d6", "jax": "#eb6834"}   # blue, orange
FRAMEWORK_LABELS = {"torch": "PyTorch", "jax": "JAX"}
INK, MUTED_INK, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# One-hue scale (light = low, dark = high) for heatmaps.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


# --- Loading ------------------------------------------------------------------

def load_runs(experiment: str, tracking_uri: str = DEFAULT_TRACKING_URI) -> pd.DataFrame:
    """One row per finished run, with short column names and proper types.

    mlflow.search_runs already returns a DataFrame, with columns such as
    'params.model.num_blocks' and 'metrics.final_val_acc'. We keep the useful
    ones and give them short names.
    """
    mlflow.set_tracking_uri(tracking_uri)
    raw = mlflow.search_runs(experiment_names=[experiment])
    if raw.empty:
        raise ValueError(f"No runs found in MLflow experiment '{experiment}'.")
    raw = raw[raw["status"] == "FINISHED"]

    columns = {
        "run_id": "run_id",
        "tags.mlflow.runName": "run_name",
        "params.framework.name": "framework",
        "params.model.num_blocks": "num_blocks",
        "params.model.base_channels": "base_channels",
        "params.model.hidden_units": "hidden_units",
        "params.seed": "seed",
        "metrics.num_params": "num_params",
        "metrics.final_val_acc": "final_val_acc",
        "metrics.best_val_acc": "best_val_acc",
        "metrics.test_acc": "test_acc",
        "metrics.first_epoch_time_s": "first_epoch_time_s",
        "metrics.mean_epoch_time_s": "mean_epoch_time_s",
        "metrics.total_train_time_s": "total_train_time_s",
        "tags.device": "device",
    }
    df = raw[[c for c in columns if c in raw.columns]].rename(columns=columns)

    # MLflow stores parameters as text; turn the numeric ones back into integers.
    for col in ["num_blocks", "base_channels", "hidden_units", "seed", "num_params"]:
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def load_trained_model(run: pd.Series, tracking_uri: str = DEFAULT_TRACKING_URI):
    """Rebuild the trainer of a finished run and load its saved weights from MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    model_cfg = SimpleNamespace(num_blocks=run.num_blocks, base_channels=run.base_channels,
                                hidden_units=run.hidden_units, num_classes=10)
    weights_dir = Path(mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path="model"))

    if run.framework == "torch":
        from torch_model import TorchTrainer
        trainer = TorchTrainer(model_cfg, lr=0.0, seed=0)  # lr is unused: no more training
        trainer.load(str(weights_dir / "model.pt"))
    else:
        from jax_model import JaxTrainer
        trainer = JaxTrainer(model_cfg, lr=0.0, seed=0)
        trainer.load(str(weights_dir / "model_params.pkl"))
    return trainer


def metric_history(runs: pd.DataFrame, metric: str = "val_acc",
                   tracking_uri: str = DEFAULT_TRACKING_URI) -> pd.DataFrame:
    """Per-epoch values of one metric for every run, in 'long' format:
    one row per (run, epoch), which is the easiest shape to group and plot."""
    client = MlflowClient(tracking_uri)
    rows = [
        {"run_id": run.run_id, "framework": run.framework, "seed": run.seed,
         "epoch": point.step, metric: point.value}
        for run in runs.itertuples()
        for point in client.get_metric_history(run.run_id, metric)
    ]
    return pd.DataFrame(rows).sort_values(["run_id", "epoch"]).reset_index(drop=True)


# --- Stage 1: the sweep -------------------------------------------------------

def sweep_table(runs: pd.DataFrame) -> pd.DataFrame:
    """All sweep runs ranked by final validation accuracy (the selection rule)."""
    cols = ["num_blocks", "base_channels", "num_params", "final_val_acc",
            "best_val_acc", "mean_epoch_time_s"]
    return (runs.sort_values("final_val_acc", ascending=False)[cols]
            .reset_index(drop=True)
            .rename_axis("rank"))


def sweep_grid(runs: pd.DataFrame, value: str = "final_val_acc") -> pd.DataFrame:
    """Depth x width table of one metric: rows = num_blocks, columns = base_channels."""
    return runs.pivot_table(index="num_blocks", columns="base_channels", values=value)


def best_config(runs: pd.DataFrame) -> pd.Series:
    """The winning sweep run: highest final validation accuracy."""
    return runs.loc[runs["final_val_acc"].idxmax()]


# --- Stage 2: PyTorch vs JAX --------------------------------------------------

def framework_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, one row per framework."""
    summary = runs.groupby("framework").agg(
        runs=("seed", "count"),
        num_params=("num_params", "first"),
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        val_acc_mean=("final_val_acc", "mean"),
        val_acc_std=("final_val_acc", "std"),
        first_epoch_s=("first_epoch_time_s", "mean"),
        later_epoch_s=("mean_epoch_time_s", "mean"),
        total_train_s=("total_train_time_s", "mean"),
    )
    # How many times faster each framework is than PyTorch, per epoch after the first.
    summary["speed_vs_torch"] = summary.loc["torch", "later_epoch_s"] / summary["later_epoch_s"]
    return summary.rename(index=FRAMEWORK_LABELS)


# --- Plots --------------------------------------------------------------------

def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Quiet axes so the data stands out."""
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED_INK)
    ax.set_ylabel(ylabel, color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_sweep_heatmap(grid: pd.DataFrame, ax=None, title="Validation accuracy by depth and width"):
    """Heatmap of the sweep grid, with each cell labelled with its value."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    image = ax.imshow(grid.values, cmap=SEQUENTIAL_BLUE, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    # Dark text on light cells, white text on dark cells, so labels stay readable.
    midpoint = (np.nanmin(grid.values) + np.nanmax(grid.values)) / 2
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            ax.text(j, i, f"{value:.1%}", ha="center", va="center",
                    color="white" if value > midpoint else INK, fontsize=11)
    _style(ax, title, "Base channels (width)", "Conv blocks (depth)")
    ax.figure.colorbar(image, ax=ax, format=lambda v, _: f"{v:.0%}")
    return ax


def plot_val_curves(history: pd.DataFrame, ax=None, metric: str = "val_acc"):
    """Validation accuracy per epoch: mean over seeds (line) and min-max range (band)."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    line_ends = []
    for framework, group in history.groupby("framework"):
        per_epoch = group.groupby("epoch")[metric].agg(["mean", "min", "max"])
        color = FRAMEWORK_COLORS[framework]
        ax.fill_between(per_epoch.index, per_epoch["min"], per_epoch["max"], color=color, alpha=0.15, lw=0)
        ax.plot(per_epoch.index, per_epoch["mean"], color=color, lw=2,
                label=FRAMEWORK_LABELS[framework])
        line_ends.append((per_epoch["mean"].iloc[-1], per_epoch.index[-1], framework))

    # Direct label at the end of each line, so colour is not the only cue.
    # The lower line's label goes below and the higher one's above, so they never overlap.
    for position, (value, last_epoch, framework) in enumerate(sorted(line_ends)):
        ax.annotate(f"{FRAMEWORK_LABELS[framework]} {value:.1%}", (last_epoch, value),
                    xytext=(6, -7 if position == 0 else 7), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # whole epochs only
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False)
    _style(ax, "Validation accuracy per epoch (band = range over seeds)", "Epoch", "Validation accuracy")
    return ax


def plot_epoch_times(summary: pd.DataFrame, ax=None):
    """Training time per epoch: first epoch (includes JAX compilation) vs later epochs."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    label_to_key = {label: key for key, label in FRAMEWORK_LABELS.items()}
    frameworks = list(summary.index)                       # e.g. ["JAX", "PyTorch"]
    colors = [FRAMEWORK_COLORS[label_to_key[f]] for f in frameworks]
    x = np.arange(len(frameworks))
    width = 0.36
    # Two bars per framework: the first epoch (lighter) and the average later epoch.
    for offset, column, alpha in [(-width / 2, "first_epoch_s", 0.45), (width / 2, "later_epoch_s", 1.0)]:
        bars = ax.bar(x + offset, summary[column], width * 0.94, color=colors, alpha=alpha)
        ax.bar_label(bars, labels=[f"{v:.1f}s" for v in summary[column]],
                     color=MUTED_INK, fontsize=9, padding=2)
    ax.set_xticks(x, [f"{f}\nfirst | later" for f in frameworks])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax, "Training time per epoch: first epoch vs later epochs", "", "Seconds")
    return ax


def plot_accuracy_vs_params(runs: pd.DataFrame, ax=None):
    """Sweep runs: is a bigger model a better model? Each point is labelled blocks x channels."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.scatter(runs["num_params"], runs["final_val_acc"], s=60,
               color=FRAMEWORK_COLORS["torch"], edgecolor="white", linewidth=1.5, zorder=3)
    for run in runs.itertuples():
        ax.annotate(f"{run.num_blocks} x {run.base_channels}", (run.num_params, run.final_val_acc),
                    xytext=(6, 4), textcoords="offset points", color=MUTED_INK, fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:g}M")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax, "Accuracy vs model size (label = blocks x channels)",
           "Trainable parameters", "Validation accuracy")
    return ax


def plot_per_class_accuracy(per_class: pd.DataFrame, ax=None):
    """Horizontal grouped bars: test accuracy for each class, one bar per framework.

    per_class: index = class names, one column per framework label (e.g. "PyTorch", "JAX").
    """
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    label_to_key = {label: key for key, label in FRAMEWORK_LABELS.items()}
    per_class = per_class.sort_values(per_class.columns[0])  # hardest class at the bottom
    y = np.arange(len(per_class))
    height = 0.4
    for i, column in enumerate(per_class.columns):
        ax.barh(y + (0.5 - i) * height, per_class[column], height * 0.92,
                color=FRAMEWORK_COLORS[label_to_key[column]], label=column)
    ax.set_yticks(y, per_class.index)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "Test accuracy per class", "Accuracy", "")
    return ax
