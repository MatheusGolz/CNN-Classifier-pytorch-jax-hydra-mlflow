"""Stage 3: compare PyTorch and JAX on the winning architecture with pandas.

Run from the repository root after the comparison runs:
    python src/compare.py

Prints a per-run table and a per-framework summary (mean and standard deviation
over seeds), and saves both tables and a figure to results/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results import (COMPARE_EXPERIMENT, DEFAULT_TRACKING_URI, REPO_ROOT, framework_summary,
                     load_runs, metric_history, plot_epoch_times, plot_val_curves)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", default=COMPARE_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    runs = load_runs(args.experiment, args.tracking_uri)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Per-run table: one row per (framework, seed).
    per_run = runs.sort_values(["framework", "seed"])[
        ["framework", "seed", "num_params", "final_val_acc", "test_acc",
         "first_epoch_time_s", "mean_epoch_time_s", "total_train_time_s"]
    ]
    print("\nEvery run:\n")
    print(per_run.to_string(index=False, float_format="{:.4f}".format))

    # Summary table: mean and standard deviation over the seeds.
    summary = framework_summary(runs)
    print("\nPer framework (mean over seeds, times in seconds):\n")
    print(summary.to_string(float_format="{:.4f}".format))

    # A sanity check that JAX really replicated the PyTorch model.
    if runs["num_params"].nunique() > 1:
        print("\nWARNING: parameter counts differ, so the two models are not the same architecture.")
    else:
        print(f"\nBoth frameworks trained the same architecture: {summary['num_params'].iloc[0]:,} parameters.")

    per_run.to_csv(out_dir / "comparison_runs.csv", index=False)
    summary.to_csv(out_dir / "comparison_summary.csv")

    # Figure: learning curves on the left, training speed on the right.
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    plot_val_curves(metric_history(runs, "val_acc", args.tracking_uri), left)
    plot_epoch_times(summary, right)
    fig.tight_layout()
    fig.savefig(out_dir / "framework_comparison.png", dpi=150)
    print(f"\nSaved tables and framework_comparison.png to {out_dir}")


if __name__ == "__main__":
    main()
