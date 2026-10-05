"""Stage 1 results: rank the PyTorch sweep runs and pick the winner.

Run from the repository root after the sweep:
    python src/select_best.py

Prints the ranking and the depth x width grid, saves them to results/, and
prints the exact command for stage 2 (training the winner in both frameworks).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results import (DEFAULT_TRACKING_URI, REPO_ROOT, SWEEP_EXPERIMENT, best_config,
                     load_runs, plot_sweep_heatmap, sweep_grid, sweep_table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", default=SWEEP_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    runs = load_runs(args.experiment, args.tracking_uri)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    table = sweep_table(runs)
    grid = sweep_grid(runs)
    print(f"\n{len(runs)} sweep runs, ranked by final validation accuracy:\n")
    print(table.to_string(float_format="{:.4f}".format))
    print("\nFinal validation accuracy, conv blocks (rows) x base channels (columns):\n")
    print(grid.to_string(float_format="{:.2%}".format))

    table.to_csv(out_dir / "sweep_results.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    plot_sweep_heatmap(grid, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_heatmap.png", dpi=150)

    best = best_config(runs)
    print(f"\nWinner: {best.num_blocks} conv blocks, {best.base_channels} base channels "
          f"({best.num_params:,} parameters), validation accuracy {best.final_val_acc:.2%}.")
    print("\nStage 2: train it in PyTorch and JAX with 3 seeds each:\n")
    print(f"  python src/train.py -m +experiment=compare "
          f"model.num_blocks={best.num_blocks} model.base_channels={best.base_channels}\n")
    print(f"Saved {out_dir / 'sweep_results.csv'} and {out_dir / 'sweep_heatmap.png'}")


if __name__ == "__main__":
    main()
