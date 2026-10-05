# Shallow CNN on CIFAR-10: PyTorch vs JAX

A small, end-to-end machine learning project that trains the same shallow
convolutional network on CIFAR-10 in **PyTorch** and in **JAX**, manages every
experiment with **Hydra**, tracks all runs with **MLflow**, and analyses the
results with **pandas**.

**[See the results notebook →](notebooks/results.ipynb)** (no need to run anything)

The project runs in three stages:

1. **Hyperparameter sweep (PyTorch).** Hydra trains 9 configurations: 3 depths × 3 widths.
2. **Replication (JAX).** The winning architecture is trained in both frameworks, with 3 seeds each.
3. **Comparison (pandas).** Runs are pulled from MLflow into DataFrames to compare accuracy and speed.

<!-- After your runs, the figures below are created by src/select_best.py and src/compare.py. -->
![Sweep results](results/sweep_heatmap.png)
![PyTorch vs JAX](results/framework_comparison.png)

<!-- Paste the takeaways printed at the end of notebooks/results.ipynb here. -->

---

## How each tool is used

| Tool | Role in this project |
|---|---|
| **PyTorch** | Main implementation: model class and a hand-written training loop (`src/torch_model.py`) |
| **JAX** (Flax NNX + Optax) | Second implementation of the same model, written to mirror the PyTorch file (`src/jax_model.py`) |
| **Hydra** | All settings live in YAML under `conf/`; any value can be changed from the command line; `-m` runs the sweep |
| **MLflow** | Every run logs its config, per-epoch metrics, timings, confusion matrix, weights and Hydra config |
| **pandas** | `mlflow.search_runs` returns runs as a DataFrame; `src/results.py` ranks, pivots and aggregates them |

## The model

A deliberately shallow, VGG-style CNN, identical in both frameworks:

```
input 32×32×3
→ [Conv 3×3 → ReLU → MaxPool 2×2] × num_blocks      (block i has base_channels × 2^i filters)
→ Flatten → Dense 128 → ReLU → Dense 10
```

BatchNorm and Dropout are left out on purpose: in JAX they need extra state and
random-key handling, which would make the two implementations harder to compare.

### The two hyperparameters searched

| Hyperparameter | Values | Meaning |
|---|---|---|
| `model.num_blocks` | 1, 2, 3 | depth (number of conv blocks) |
| `model.base_channels` | 16, 32, 64 | width (filters in the first block) |

Everything else is fixed: Adam, learning rate 0.001, batch size 128, 15 epochs.
The winner is chosen by **validation** accuracy; the **test** set is only used in stage 2.

### Keeping the comparison fair

- **Same data, same order.** Both frameworks read the same NumPy arrays from `src/data.py`,
  and the same seed gives identical batches. CIFAR-10 is downloaded once with torchvision
  (JAX ships no datasets of its own).
- **Same architecture.** A unit test checks that both versions have exactly the same number of parameters.
- **Several seeds.** Each framework is trained with 3 seeds, so differences can be told apart from noise.
- **Compilation is timed separately.** JAX compiles the training step with XLA during the first epoch,
  so speed is compared on the later epochs.
- One thing differs by design: PyTorch and Flax use different default weight initialisers.

## Project structure

```
conf/
  config.yaml              defaults: seed, data split, training settings, MLflow
  model/shallow_cnn.yaml   num_blocks, base_channels, hidden_units
  framework/{torch,jax}.yaml
  experiment/sweep.yaml    stage 1: 3 x 3 grid with PyTorch
  experiment/compare.yaml  stage 2: torch,jax x seeds 0,1,2, with test evaluation
src/
  data.py                  download, split, normalise; shared batch iterator
  torch_model.py           PyTorch model and trainer
  jax_model.py             Flax NNX model and trainer (mirrors torch_model.py)
  train.py                 Hydra entry point; trains one run and logs it to MLflow
  results.py               MLflow -> pandas helpers and plots
  select_best.py           stage 1 ranking; prints the stage 2 command
  compare.py               stage 3 PyTorch vs JAX summary
notebooks/results.ipynb    the results, with outputs saved
tests/test_models.py       fast checks (no GPU, no download)
results/                   tables and figures written by the scripts
```

---

## Setup on Windows (WSL2)

JAX only supports NVIDIA GPUs on Linux. On Windows, use **WSL2**, which runs Ubuntu
inside Windows with full GPU access. PyTorch works natively on Windows too, but running
everything in WSL2 keeps the two frameworks on equal terms.

1. **Install WSL2 with Ubuntu.** In PowerShell as administrator, then restart:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. **Update the NVIDIA driver on Windows** (from nvidia.com or the NVIDIA app).
   Do *not* install a separate Linux driver inside Ubuntu; WSL2 uses the Windows one.
   In the Ubuntu terminal, `nvidia-smi` should list your GPU.
3. **Get the code and create a virtual environment** in the Ubuntu terminal. Keep the
   project in your Linux home folder (`~`), not under `/mnt/c`, where file access is much slower.
   ```bash
   sudo apt update && sudo apt install -y python3-venv git
   git clone https://github.com/<your-username>/cifar10-torch-vs-jax.git
   cd cifar10-torch-vs-jax
   python3 -m venv .venv && source .venv/bin/activate
   ```
4. **Install the GPU builds, then the rest.** PyTorch's Linux wheels on PyPI use CUDA 13,
   so install the matching JAX build:
   ```bash
   pip install torch torchvision
   pip install -U "jax[cuda13]"
   pip install -r requirements.txt
   ```
5. **Check that both frameworks see the GPU:**
   ```bash
   python -c "import torch; print('PyTorch:', torch.cuda.is_available(), torch.version.cuda)"
   python -c "import jax; print('JAX:', jax.devices())"
   ```
   Expect `True` and a `CudaDevice`. If JAX lists only a CPU device, check that the CUDA
   version in the first line matches the `jax[cudaXX]` extra you installed.

On Linux, steps 3–5 are all you need.

## Running the project

Run everything from the repository root, with the virtual environment active.

```bash
# 0. Quick checks: unit tests, then a 1-epoch run (this also downloads CIFAR-10, ~170 MB)
pytest
python src/train.py train.epochs=1

# 1. Stage 1: the 9-run PyTorch sweep
python src/train.py -m +experiment=sweep
python src/select_best.py              # ranks the runs and prints the stage 2 command

# 2. Stage 2: the winner in PyTorch and JAX, 3 seeds each (use the numbers printed above)
python src/train.py -m +experiment=compare model.num_blocks=2 model.base_channels=64

# 3. Stage 3: compare the frameworks
python src/compare.py

# 4. Save the results into the notebook, so GitHub shows them
jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
```

Each run takes a few minutes on a typical GPU. Expect test accuracy of roughly 65–75%,
which is normal for a shallow network trained without data augmentation.

### Browsing runs in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000, pick an experiment (`cifar10-torch-sweep` or
`cifar10-framework-comparison`), select runs and click **Compare** to see their
learning curves side by side. Each run also stores its confusion matrix, weights
and the exact Hydra config under **Artifacts**.

### Trying other settings with Hydra

Any config value can be overridden from the command line:

```bash
python src/train.py framework=jax model.num_blocks=3 train.epochs=5 train.lr=0.0005
python src/train.py --cfg job       # print the final config without training
```

### What to commit

Commit the code, `results/` and the executed notebook. `mlflow.db`, `mlruns/`, `data/`
and the Hydra folders are in `.gitignore`: they are large and can be regenerated.
