# Installation

This guide sets up everything needed to train the GRM, run the experiments and play with the uccchess bot.

You need:
- Python 3.8 or newer;
- the Python packages listed in `requirements.txt`;
- PyTorch;
- a Stockfish binary.

A GPU is only used to train the opponent-skill estimator. Every other stage runs on the CPU.

| File | Use it for |
|---|---|
| `environment-cuda122.yml` | Conda environment for CUDA 12.x machines (PyTorch 2.x) |
| `environment-cuda114.yml` | Conda environment for CUDA 11.4 machines (PyTorch 1.9.1, Python 3.8) |
| `environment-cpu.yml` | Conda environment without GPU |
| `requirements.txt` | Core packages without PyTorch |
| `requirements-cuda122.txt`, `requirements-cuda114.txt`, `requirements-cpu.txt` | pip: core packages plus the matching PyTorch build |
| `pyproject.toml` | `pip install -e .` makes `grm_rbc`, `uccchess` and `strangefish` importable from anywhere |
| `scripts/install_stockfish.sh` | Downloads an official Stockfish binary into `engines/` |
| `scripts/check_installation.py` | Verifies the whole setup |

## 1. Get the code

```bash
git clone <repository-url> grm-rbc
cd grm-rbc
```

## 2. Create the Python environment

### Option A: conda (recommended)

```bash
# CUDA 12.x machine
conda env create -f environment-cuda122.yml && conda activate grm-rbc-cu122

# CUDA 11.4 machine
conda env create -f environment-cuda114.yml && conda activate grm-rbc-cu114

# no GPU
conda env create -f environment-cpu.yml && conda activate grm-rbc-cpu
```

### Option B: pip and venv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

pip install -r requirements-cuda122.txt     # CUDA 12.x
pip install -r requirements-cuda114.txt     # CUDA 11.4 (use Python 3.8 or 3.9)
pip install -r requirements-cpu.txt         # CPU only
```

If the machine already has a working PyTorch (for example an older build on a shared CUDA 11.4 node), install only the core packages so that PyTorch is not replaced:

```bash
pip install -r requirements.txt
```

### Install the repository itself (both options)

```bash
pip install -e .
```

The editable install lets you import `grm_rbc`, `uccchess` and `strangefish` from any folder.

Always use `-e`. StrangeFish2 writes its game logs next to the package folder, and an editable install keeps those logs in the repository instead of in `site-packages`.

## 3. Install Stockfish

Both StrangeFish2 and the opponent bots use Stockfish. On Linux (x86-64) or macOS, run:

```bash
bash scripts/install_stockfish.sh          # newest supported release, best build for this CPU
export STOCKFISH_EXECUTABLE=$PWD/engines/stockfish
```

The script picks the fastest build your CPU supports (bmi2, avx2, sse4.1 or generic). It then checks that the binary starts. On Windows, download Stockfish from <https://stockfishchess.org/download/> and set `STOCKFISH_EXECUTABLE` to the `.exe`.

Add the `export` line to your shell profile. The alternative is to fill `stockfish_path` in the CONFIG block of each training script.

## 4. Check the installation

```bash
python scripts/check_installation.py
python -m pytest -q
```

The checker reports:
- the Python and package versions;
- whether PyTorch sees the GPUs;
- which estimator training script matches the installed PyTorch;
- whether Stockfish starts and offers the `Skill Level` option;
- whether the bot and its tournament configuration load.

The unit tests do not need Stockfish.

## 5. Choosing the estimator training script

| PyTorch on the machine | Script |
|---|---|
| older than 1.10 (CUDA 11.4 node) | `scripts/train_estimator_cuda114.py` |
| 2.x (CUDA 12.x) | `scripts/train_estimator_cuda122.py` |

The 11.4 script calls `torch.cuda.amp.autocast(enabled=...)` without a `dtype` argument, because older releases do not accept one. It avoids every API introduced after PyTorch 1.8.

The 12.2 script adds selectable bf16/fp16 autocast, TF32 and optional `torch.compile`.

Checkpoints written by either script load with either PyTorch version.

## Troubleshooting

**`KeyError: ... 'STOCKFISH_EXECUTABLE'` on import.** StrangeFish2 checks this variable when it is imported. Export it before running any script, or set `stockfish_path` in the script's CONFIG.

**`Illegal instruction` when Stockfish starts.** The binary was built for a newer CPU than the one you are running on. Install the generic build instead:

```bash
bash scripts/install_stockfish.sh sf_17.1 generic
```

**Engine processes killed, or out of memory, with many workers.** StrangeFish2 normally reserves a 2 GB hash table per game. The training scripts give each game `engine_hash_mb` (64 MB by default). Lower `workers` if memory is still short.

**`numpy` errors with PyTorch 1.x.** PyTorch 1.x needs NumPy below 2.0. The CUDA 11.4 environment files already pin this.

**`torch.load` complains about `weights_only`.** Checkpoints written by this code contain only tensors and plain Python values, so they load with recent PyTorch. Older checkpoints from other code may need `weights_only=False`.

**Replay window does not open.** The reconchess replay tool needs `pygame`, `setuptools` (for `pkg_resources`) and a display. Install them with:

```bash
pip install pygame setuptools
```
