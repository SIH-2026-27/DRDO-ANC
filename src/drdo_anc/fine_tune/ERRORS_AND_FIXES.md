# DFN3 fine-tuning pipeline: errors, fixes, and worker settings

This document records the errors encountered while preparing the DeepFilterNet3
fine-tuning pipeline and explains the final behavior of each component.

## Final recommended platform

For a Windows computer with an NVIDIA GPU, run the pipeline inside **WSL2
Ubuntu** with NVIDIA CUDA support. Use the provided Bash launcher from WSL2:

```bash
bash src/drdo_anc/fine_tune/run_finetune.sh
```

Native PowerShell and native Git Bash are not the target environments for the
DeepFilterNet training stack. The upstream project documents training as Linux-
only. The launcher therefore checks for Linux and `nvidia-smi` before it starts.

If the WSL distribution does not provide a `python3.12` apt package, the
launcher automatically falls back to Python 3.11 and then the distribution's
generic `python3` package. If all of these are unavailable, the apt sources or
the WSL distribution need to be repaired before continuing.

## What the launcher does

`run_finetune.sh` performs the complete sequence:

1. Installs WSL system dependencies, including Python 3.12, build tools, and
   HDF5 headers.
2. Creates or reuses `.venv-dfn3-cuda`.
3. Installs pinned PyTorch CUDA wheels and Python dependencies.
4. Checks that PyTorch can see an NVIDIA CUDA device.
5. Downloads the public Hugging Face dataset when `dataset/metadata.csv` is
   missing.
6. Runs `scan_filter.py` to create deterministic train/valid/test manifests.
7. Runs `prepare_dfn3_dataset.py` to create mono 48 kHz WAV files, file lists,
   six HDF5 files, and `dataset.cfg`.
8. Extracts the pretrained DFN3 config and checkpoint into the run directory.
9. Applies only the requested fine-tuning overrides: device, epochs, batch
   size, and training workers.
10. Starts the upstream DFN3 `train.py` using CUDA.

The default fine-tuning run performs 10 additional epochs and uses batch size
32. Because the pretrained checkpoint is epoch 120, the first run targets epoch
130. Override them with
environment variables, for example:

```bash
MAX_EPOCHS=20 BATCH_SIZE=64 bash src/drdo_anc/fine_tune/run_finetune.sh
```

## `num_workers` explained

`num_workers` controls how many worker processes read and prepare audio in
parallel. It does not change model size, batch size, epoch count, or the number
of CUDA devices.

The default is now **3**:

```text
prepare_dfn3_dataset.py: 3 preparation workers
run_finetune.sh:          3 preparation workers
run_finetune.sh:          3 training data-loader workers
```

On Linux/WSL2, the requested value of 3 is passed to DFN3. On macOS, the
preparation script automatically changes the effective HDF5 worker count to 0
because macOS uses Python's `spawn` multiprocessing method and the upstream
DFN3 preparation code is not safe with that worker function. The command may
still specify 3, but the script prints that it is using serial preparation.

Worker trade-offs:

| Value | Behavior |
|---:|---|
| 0 | Fully serial; slowest but safest and lowest memory use |
| 1 | One worker; useful on Linux for simple debugging |
| 3 | Default; good starting point on a CUDA Linux/WSL2 machine |
| 4+ | Potentially faster, but uses more CPU, RAM, file handles, and disk bandwidth |

The training worker setting is separate from HDF5 conversion. It is written to
the copied DFN3 `config.ini` as `train.num_workers`.

## Errors encountered and fixes

### 1. `AudioMetaData` import failure

Error:

```text
ImportError: cannot import name 'AudioMetaData' from 'torchaudio'
ModuleNotFoundError: No module named 'torchaudio.backend'
```

Cause: the original Python 3.13 environment installed a newer Torchaudio
release whose legacy audio metadata API had been removed. The bundled DFN3
source imports that older API.

Fix: use Python 3.12 with compatible pinned versions:

```text
torch==2.5.1
torchaudio==2.5.1
```

The CUDA launcher installs the matching CUDA wheels instead of using the
incompatible Python 3.13 environment.

### 2. Missing `loguru`

Error:

```text
ModuleNotFoundError: No module named 'loguru'
```

Fix: pin and install `loguru==0.7.3` in the pipeline environment.

### 3. Missing `libdf`

Error:

```text
ModuleNotFoundError: No module named 'libdf'
```

Cause: `libdf` is supplied by DeepFilterNet's `deepfilterlib` package.

Fix: install `deepfilterlib==0.5.6`. The launcher also places the local
DeepFilterNet Python source on `PYTHONPATH` before training.

### 4. Missing HDF5 Python support

Error:

```text
ModuleNotFoundError: No module named 'h5py'
```

Fix: install `h5py==3.11.0`.

### 5. NumPy version conflict

Error:

```text
deepfilternet requires numpy<2.0, but numpy 2.x is installed
```

Fix: pin:

```text
numpy==1.26.4
```

### 6. SciPy version conflict

Problem: a newer SciPy release pulled in NumPy 2.x, conflicting with
DeepFilterNet.

Fix: pin:

```text
scipy==1.14.1
```

### 7. macOS multiprocessing pickling failure

Error:

```text
AttributeError: Can't get attribute '_check_file'
```

Cause: the upstream `prepare_data.py` defines `_check_file` inside its main
block. That function cannot be pickled by macOS's `spawn` multiprocessing
implementation.

Fix: the project preparation script forces serial HDF5 preparation on macOS.
The vendored DeepFilterNet source itself was restored and is not modified.

### 8. macOS HDF5/DataLoader hang

Problem: even after avoiding the first worker failure, PyTorch's HDF5 DataLoader
could remain alive after writing a sample when multiprocessing was enabled.

Fix: on macOS the project script passes `--num_workers 0` to the upstream
converter. Linux/WSL2 keeps the requested default of 3.

### 9. `deepfilterdataloader` build failure on macOS

Error:

```text
Invalid H5_VERSION: "2.2.0"
```

Cause: the old Rust HDF5 bindings used by the DeepFilterNet training data
loader do not understand Homebrew HDF5 2.2.

Fix for the supported target: install the loader in Linux/WSL2, where the
system HDF5 development package is compatible. The macOS environment is used
only for preparation experiments, not as the supported full training target.

### 10. HDF5 warning about SWMR

Warning:

```text
swmr=True only affects read mode
```

This comes from the upstream converter and does not prevent HDF5 creation. It
is informational for this pipeline.

## Output locations

The generated dataset is written to:

```text
data/mvp/dfn3/
```

It contains converted WAVs, file lists, six HDF5 files, and `dataset.cfg`.

The fine-tuning run is written to:

```text
data/mvp/finetune/dfn3-custom/
```

It contains `config.ini`, logs, summaries, and checkpoints. The pretrained
checkpoint is copied as `checkpoints/model_120.ckpt.best` on the first run.

## Resume behavior

The WAV preparation stage reuses existing output WAV files. The fine-tuning
stage reuses checkpoints in the run directory because upstream DFN3 resumes by
default. The launcher interprets `MAX_EPOCHS` as additional epochs and computes
the target from the latest checkpoint epoch. Do not delete the run directory if
you want to continue an interrupted training run.

If the manifests or dataset contents change, use a new `RUN_DIR` and
`DATA_DIR` to avoid mixing old HDF5 data or checkpoints with the new dataset.

## Large-file policy

The following are ignored by Git:

```text
dataset/
data/mvp/dfn3/
data/mvp/finetune/
*.wav
*.hdf5
*.ckpt
*.onnx
*.zip
virtual environments
logs and generated text outputs
```

The pretrained `DeepFilterNet3.zip` archive must therefore be copied separately
to the expected model directory or supplied through `MODEL_ARCHIVE`.

## Validation performed

The following checks pass in the compatible local environment:

- Python compilation of the fine-tuning scripts.
- Bash syntax validation of the launcher.
- One-file-per-split extraction smoke test.
- Six-file HDF5 creation smoke test.
- Pretrained config/checkpoint extraction smoke test.
- Git-ignore checks for datasets, HDF5 files, checkpoints, and environments.
