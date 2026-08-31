# DFN3 fine-tuning pipeline

The entrypoint is `run_finetune.sh`. It is designed for Windows users running
Ubuntu in WSL2 with an NVIDIA GPU. Native Windows Git Bash is not a supported
DeepFilterNet training environment.

The pipeline performs the following sequence:

1. Installs missing WSL system tools (Git, Python, compilers, Rust, HDF5,
   certificates, and curl), then initializes the pinned DeepFilterNet Git
   submodule and creates or reuses a Python 3.12 virtual environment, falling
   back to Python 3.11 or the distribution's generic `python3` package when
   necessary.
2. Installs pinned PyTorch CUDA, audio, HDF5, scientific Python, and
   DeepFilterNet dependencies.
3. Verifies that PyTorch can see the NVIDIA GPU.
4. Runs `scan_filter.py` to create deterministic train/valid/test manifests.
5. Runs `prepare_dfn3_dataset.py` to extract mono 48 kHz WAV files and create
   six DFN3 HDF5 files plus `dataset.cfg`.
6. Creates a fine-tuning run directory from the pretrained DFN3 archive,
   preserving the pretrained checkpoint and configuration.
7. Starts the upstream DFN3 `train.py` with CUDA and the prepared dataset.
8. Resumes automatically from checkpoints already present in the run directory.

Useful overrides:

```bash
MAX_EPOCHS=10 BATCH_SIZE=32 PREPARE_WORKERS=3 TRAIN_WORKERS=3 \
  bash src/drdo_anc/fine_tune/run_finetune.sh
```

Generated artifacts are intentionally excluded from Git: downloaded ZIPs,
converted WAVs, HDF5 files, checkpoints, logs, and training outputs.

The pretrained archive is expected at
`dfn3-model-files/deepfilternet/models/DeepFilterNet3.zip`. Because model
weights are ignored as large artifacts, copy that archive into the indicated
location or pass a different path with `MODEL_ARCHIVE=/path/to/archive.zip`.
