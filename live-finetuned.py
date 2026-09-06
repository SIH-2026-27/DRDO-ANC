#!/usr/bin/env python3
"""Simple microphone -> DFN3 -> speaker test for a fine-tuned checkpoint."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch


ROOT = Path(__file__).resolve().parent
DFN_ROOT = ROOT / "dfn3-model-files" / "deepfilternet" / "DeepFilterNet"
sys.path.insert(0, str(DFN_ROOT))
from df.enhance import enhance, init_df  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing config.ini and checkpoints/")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--seconds-per-block", type=float, default=1.0)
    args = parser.parse_args()

    model, state, _, epoch = init_df(str(args.model_dir), epoch="best", log_file=None)
    sample_rate = state.sr()
    block_size = int(sample_rate * args.seconds_per_block)
    if block_size <= 0:
        raise ValueError("--seconds-per-block must be positive")
    print(f"Loaded epoch {epoch} at {sample_rate} Hz. Speak into the microphone; press Ctrl+C to stop.")
    print(sd.query_devices())

    model.eval()
    with sd.Stream(samplerate=sample_rate, blocksize=block_size, channels=1, dtype="float32", device=(args.input_device, args.output_device)) as stream:
        while True:
            noisy, _ = stream.read(block_size)
            audio = torch.from_numpy(np.asarray(noisy[:, 0], dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                clean = enhance(model, state, audio, pad=True)
            output = clean.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            stream.write(output.reshape(-1, 1))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(DFN_ROOT))
    main()
