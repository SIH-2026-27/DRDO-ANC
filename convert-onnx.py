#!/usr/bin/env python3
"""Export a fine-tuned DeepFilterNet checkpoint in the official DFN3 format."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DFN_ROOT = ROOT / "dfn3-model-files" / "deepfilternet" / "DeepFilterNet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_122.ckpt.best")
    parser.add_argument("--config", type=Path, required=True, help="Matching DFN3 config.ini")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "dfn3-epoch-122-onnx")
    parser.add_argument("--epoch", type=int, default=122)
    args = parser.parse_args()

    for path in (args.checkpoint, args.config):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    model_dir = args.output_dir / "_export_model"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_text = args.config.read_text()
    # Export is performed on CPU here; the training config may still say cuda.
    config_text = re.sub(r"(?im)^(\s*device\s*=\s*).*$", r"\1cpu", config_text)
    (model_dir / "config.ini").write_text(config_text)
    shutil.copy2(args.checkpoint, checkpoint_dir / f"model_{args.epoch}.ckpt.best")

    export_dir = args.output_dir / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DFN_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [
        sys.executable, "-m", "df.scripts.export", "--model-base-dir", str(model_dir),
        "--epoch", str(args.epoch), str(export_dir), "--opset", "14",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=DFN_ROOT, env=env, check=True)
    archives = sorted(export_dir.glob("*_onnx.tar.gz"))
    print(f"Export complete: {archives[-1] if archives else export_dir}")


if __name__ == "__main__":
    main()
