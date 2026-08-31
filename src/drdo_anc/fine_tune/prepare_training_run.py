"""Create a resumable DFN3 fine-tuning run from the pretrained archive."""

import argparse
import re
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def update_config(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    section = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].lower()
            continue
        if section != "train" or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().lower()
        if key in values:
            lines[index] = f"{key} = {values[key]}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    archive = args.model_archive.resolve()
    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    config_path = run_dir / "config.ini"
    checkpoint_path = checkpoint_dir / "model_120.ckpt.best"

    if not archive.is_file():
        raise FileNotFoundError(f"Model archive not found: {archive}")

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        config_member = next(
            name for name in source.namelist() if name.endswith("/config.ini")
        )
        checkpoint_member = next(
            name
            for name in source.namelist()
            if name.endswith("/checkpoints/model_120.ckpt.best")
        )

        if not config_path.exists():
            config_path.write_bytes(source.read(config_member))
        if not checkpoint_path.exists():
            checkpoint_path.write_bytes(source.read(checkpoint_member))

    checkpoint_epochs = [
        int(match.group(1))
        for path in checkpoint_dir.glob("model_*.ckpt*")
        if (match := re.search(r"model_(\d+)", path.name))
    ]
    start_epoch = max(checkpoint_epochs, default=0)

    update_config(
        config_path,
        {
            "device": args.device,
            # DFN3 resumes from the epoch encoded in the checkpoint.
            "max_epochs": str(start_epoch + args.max_epochs),
            "batch_size": str(args.batch_size),
            "num_workers": str(args.train_workers),
        },
    )

    print(f"Run directory: {run_dir}")
    print(f"Config:       {config_path}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Target epoch:   {start_epoch + args.max_epochs}")


if __name__ == "__main__":
    main()
