#script to download entire dataset locally
#run script locally to download all files locally during fine-tune run

import argparse
from pathlib import Path
DEFAULT_REPO_ID = "Panav-Payappagoudar/sih-26-processed-audio"
DEFAULT_OUTPUT_DIR = Path("dataset")


def parse_args() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description="Download the complete Hugging Face dataset repository."
    )

    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository ID.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for the downloaded repository.",
    )

    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset branch, tag, or commit.",
    )

    return parser.parse_args()
#main
def main() -> None:
    """Download all files from the dataset repository."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Install it with:\n"
            "python3 -m pip install huggingface_hub"
        ) from error

    args = parse_args()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True, exist_ok=True,
    )

    print(f"Repository: {args.repo_id}")
    print(f"Revision:   {args.revision}")
    print(f"Output:     {output_dir}")
    print("\nDownloading all repository files...\n")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output_dir),
    )

    downloaded_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
    )

    print("\nDownload complete.")
    print(f"Downloaded files: {len(downloaded_files)}")

    for file_path in downloaded_files:
        relative_path = file_path.relative_to(output_dir)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"{relative_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()