"""Create clean-speech and noise train/valid/test manifests."""

import argparse
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "mvp" / "splits"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter metadata.csv and create dataset splits."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where split files will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()


def assign_group_splits(groups, seed, train_ratio, valid_ratio):
    """Assign complete groups to approximately 80/10/10 splits."""

    test_ratio = 1.0 - train_ratio - valid_ratio
    group_sizes = groups.groupby("group_id").size()
    group_names = list(group_sizes.index)

    random.Random(seed).shuffle(group_names)
    group_names.sort(
        key=lambda name: group_sizes[name],
        reverse=True,
    )

    targets = {
        "train": len(groups) * train_ratio,
        "valid": len(groups) * valid_ratio,
        "test": len(groups) * test_ratio,
    }
    counts = {split: 0 for split in targets}
    assignments = {}

    for name in group_names:
        split = min(
            ("train", "valid", "test"),
            key=lambda item: counts[item] / targets[item],
        )
        assignments[name] = split
        counts[split] += group_sizes[name]

    return groups["group_id"].map(assignments)


def main():
    args = parse_args()

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if min(args.train_ratio, args.valid_ratio, test_ratio) <= 0:
        raise ValueError(
            "Train, valid, and test ratios must all be greater than zero."
        )

    metadata_path = args.dataset_dir.resolve() / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Could not find metadata.csv at {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)
    required = {
        "archive_name",
        "internal_path",
        "audio_class",
        "dataset_source",
        "inferred_subclass",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"metadata.csv is missing: {sorted(missing)}"
        )

    # MS-SNSD clean_train and clean_test are labelled as noise in metadata.
    ms_snsd_clean = (
        metadata["dataset_source"].eq("MS-SNSD-Complex-Noise")
        & metadata["inferred_subclass"].isin(
            ["clean_train", "clean_test"]
        )
    )
    clean = metadata["audio_class"].eq("clean_speech") | ms_snsd_clean
    noise = metadata["audio_class"].eq("noise") & ~ms_snsd_clean

    selected = metadata.loc[clean | noise].copy()
    selected["kind"] = "noise"
    selected.loc[clean.loc[selected.index], "kind"] = "speech"

    # Keep speakers, recording conditions, or source files together.
    selected["group_id"] = (
        selected["dataset_source"] + ":" + selected["internal_path"]
    )

    speech_speaker = selected["dataset_source"].isin(
        ["English-with-various-accents", "LibriSpeech"]
    )
    selected.loc[speech_speaker, "group_id"] = (
        selected.loc[speech_speaker, "dataset_source"]
        + ":"
        + selected.loc[speech_speaker, "inferred_subclass"]
    )

    grouped_noise = selected["dataset_source"].isin(
        [
            "DEMAND-Background-Noise",
            "firearms-audio-dataset-contains-58-guntypes",
        ]
    ) & selected["kind"].eq("noise")
    
    selected.loc[grouped_noise, "group_id"] = (
        selected.loc[grouped_noise, "dataset_source"]
        + ":"
        + selected.loc[grouped_noise, "inferred_subclass"]
    )

    selected["split"] = ""

    # Split every source/category independently to preserve source diversity.
    for (kind, source), subset in selected.groupby(
        ["kind", "dataset_source"],
        sort=True,
    ):
        selected.loc[subset.index, "split"] = assign_group_splits(
            subset,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected.to_csv(
        output_dir / "split_manifest.csv",
        index=False,
    )

    for split in ("train", "valid", "test"):
        for kind in ("speech", "noise"):
            subset = selected[
                selected["split"].eq(split)
                & selected["kind"].eq(kind)
            ]
            subset.to_csv(
                output_dir / f"{split}_{kind}.csv",
                index=False,
            )

    print(f"Metadata: {metadata_path}")
    print(f"Output:   {output_dir}")
    print(f"Selected: {len(selected):,} files")
    print(
        selected.groupby(["split", "kind"])
        .size()
        .unstack(fill_value=0)
        .reindex(["train", "valid", "test"])
        .to_string()
    )


if __name__ == "__main__":
    main()
