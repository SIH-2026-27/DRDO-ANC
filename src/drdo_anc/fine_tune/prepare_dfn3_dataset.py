"""Extract split audio and prepare it in DeepFilterNet HDF5 format."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"


def load_audio_helpers():
    """Load the existing audio helpers without importing all enhancers."""

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load audio helper: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    audio_io = load_module(
        "drdo_anc_audio_io",
        SRC_DIR / "drdo_anc" / "audio" / "io.py",
    )
    resampling = load_module(
        "drdo_anc_audio_resampling",
        SRC_DIR / "drdo_anc" / "audio" / "resampling.py",
    )
    return (
        audio_io.load_mono_wav_bytes,
        audio_io.save_mono_wav,
        resampling.resample_mono,
    )


DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data" / "mvp" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "mvp" / "dfn3"
DEFAULT_DFN_ROOT = PROJECT_ROOT / "dfn3-model-files" / "deepfilternet"
TARGET_SAMPLE_RATE = 48_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract split audio and prepare DFN3 HDF5 datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the downloaded ZIP files.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Directory containing scan_filter.py CSV manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extracted WAVs, lists, HDF5 files, and config.",
    )
    parser.add_argument(
        "--dfn-root",
        type=Path,
        default=DEFAULT_DFN_ROOT,
        help="DeepFilterNet repository root.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Workers passed to DFN3 prepare_data.py.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit files per split/category for a small smoke test.",
    )
    parser.add_argument(
        "--skip-hdf5",
        action="store_true",
        help="Only extract WAVs and write lists; do not run prepare_data.py.",
    )
    return parser.parse_args()


def load_split_manifest(path: Path) -> pd.DataFrame:
    """Load one split manifest and validate the columns used here."""

    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    frame = pd.read_csv(path)
    required = {"archive_name", "internal_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing: {sorted(missing)}")

    return frame


def extract_audio(
    row: pd.Series,
    archive_dir: Path,
    output_path: Path,
    zip_cache: dict[str, zipfile.ZipFile],
    load_mono_wav_bytes,
    save_mono_wav,
    resample_mono,
) -> None:
    """Extract one archive member, convert it to mono 48 kHz, and save it."""

    archive_name = row["archive_name"]
    archive_path = archive_dir / archive_name

    if archive_name not in zip_cache:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        zip_cache[archive_name] = zipfile.ZipFile(archive_path)

    with zip_cache[archive_name].open(row["internal_path"]) as member:
        audio, sample_rate = load_mono_wav_bytes(member.read())

    audio = resample_mono(
        audio,
        source_sample_rate=sample_rate,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )
    save_mono_wav(output_path, audio, TARGET_SAMPLE_RATE)


def extract_split(
    frame: pd.DataFrame,
    *,
    split: str,
    kind: str,
    archive_dir: Path,
    output_dir: Path,
    max_files: int | None,
    zip_cache: dict[str, zipfile.ZipFile],
    load_mono_wav_bytes,
    save_mono_wav,
    resample_mono,
) -> list[Path]:
    """Extract one speech/noise split and return local WAV paths."""

    if max_files is not None:
        frame = frame.head(max_files)

    target_dir = output_dir / split / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for number, (_, row) in enumerate(
        tqdm(
            frame.iterrows(),
            total=len(frame),
            desc=f"Preparing {split}/{kind}",
            unit="file",
        )
    ):
        filename = Path(row["internal_path"]).stem
        output_path = target_dir / f"{number:06d}_{filename}.wav"
        if output_path.is_file():
            paths.append(output_path.resolve())
            continue
        extract_audio(
            row,
            archive_dir=archive_dir,
            output_path=output_path,
            zip_cache=zip_cache,
            load_mono_wav_bytes=load_mono_wav_bytes,
            save_mono_wav=save_mono_wav,
            resample_mono=resample_mono,
        )
        paths.append(output_path.resolve())

    return paths


def write_file_list(path: Path, audio_paths: list[Path]) -> None:
    """Write one absolute WAV path per line for prepare_data.py."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{audio_path}\n" for audio_path in audio_paths),
        encoding="utf-8",
    )


def write_dataset_config(output_dir: Path) -> Path:
    """Create the dataset.cfg expected by DFN3 train.py."""

    config = {
        split: [
            [f"speech_{split}.hdf5", 1.0],
            [f"noise_{split}.hdf5", 1.0],
        ]
        for split in ("train", "valid", "test")
    }

    config_path = output_dir / "dataset.cfg"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def prepare_hdf5(
    *,
    dfn_root: Path,
    output_dir: Path,
    file_lists: dict[tuple[str, str], Path],
    num_workers: int,
) -> None:
    """Run the upstream DFN3 converter for every split and category."""

    deepfilternet_dir = dfn_root / "DeepFilterNet"
    prepare_data = deepfilternet_dir / "df" / "scripts" / "prepare_data.py"
    dfn3_model = dfn_root / "DeepFilterNet" / "df" / "deepfilternet3.py"

    if not dfn3_model.is_file():
        raise FileNotFoundError(
            f"DFN3 model implementation not found: {dfn3_model}"
        )

    if not prepare_data.is_file():
        raise FileNotFoundError(f"DFN3 prepare_data.py not found: {prepare_data}")

    if sys.platform == "darwin" and num_workers > 1:
        print("macOS detected; using serial HDF5 preparation for DFN3 compatibility.")
        num_workers = 0

    jobs = [
        (split, kind)
        for split in ("train", "valid", "test")
        for kind in ("speech", "noise")
    ]

    for split, kind in tqdm(jobs, desc="Creating HDF5 files", unit="file"):
        list_path = file_lists[(split, kind)]
        hdf5_path = output_dir / f"{kind}_{split}.hdf5"
        command = [
            sys.executable,
            "-m",
            "df.scripts.prepare_data",
            "--sr",
            str(TARGET_SAMPLE_RATE),
            "--mono",
            "--num_workers",
            str(num_workers),
            kind,
            str(list_path),
            str(hdf5_path),
        ]

        print("Running:", " ".join(command))
        subprocess.run(
            command,
            cwd=deepfilternet_dir,
            check=True,
        )


def main():
    started_at = time.perf_counter()
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_mono_wav_bytes, save_mono_wav, resample_mono = load_audio_helpers()
    zip_cache = {}
    file_lists = {}

    extraction_started_at = time.perf_counter()
    try:
        for split in ("train", "valid", "test"):
            for kind in ("speech", "noise"):
                manifest_path = (
                    args.splits_dir.resolve()
                    / f"{split}_{kind}.csv"
                )
                manifest = load_split_manifest(manifest_path)
                audio_paths = extract_split(
                    manifest,
                    split=split,
                    kind=kind,
                    archive_dir=args.dataset_dir.resolve(),
                    output_dir=output_dir / "wav",
                    max_files=args.max_files,
                    zip_cache=zip_cache,
                    load_mono_wav_bytes=load_mono_wav_bytes,
                    save_mono_wav=save_mono_wav,
                    resample_mono=resample_mono,
                )
                list_path = output_dir / f"{kind}_{split}.txt"
                write_file_list(list_path, audio_paths)
                file_lists[(split, kind)] = list_path
                print(f"{split}/{kind}: {len(audio_paths):,} WAV files")
    finally:
        for archive in zip_cache.values():
            archive.close()

    print(
        f"WAV extraction completed in "
        f"{time.perf_counter() - extraction_started_at:.1f} seconds."
    )

    config_path = write_dataset_config(output_dir)

    if not args.skip_hdf5:
        hdf5_started_at = time.perf_counter()
        prepare_hdf5(
            dfn_root=args.dfn_root.resolve(),
            output_dir=output_dir,
            file_lists=file_lists,
            num_workers=args.num_workers,
        )
        print(
            f"HDF5 conversion completed in "
            f"{time.perf_counter() - hdf5_started_at:.1f} seconds."
        )

    print(f"\nDFN3 dataset directory: {output_dir}")
    print(f"Dataset config: {config_path}")
    print(f"Total time: {time.perf_counter() - started_at:.1f} seconds.")


if __name__ == "__main__":
    main()
