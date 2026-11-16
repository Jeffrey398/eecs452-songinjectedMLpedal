
"""Generate Model 2 training data from Moises guitar stems."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from expanded_features import (
    extract_features,
    SR,
)

from generate_training_data import (
    apply_eq,
    apply_distortion,
    apply_compression,
    TARGET_COLUMNS,
)

ROOT = Path(__file__).resolve().parent

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".aif", ".aiff"
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "trainingData" / "genre",
    )

    parser.add_argument(
        "--examples-per-file",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    files = sorted(
        p for p in args.input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    print(f"Found {len(files)} audio files")

    if not files:
        raise RuntimeError(
            f"No audio files found in {args.input_dir}"
        )

    rng = np.random.default_rng(args.seed)

    rows = []
    duration_samples = int(SR * args.duration)

    for file_index, path in enumerate(files, start=1):
        try:
            y, sr = librosa.load(
                path,
                sr=SR,
                mono=True,
            )
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

        if len(y) < duration_samples:
            print(f"Skipping short file: {path.name}")
            continue

        # Preserve the folder-based genre label
        relative_path = path.relative_to(args.input_dir)
        genre = relative_path.parts[0] if len(
            relative_path.parts
        ) > 1 else "unknown"

        # Unique identifier for each source recording
        source_id = relative_path.as_posix()

        for example_id in range(args.examples_per_file):

            start = int(
                rng.integers(
                    0,
                    len(y) - duration_samples + 1,
                )
            )

            clip = y[
                start:start + duration_samples
            ].copy()

            # Random additional effects
            bass, mid, treble = rng.uniform(
                -9, 9, size=3
            )

            drive = float(rng.uniform(0, 1))

            threshold = float(
                rng.uniform(-35, -10)
            )

            ratio = float(rng.uniform(1, 6))

            # Apply the same DSP as Model 1
            processed = apply_eq(
                clip, sr, bass, mid, treble
            )

            processed = apply_distortion(
                processed, drive
            )

            processed = apply_compression(
                processed, sr, threshold, ratio
            )

            processed = np.clip(
                processed, -1.0, 1.0
            )

            # Extract 94 audio features
            features = extract_features(
                processed, sr
            )

            row = {
                "source_file": source_id,
                "genre": genre,
                "example_id": example_id,
            }

            row.update(features)

            row.update({
                "bass_gain_db": bass,
                "mid_gain_db": mid,
                "treble_gain_db": treble,
                "drive": drive,
                "compression_threshold_db": threshold,
                "compression_ratio": ratio,
            })

            rows.append(row)

        print(
            f"[{file_index}/{len(files)}] "
            f"{genre} / {path.name} "
            f"-> {args.examples_per_file} examples",
            flush=True,
        )

    if not rows:
        raise RuntimeError(
            "No training examples generated."
        )

    output = ROOT / "labeled_training_data_v2.csv"

    columns = (
        ["source_file", "genre", "example_id"]
        + FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    df = pd.DataFrame(rows, columns=columns)

    df.to_csv(output, index=False)

    print()
    print(f"Saved: {output}")
    print(f"Training examples: {len(df)}")
    print(f"Unique recordings: {df['source_file'].nunique()}")
    print(f"Features per example: {len(FEATURE_COLUMNS)}")
    print()
    print("Genre distribution:")
    print(df["genre"].value_counts())


if __name__ == "__main__":
    main()
