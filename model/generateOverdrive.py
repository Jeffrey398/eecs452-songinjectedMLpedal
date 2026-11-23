"""Extract 94 features and pedal labels from recorded overdrive WAV files."""
import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from ML_Pedal.model.expanded_features import FEATURE_COLUMNS, SR, extract_features

PATTERN = re.compile(r"(?:^|_)drive(\d{1,3})_tone(\d{1,3})_level(\d{1,3})$", re.I)
TARGET_COLUMNS = ["drive", "tone", "level"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("overdrive"))
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=Path("overdrive_training_data.csv"))
    args = p.parse_args()
    if args.duration <= 0:
        p.error("--duration must be positive")
    n = round(args.duration * SR)
    rows = []
    paths = sorted(x for x in args.input_dir.rglob("*.wav") if x.is_file())
    if not paths:
        p.error(f"No WAV files found under {args.input_dir}")
    for path in paths:
        match = PATTERN.search(path.stem)
        if not match:
            print(f"Skipping filename without drive/tone/level labels: {path.name}")
            continue
        knobs = [int(v) for v in match.groups()]
        if any(v > 100 for v in knobs):
            print(f"Skipping out-of-range labels: {path.name}")
            continue
        y, _ = librosa.load(path, sr=SR, mono=True)
        source = str(path.relative_to(args.input_dir))
        count = 0
        # Nonoverlapping windows prevent overlapping examples from the same recording.
        # segments from the same recording are not independent performances
        for start in range(0, len(y) - n + 1, n):
            clip = y[start:start + n]
            if np.sqrt(np.mean(clip.astype(np.float64) ** 2)) < 1e-4:
                continue
            features = extract_features(clip, SR)
            rows.append({"source_file": source, "start_seconds": start / SR,
                         **features, **dict(zip(TARGET_COLUMNS, [v / 100 for v in knobs]))})
            count += 1
        print(f"{path.name}: {count} examples")
    if not rows:
        raise RuntimeError("No usable labeled audio segments found")
    columns = ["source_file", "start_seconds"] + FEATURE_COLUMNS + TARGET_COLUMNS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} examples from {len(set(r['source_file'] for r in rows))} recordings to {args.output}")


if __name__ == "__main__":
    main()
