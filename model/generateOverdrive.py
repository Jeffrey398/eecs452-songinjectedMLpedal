"""Extract 94 features and pedal labels from recorded overdrive WAV files."""
import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from ML_Pedal.model.expanded_features import FEATURE_COLUMNS, SR, extract_features

PATTERN = re.compile(r"^a_(.+)_g([0-5])_t([0-5])$", re.I)
TARGET_COLUMNS = ["drive", "tone"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("overdrive"))
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=Path("overdrive_training_data.csv"))
    p.add_argument("--pedal", type=str, default=None)
    args = p.parse_args()
    if args.duration <= 0:
        p.error("--duration must be positive")
    n = round(args.duration * SR)
    rows = []
    paths = sorted(x for x in args.input_dir.rglob("*.wav") if x.is_file())
    if not paths:
        p.error(f"No WAV files found under {args.input_dir}")
    for path in paths:
        match = PATTERN.fullmatch(path.stem)

        if not match:
            continue

        pedal, gain, tone = match.groups()
        pedal = pedal.lower()

        drive = int(gain) / 5.0
        tone = int(tone) / 5.0
        if args.pedal and pedal != args.pedal.lower():
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
            rows.append({
                "source_file": source,
                "pedal": pedal,
                "start_seconds": start / SR,
                **features,
                "drive": drive,
                "tone": tone
            })
            count += 1
        print(f"{path.name}: {count} examples")
    if not rows:
        raise RuntimeError("No usable labeled audio segments found")
    columns = ["source_file","pedal","start_seconds"] + FEATURE_COLUMNS + TARGET_COLUMNS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} examples from {len(set(r['source_file'] for r in rows))} recordings to {args.output}")


if __name__ == "__main__":
    main()
