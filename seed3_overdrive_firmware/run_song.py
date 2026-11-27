"""
run_song.py - song or guitar stem  ->  predicted pedal knobs  ->  Seed3.   (Raspberry Pi / laptop)

Put this file in the repo root next to predictOverdrive.py, expanded_features.py, pipeline.py
and send_knobs.py. It does the same prediction as predictOverdrive.py, then sends the three
knob values and waits for the Seed3 to confirm.

    python run_song.py --stem pipeline_output/stems/<song>/guitar.wav --port /dev/ttyACM0
    python run_song.py --url "https://www.youtube.com/watch?v=..." --port /dev/ttyACM0   (full pipeline)
    python run_song.py --stem guitar.wav --dry-run                                      (prediction only)

Options: --model overdrive_model.joblib   --level 0.5 (level is a manual knob; the model does not learn it)
"""
import argparse
import sys
from pathlib import Path


def predict_knobs(stem_wav, model_path, level):
    import joblib
    import librosa
    import numpy as np
    from expanded_features import extract_features

    bundle = joblib.load(model_path)  # only load models you trust
    sr = bundle["sample_rate"]
    window_s = bundle.get("window_seconds", 5.0)
    y, _ = librosa.load(stem_wav, sr=sr, mono=True)
    n = round(window_s * sr)
    if len(y) < n:
        raise ValueError(f"stem must be at least {window_s} s long (got {len(y) / sr:.1f} s)")
    starts = range(0, len(y) - n + 1, n)
    start = max(starts, key=lambda s: float(np.mean(y[s:s + n] ** 2)))   # loudest window
    feats = extract_features(y[start:start + n], sr)
    X = np.array([[feats[k] for k in bundle["features"]]], dtype=float)
    vals = np.clip(bundle["model"].predict(X)[0], 0.0, 1.0)
    pred = dict(zip(bundle["targets"], map(float, vals)))
    return pred["drive"], pred["tone"], float(level), start / sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stem", type=Path, help="guitar stem WAV (at least 5 s)")
    src.add_argument("--url", help="YouTube URL: runs pipeline.py first (download + stem split)")
    ap.add_argument("--model", type=Path, default=Path("overdrive_model.joblib"))
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--port", default=None, help="serial port of the Seed3; omit to only print the prediction")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto", action="store_true", help="with --url: no prompts")
    a = ap.parse_args()
    if not 0.0 <= a.level <= 1.0:
        ap.error("--level must be between 0 and 1")

    stem = a.stem
    if a.url:
        from pipeline import run_pipeline
        stem = Path(run_pipeline(a.url, auto=a.auto))
    if not stem.is_file():
        sys.exit(f"stem not found: {stem}")
    if not a.model.is_file():
        sys.exit(f"model not found: {a.model}  (train it with trainOverdrive.py, or copy overdrive_model.joblib here)")

    drive, tone, level, at = predict_knobs(stem, a.model, a.level)
    print(f"stem   : {stem}  (analysed window at {at:.1f} s)")
    print(f"drive  : {drive:.3f}\ntone   : {tone:.3f}\nlevel  : {level:.3f}  [manual]")

    from send_knobs import build_line, send
    line = build_line(drive, tone, level)
    print("message:", line.strip())
    if a.dry_run or not a.port:
        print("(not sent: " + ("dry run" if a.dry_run else "no --port") + ")")
        return 0
    reply = send(a.port, line)
    print("reply  :", reply)
    return 0 if reply and reply.startswith("OK") else 1


if __name__ == "__main__":
    sys.exit(main())
