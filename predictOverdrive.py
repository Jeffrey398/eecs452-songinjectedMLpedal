"""Predict Positive Grid-style overdrive knob settings from a recording."""
import argparse
from pathlib import Path
import joblib
import librosa
import numpy as np
from expanded_features import extract_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio_file", type=Path)
    p.add_argument("--model", type=Path, default=Path("overdrive_model.joblib"))
    p.add_argument("--port", default=None, help="Optional USB serial device, only if Seed3 firmware expects 3 normalized knobs")
    args = p.parse_args()
    bundle = joblib.load(args.model)  # Load only models you trust.
    sr = bundle["sample_rate"]
    y, _ = librosa.load(args.audio_file, sr=sr, mono=True)
    n = 5 * sr
    if len(y) < n:
        raise ValueError("Recording must be at least 5 seconds")
    starts = range(0, len(y) - n + 1, n)
    start = max(starts, key=lambda s: np.mean(y[s:s + n] ** 2))
    features = extract_features(y[start:start + n], sr)
    X = np.array([[features[k] for k in bundle["features"]]], dtype=float)
    values = np.clip(bundle["model"].predict(X)[0], 0.0, 1.0)
    names = bundle["targets"]
    for name, value in zip(names, values):
        print(f"{name}: {value:.3f} ({value * 100:.1f}%)")
    message = ",".join(f"{v:.4f}" for v in values) + "\n"
    print("Serial message (drive,tone,level):", message.strip())
    if args.port:
        import serial
        with serial.Serial(args.port, 115200, timeout=2) as ser:
            ser.write(message.encode("ascii"))
            ser.flush()
        print("Settings sent (receiver must support this three-value protocol)")


if __name__ == "__main__":
    main()
