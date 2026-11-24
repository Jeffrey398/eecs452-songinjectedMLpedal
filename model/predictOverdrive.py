
import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np

from ML_Pedal.model.expanded_features import extract_features


def main():
    p = argparse.ArgumentParser()

    p.add_argument("audio_file", type=Path)

    p.add_argument(
        "--model",
        type=Path,
        default=Path("pod_bluesdriver_model.joblib")
    )

    p.add_argument(
        "--level",
        type=float,
        default=0.5,
        help="Manual output level between 0 and 1"
    )

    p.add_argument(
        "--port",
        default=None,
        help="Optional USB serial port for Seed3"
    )

    args = p.parse_args()

    if not 0.0 <= args.level <= 1.0:
        p.error("--level must be between 0 and 1")

    if not args.audio_file.is_file():
        p.error(f"Audio file not found: {args.audio_file}")

    if not args.model.is_file():
        p.error(f"Model not found: {args.model}")

    # Load trained model
    bundle = joblib.load(args.model)

    sr = bundle["sample_rate"]
    window_seconds = bundle.get("window_seconds", 5.0)

    # Load reference guitar recording
    y, _ = librosa.load(
        args.audio_file,
        sr=sr,
        mono=True
    )

    n = round(window_seconds * sr)

    if len(y) < n:
        raise ValueError(
            f"Recording must be at least {window_seconds} seconds"
        )

    # Select the highest-energy audio window
    starts = range(0, len(y) - n + 1, n)

    start = max(
        starts,
        key=lambda s: np.mean(y[s:s+n] ** 2)
    )

    clip = y[start:start+n]

    # Extract audio features
    features = extract_features(clip, sr)

    # Match the feature ordering used during training
    X = np.array(
        [[features[k] for k in bundle["features"]]],
        dtype=float
    )

    # Predict normalized pedal settings
    values = np.clip(
        bundle["model"].predict(X)[0],
        0.0,
        1.0
    )

    settings = dict(
        zip(bundle["targets"], values)
    )

    drive = float(settings["drive"])
    tone = float(settings["tone"])
    level = args.level

    # Display results
    print("\nPredicted pedal settings:")
    print(f"Drive: {drive:.4f} ({drive * 100:.1f}%)")
    print(f"Tone:  {tone:.4f} ({tone * 100:.1f}%)")
    print(f"Level: {level:.4f} ({level * 100:.1f}%) [manual]")

    print(f"\nSelected audio window: {start / sr:.2f}s")

    # Format message for Seed3
    message = f"{drive:.4f},{tone:.4f},{level:.4f}\n"

    print("\nSerial message (drive,tone,level):")
    print(message.strip())

    # Optional serial transmission
    if args.port:
        import serial

        with serial.Serial(
            args.port,
            115200,
            timeout=2
        ) as ser:
            ser.write(message.encode("ascii"))
            ser.flush()

        print(f"\nSent settings to {args.port}")


if __name__ == "__main__":
    main()
