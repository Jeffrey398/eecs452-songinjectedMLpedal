
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "trainingData/input/a_0-input.wav"
OUTPUT_DIR = ROOT / "trainingData/bluesdriver"

EXPECTED_SR = 48000

WINDOW_SECONDS = 5
MAX_LAG_SECONDS = 0.1

# Check multiple sections for consistency
START_TIMES = [10, 30, 70, 120, 200, 300]


def load_segment(path, start, duration):

    with sf.SoundFile(str(path)) as f:

        sr = f.samplerate

        if sr != EXPECTED_SR:
            raise ValueError(
                f"Unexpected sample rate: {sr}"
            )

        if f.channels != 1:
            raise ValueError("Expected mono audio")

        f.seek(int(start * sr))

        audio = f.read(
            int(duration * sr),
            dtype="float64"
        )

    return audio, sr


def preprocess(audio, sr):

    # Remove DC and emphasize shared guitar frequencies
    sos = butter(
        2,
        [100, 3000],
        btype="bandpass",
        fs=sr,
        output="sos"
    )

    audio = sosfiltfilt(sos, audio)

    audio -= np.mean(audio)

    return audio


def estimate_lag(x, y, sr):

    x = preprocess(x, sr)
    y = preprocess(y, sr)

    corr = correlate(
        y, x,
        mode="full",
        method="fft"
    )

    lags = correlation_lags(
        len(y),
        len(x),
        mode="full"
    )

    max_lag = int(MAX_LAG_SECONDS * sr)

    valid = np.abs(lags) <= max_lag

    corr = corr[valid]
    lags = lags[valid]

    norm = np.linalg.norm(x) * np.linalg.norm(y)

    if norm < 1e-12:
        raise ValueError("Audio segment is silent")

    corr = corr / norm

    peak = np.argmax(np.abs(corr))

    return int(lags[peak]), float(corr[peak])


def main():

    configurations = [
        (0, 0),
        (0, 5),
        (3, 3),
        (5, 0),
        (5, 5),
    ]

    for gain, tone in configurations:

        filename = (
            f"a_bluesdriver_g{gain}_t{tone}.wav"
        )

        output_file = OUTPUT_DIR / filename

        print(f"\n{'=' * 55}")
        print(filename)
        print(f"{'=' * 55}")

        lag_values = []

        for start in START_TIMES:

            x, sr = load_segment(
                INPUT_FILE,
                start,
                WINDOW_SECONDS
            )

            y, _ = load_segment(
                output_file,
                start,
                WINDOW_SECONDS
            )

            lag, score = estimate_lag(x, y, sr)

            lag_ms = 1000 * lag / sr

            lag_values.append(lag)

            print(
                f"Time: {start:3d}s | "
                f"Lag: {lag:5d} samples | "
                f"{lag_ms:8.3f} ms | "
                f"Correlation: {score:.4f}"
            )

        print(
            f"\nMedian lag: "
            f"{np.median(lag_values):.1f} samples"
        )

        print(
            f"Lag standard deviation: "
            f"{np.std(lag_values):.2f} samples"
        )


if __name__ == "__main__":
    main()
