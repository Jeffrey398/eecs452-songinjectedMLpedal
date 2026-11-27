
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import least_squares

from .wh_model import (
    SR,
    PARAMETER_NAMES,
    INITIAL_PARAMS,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    process,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "trainingData/input/a_0-input.wav"
OUTPUT_DIR = ROOT / "trainingData/bluesdriver"

# Separate musical passages for training and testing
TRAIN_TIMES = [10, 30, 70, 120, 200]
TEST_TIMES = [45, 90, 150, 240]

SEGMENT_SECONDS = 1.0
WARMUP_SECONDS = 0.1

WARMUP_SAMPLES = int(WARMUP_SECONDS * SR)


def load_segment(path, start):

    with sf.SoundFile(str(path)) as f:

        if f.samplerate != SR:
            raise ValueError("Expected 48 kHz audio")

        if f.channels != 1:
            raise ValueError("Expected mono audio")

        offset = int(
            (start - WARMUP_SECONDS) * SR
        )

        count = int(
            (SEGMENT_SECONDS + WARMUP_SECONDS) * SR
        )

        f.seek(offset)

        audio = f.read(
            count,
            dtype="float64"
        )

        if len(audio) != count:
            raise ValueError("Incomplete segment")

    return audio


def load_pairs(output_file, times):

    pairs = []

    for start in times:

        x = load_segment(INPUT_FILE, start)
        y = load_segment(output_file, start)

        pairs.append((x, y))

    return pairs


def residuals(params, pairs):

    errors = []

    for x, target in pairs:

        predicted = process(x, params, SR)

        # Ignore initial filter transients
        predicted = predicted[WARMUP_SAMPLES:]
        target = target[WARMUP_SAMPLES:]

        scale = np.sqrt(
            np.mean(target ** 2)
        ) + 1e-8

        error = (predicted - target) / scale

        errors.append(error)

    return np.concatenate(errors)


def evaluate(params, pairs):

    squared_error = 0.0
    signal_energy = 0.0
    samples = 0

    for x, target in pairs:

        predicted = process(x, params, SR)

        predicted = predicted[WARMUP_SAMPLES:]
        target = target[WARMUP_SAMPLES:]

        error = predicted - target

        squared_error += np.sum(error ** 2)
        signal_energy += np.sum(target ** 2)
        samples += len(target)

    rmse = np.sqrt(squared_error / samples)

    esr = squared_error / (
        signal_energy + 1e-12
    )

    return rmse, esr


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gain",
        type=int,
        choices=range(6),
        default=3
    )

    parser.add_argument(
        "--tone",
        type=int,
        choices=range(6),
        default=3
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=100
    )

    args = parser.parse_args()

    output_file = OUTPUT_DIR / (
        f"a_bluesdriver_g{args.gain}"
        f"_t{args.tone}.wav"
    )

    train_pairs = load_pairs(
        output_file,
        TRAIN_TIMES
    )

    test_pairs = load_pairs(
        output_file,
        TEST_TIMES
    )

    print("Training Wiener-Hammerstein model")
    print(f"Target: {output_file.name}")
    print(f"Sample rate: {SR}")
    print(f"Drive: {args.gain / 5:.2f}")
    print(f"Tone: {args.tone / 5:.2f}")

    print("\nInitial performance:")

    rmse, esr = evaluate(
        INITIAL_PARAMS,
        test_pairs
    )

    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test ESR: {esr:.6f}")

    result = least_squares(
        residuals,
        INITIAL_PARAMS,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        args=(train_pairs,),
        x_scale="jac",
        max_nfev=args.max_nfev,
        verbose=2,
    )

    params = result.x

    print("\nFitted parameters:")

    for name, value in zip(
        PARAMETER_NAMES, params
    ):
        print(f"{name}: {value:.6f}")

    train_rmse, train_esr = evaluate(
        params, train_pairs
    )

    test_rmse, test_esr = evaluate(
        params, test_pairs
    )

    print("\nFinal performance:")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Train ESR: {train_esr:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test ESR: {test_esr:.6f}")

    print(f"\nOptimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")

    data = {
        "pedal": "bluesdriver",
        "sample_rate": SR,
        "drive": args.gain / 5,
        "tone": args.tone / 5,
        "parameters": dict(
            zip(PARAMETER_NAMES, params.tolist())
        ),
        "train_rmse": float(train_rmse),
        "train_esr": float(train_esr),
        "test_rmse": float(test_rmse),
        "test_esr": float(test_esr),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }

    save_path = Path(__file__).parent / (
        f"bluesdriver_wh_g{args.gain}"
        f"_t{args.tone}.json"
    )

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
