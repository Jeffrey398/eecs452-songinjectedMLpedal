"""Generate labeled guitar-tone examples from clean GuitarSet recordings.

Install: python3 -m pip install numpy scipy librosa soundfile pandas scikit-learn joblib
Run:     python3 generate_training_data.py --examples-per-file 50
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, sosfilt, sosfiltfilt
import soundfile as sf

ROOT = Path(__file__).resolve().parent
# 32 features: 6 global spectral + 13 MFCC means + 13 MFCC stddevs
FEATURE_COLUMNS = (["centroid_mean", "centroid_std", "rolloff_mean", "rms_mean", "rms_std", "flatness_mean"]
                   + [f"mfcc_{i}_mean" for i in range(1, 14)]
                   + [f"mfcc_{i}_std" for i in range(1, 14)])
# controls the effects chain: 3-band EQ, distortion, and compression
TARGET_COLUMNS = ["bass_gain_db", "mid_gain_db", "treble_gain_db", "drive", "compression_threshold_db", "compression_ratio"]
SR = 22050 # 22050 samples per second is a common sample rate for audio processing

# program works on a numpy array of audio samples, and returns a dictionary of features extracted from the audio. 
# The features include spectral centroid, spectral rolloff, root mean square (RMS) energy, spectral flatness, and Mel-frequency cepstral coefficients (MFCCs). 
# The function computes the mean and standard deviation of these features over time and returns them as a dictionary with keys corresponding to the feature names.


# max represenatable frequency is 11025Hz, so we can use 250Hz for bass, 4000Hz for mid, and 4000Hz+ for treble.
# TODO : consider what the pedal may operate at a higher sample right

# convert audio features to ML inputs
def extract_features(y, sr=SR):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    vals = [np.mean(centroid), np.std(centroid), np.mean(rolloff), np.mean(rms), np.std(rms), np.mean(flatness)]
    vals.extend(np.mean(mfcc, axis=1))
    vals.extend(np.std(mfcc, axis=1))
    return dict(zip(FEATURE_COLUMNS, map(float, vals)))

# butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
# Second-order sections (SOS), also known as biquads, break down complex, high-order digital filters into a series of smaller, second-order filter blocks. stable for IIR filters
def filter_band(y, sr, lo=None, hi=None):
    if lo is None:
        sos = butter(2, hi, btype="lowpass", fs=sr, output="sos")
    elif hi is None:
        sos = butter(2, lo, btype="highpass", fs=sr, output="sos")
    else:
        sos = butter(2, (lo, hi), btype="bandpass", fs=sr, output="sos")
    # zero-phase filtering for offline dataset generation; same algorithm must be matched at deployment.
    # not directly usable for real-time processing because it needs future samples to filter the current sample. for real-time processing, use sosfilt() instead of sosfiltfilt().
    return sosfiltfilt(sos, y)


def apply_eq(y, sr, bass_db, mid_db, treble_db):
    """Simple overlapping three-band *approximate* EQ, not a studio parametric EQ."""
    low = filter_band(y, sr, hi=250)
    mid = filter_band(y, sr, lo=250, hi=4000)
    high = filter_band(y, sr, lo=4000)

    # multiply each band by its gain
    return ((10 ** (bass_db / 20)) * low
            + (10 ** (mid_db / 20)) * mid
            + (10 ** (treble_db / 20)) * high)


# nonlinear distortion algorithms are often implemented using a tanh function, which compresses the dynamic range of the signal and adds harmonic content. 
# The drive parameter controls the amount of distortion applied to the signal, with higher values resulting in more distortion. 
# The output is normalized to prevent clipping and maintain a consistent output level.

def apply_distortion(y, drive):
    """Soft clipping: drive in [0, 1]; drive=0 is an exact bypass."""
    if drive <= 0:
        return y #bypasses distortion 
    gain = 1 + 19 * drive # increase signal level by up to 20x (26dB) before soft clipping
    return np.tanh(gain * y) / np.tanh(gain)

# A compressor reduces the dynamic range of an audio signal by reducing gain when the detected signal level exceeds a threshold. 
# A smooth peak detector is used to avoid abrupt changes in gain, which can cause audible artifacts.
def apply_compression(y, sr, threshold_db, ratio, attack_ms=10, release_ms=100):
    """Basic feed-forward compressor with smoothed peak detector."""
    attack = np.exp(-1 / (sr * attack_ms / 1000))
    release = np.exp(-1 / (sr * release_ms / 1000))
    env = 0.0
    out = np.empty_like(y)
    threshold = 10 ** (threshold_db / 20)
    for i, sample in enumerate(y):
        level = abs(float(sample))
        coeff = attack if level > env else release
        env = coeff * env + (1 - coeff) * level
        if env > threshold:
            gain = (env / threshold) ** (1 / ratio - 1)
        else:
            gain = 1.0
        out[i] = sample * gain
    return out


def main():

    # These let you change the number of generated examples, excerpt duration, and random seed without editing the script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "trainingData" / "genre")
    parser.add_argument("--examples-per-file", type=int, default=50)
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per example")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-audio", action="store_true", help="Save processed examples as WAV files")
    args = parser.parse_args()
    if args.examples_per_file < 1 or args.duration <= 0:
        parser.error("--examples-per-file and --duration must be positive")
    files = sorted(p for p in args.input_dir.rglob("*") if p.suffix.lower() in {".wav", ".aif", ".aiff", ".mp3"}) if args.input_dir.exists() else []
    if not files:
        parser.error(f"No audio files found in {args.input_dir}")
    rng = np.random.default_rng(args.seed)
    rows = []
    audio_dir = ROOT / "generated_audio"
    if args.save_audio:
        audio_dir.mkdir(exist_ok=True)
    for file_idx, path in enumerate(files):
        try:
            y, sr = librosa.load(path, sr=SR, mono=True)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        clip_len = int(sr * args.duration)
        if len(y) < clip_len:
            print(f"Skipping short file: {path.name}")
            continue
        for version in range(args.examples_per_file):
            start = int(rng.integers(0, len(y) - clip_len + 1))
            clip = y[start:start + clip_len].copy()
            # Avoid almost silent excerpts; a quiet clip is still usable but warn.
            if np.sqrt(np.mean(clip ** 2)) < 1e-4:
                print(f"Warning: quiet excerpt in {path.name}, version {version}")

            # 6 random DSP parameters: 3-band EQ, distortion drive, compression threshold and ratio. 
            # The ranges are chosen to cover typical guitar pedal settings.
            bass, mid, treble = rng.uniform(-9, 9, size=3)
            drive = float(rng.uniform(0, 1))
            threshold = float(rng.uniform(-35, -10))
            ratio = float(rng.uniform(1, 6))
            processed = apply_eq(clip, sr, bass, mid, treble)
            processed = apply_distortion(processed, drive)
            processed = apply_compression(processed, sr, threshold, ratio)
            # Keep amplitude intact except for clipping to valid digital range.
            processed = np.clip(processed, -1.0, 1.0)
            row = {"source_file": path.name, "example_id": version}
            row.update(extract_features(processed, sr))
            row.update(dict(zip(TARGET_COLUMNS, [bass, mid, treble, drive, threshold, ratio])))
            rows.append(row)
            if args.save_audio:
                sf.write(audio_dir / f"{file_idx:03d}_{version:03d}.wav", processed, sr)
        print(f"Processed {path.name}: {args.examples_per_file} examples")
    if not rows:
        raise RuntimeError("No examples generated; check input audio lengths")
    out = ROOT / "labeled_training_data.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} labeled examples to {out}")


if __name__ == "__main__":
    main()
