"""
extract_features.py - tone-feature extraction for the Song Injected ML Pedal.

Three ways to use it:

  1. Library (import from the rest of the project, pipeline.py or the RF code)
         from extract_features import stem_to_features, feature_vector_from_array
         vec = stem_to_features("pipeline_output/stems/<song>/guitar.wav")
         vec = feature_vector_from_array(samples, sr=48000)   # e.g. Seed3 capture

  2. Command line (batch over files / folders, writes .npz + features.csv + plots)
         python extract_features.py pipeline_output/stems --out features --plot
         python extract_features.py takes/*.wav --report            # level / DC / noise check

  3. Conditioning check for the Seed3 / guitar captures (--report), so the input
     chain can be verified before any features are computed.

Install:  pip install numpy scipy librosa matplotlib soundfile
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import lfilter

# --------------------------------------------------------------------------
# Shared settings (agree on these as a team)
# --------------------------------------------------------------------------
SR = int(os.environ.get("TONE_SR", 44100))   # override with env var TONE_SR=48000
N_FFT = 2048
HOP = 512
N_MFCC = 13
SILENCE_DB = -50.0      # frames this far below the loudest frame are ignored
TARGET_PEAK = 0.95      # peak normalization target
TARGET_RMS = 0.1        # RMS normalization target (about -20 dBFS)
DC_R = 0.999            # DC-blocker pole; cutoff ~ (1-R)*fs/(2*pi) ~ 7-8 Hz
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}

# --------------------------------------------------------------------------
# Loading and conditioning
# --------------------------------------------------------------------------
def load_audio(path, sr: int = SR) -> np.ndarray:
    """Load any audio file as mono float64 at `sr` (stereo is averaged to mono)."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float64)

def dc_block(y: np.ndarray, r: float = DC_R) -> np.ndarray:
    """One-pole DC blocker: y[n] = x[n] - x[n-1] + r*y[n-1].
    Same difference equation can run sample-by-sample on the Seed3, so offline
    and on-pedal conditioning match."""
    return lfilter([1.0, -1.0], [1.0, -r], y)

def condition(y: np.ndarray, sr: int = SR, mode: str = "file",
              normalize: str = "peak", dc_r: float = DC_R) -> np.ndarray:
    """Input conditioning.

    mode:
      "file"   - subtract the mean (offline, whole-file DC removal)
      "stream" - causal DC blocker (what the pedal can actually do in real time);
                 the first 0.2 s is dropped to skip the filter's start-up transient
    normalize:
      "peak"   - scale so the peak is TARGET_PEAK (default, matches earlier version)
      "rms"    - scale to TARGET_RMS, limited so the peak never exceeds 0.99
      "none"   - leave level alone (use this to inspect raw Seed3 levels)
    """
    y = np.asarray(y, dtype=np.float64)
    if mode == "file":
        y = y - np.mean(y)
    elif mode == "stream":
        y = dc_block(y, dc_r)[int(0.2 * sr):]
    else:
        raise ValueError(f"unknown conditioning mode: {mode!r}")

    if len(y) == 0:
        return y
    peak = np.max(np.abs(y))
    if normalize == "peak":
        if peak > 0:
            y = TARGET_PEAK * y / peak
    elif normalize == "rms":
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 0:
            gain = TARGET_RMS / rms
            if peak * gain > 0.99:
                gain = 0.99 / peak
            y = y * gain
    elif normalize != "none":
        raise ValueError(f"unknown normalize option: {normalize!r}")
    return y

def conditioning_report(y_raw: np.ndarray, sr: int = SR) -> dict:
    """Level / DC / clipping / noise-floor check on RAW (unconditioned) samples.
    Use on Seed3 and audio-interface captures before trusting any features."""
    y = np.asarray(y_raw, dtype=np.float64)
    eps = 1e-12
    rms_frames = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms_frames + eps)
    return {
        "duration_s": len(y) / sr,
        "peak_dbfs": 20 * np.log10(np.max(np.abs(y)) + eps),
        "rms_dbfs": 20 * np.log10(np.sqrt(np.mean(y ** 2)) + eps),
        "dc_offset": float(np.mean(y)),
        "clipped_samples": int(np.sum(np.abs(y) >= 0.999)),
        # quietest 10% of frames approximates the noise floor between notes
        "noise_floor_dbfs": float(np.percentile(rms_db, 10)),
        "dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)),
    }

def print_report(name: str, rep: dict) -> None:
    print(f"\n[{name}] conditioning report")
    print(f"  duration         : {rep['duration_s']:.1f} s")
    print(f"  peak             : {rep['peak_dbfs']:.1f} dBFS"
          f"   (aim for about -12 to -3 on the hardest strum)")
    print(f"  RMS              : {rep['rms_dbfs']:.1f} dBFS")
    print(f"  DC offset        : {rep['dc_offset']:+.5f}")
    print(f"  clipped samples  : {rep['clipped_samples']}"
          f"   (should be 0)")
    print(f"  noise floor      : {rep['noise_floor_dbfs']:.1f} dBFS (quietest 10% of frames)")
    print(f"  dynamic range    : {rep['dynamic_range_db']:.1f} dB")

# --------------------------------------------------------------------------
# STFT
# --------------------------------------------------------------------------
def compute_stft(y, n_fft=N_FFT, hop=HOP):
    """Return complex STFT, magnitude, phase. Shape: (1 + n_fft/2, frames)."""
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    return stft, np.abs(stft), np.angle(stft)

def active_frame_mask(rms: np.ndarray, silence_db: float = SILENCE_DB) -> np.ndarray:
    """True for frames loud enough to count (skips gaps between notes)."""
    rms_db = 20 * np.log10(rms + 1e-10)
    return rms_db > (rms_db.max() + silence_db)

# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def feature_names(n_mfcc: int = N_MFCC) -> list[str]:
    return ([f"mfcc{i}_mean" for i in range(n_mfcc)]
            + [f"mfcc{i}_std" for i in range(n_mfcc)]
            + ["centroid_mean", "bandwidth_mean", "rolloff_mean", "flatness_mean",
               "rms_mean", "rms_std"])

FEATURE_NAMES = feature_names()

def build_feature_vector(f: dict) -> np.ndarray:
    """THE one place that defines the vector the model sees. Edit here to match
    the dataset's layout if it differs."""
    return np.concatenate([
        f["mfcc_mean"], f["mfcc_std"],
        [f["centroid_mean"], f["bandwidth_mean"], f["rolloff_mean"],
         f["flatness_mean"], f["rms_mean"], f["rms_std"]],
    ])

def extract_from_array(y_raw: np.ndarray, sr: int = SR, mode: str = "file",
                       normalize: str = "peak", n_mfcc: int = N_MFCC,
                       n_fft: int = N_FFT, hop: int = HOP) -> dict:
    """Raw mono samples in -> dict of features out (includes 'vector')."""
    y = condition(y_raw, sr=sr, mode=mode, normalize=normalize)
    if len(y) < n_fft:
        raise ValueError("audio too short for analysis "
                         f"({len(y)} samples after conditioning, need >= {n_fft})")

    _, mag, phase = compute_stft(y, n_fft, hop)
    rms_all = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    n_frames = min(mag.shape[1], rms_all.shape[0])
    mag, phase, rms_all = mag[:, :n_frames], phase[:, :n_frames], rms_all[:n_frames]

    mask = active_frame_mask(rms_all)
    if mask.sum() == 0:
        mask[:] = True

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S = mag[:, mask]

    # Average spectral envelope (dB): mean power over active frames
    envelope_db = 10 * np.log10((S ** 2).mean(axis=1) + 1e-12)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=n_fft, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    rms = rms_all[mask]

    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_fft=n_fft)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)

    f = {
        "freqs": freqs,
        "envelope_db": envelope_db,
        "centroid": centroid, "bandwidth": bandwidth,
        "rolloff": rolloff, "flatness": flatness, "rms": rms,
        "mfcc": mfcc,
        "mfcc_mean": mfcc.mean(axis=1), "mfcc_std": mfcc.std(axis=1),
        "centroid_mean": float(centroid.mean()),
        "bandwidth_mean": float(bandwidth.mean()),
        "rolloff_mean": float(rolloff.mean()),
        "flatness_mean": float(flatness.mean()),
        "rms_mean": float(rms.mean()), "rms_std": float(rms.std()),
        "n_active_frames": int(mask.sum()),
        "mag": mag, "phase": phase,   # kept for the effects / plotting code
        "sr": sr, "mode": mode, "normalize": normalize,
    }
    f["vector"] = build_feature_vector(f)
    f["names"] = feature_names(n_mfcc)
    return f

def feature_vector_from_array(y_raw: np.ndarray, sr: int = SR, **kw) -> np.ndarray:
    """Convenience: samples -> fixed-length vector (e.g. for a Seed3 capture)."""
    return extract_from_array(y_raw, sr=sr, **kw)["vector"]

def stem_to_features(path, sr: int = SR, **kw) -> np.ndarray:
    """Convenience for pipeline.py: guitar stem path -> fixed-length vector."""
    return feature_vector_from_array(load_audio(path, sr), sr=sr, **kw)

# --------------------------------------------------------------------------
# Plotting (matplotlib)
# --------------------------------------------------------------------------
def plot_song(name, f, out_dir: Path, hop: int = HOP):
    import matplotlib.pyplot as plt
    import librosa.display  # noqa: F401

    sr = f["sr"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Tone features: {name}")

    mag_db = librosa.amplitude_to_db(f["mag"], ref=np.max)
    img = librosa.display.specshow(mag_db, sr=sr, hop_length=hop, x_axis="time",
                                   y_axis="log", ax=ax[0, 0])
    ax[0, 0].set_title("STFT magnitude (dB)")
    fig.colorbar(img, ax=ax[0, 0], format="%+2.0f dB")

    ax[0, 1].semilogx(f["freqs"][1:], f["envelope_db"][1:])
    ax[0, 1].set(title="Average spectral envelope", xlabel="Frequency (Hz)",
                 ylabel="Power (dB)", xlim=(30, sr / 2))
    ax[0, 1].grid(True, which="both", alpha=0.3)

    t = np.arange(len(f["centroid"]))
    ax[1, 0].plot(t, f["centroid"], label=f"centroid (mean {f['centroid_mean']:.0f} Hz)")
    ax[1, 0].plot(t, f["rolloff"], alpha=0.7,
                  label=f"85% rolloff (mean {f['rolloff_mean']:.0f} Hz)")
    ax[1, 0].set(title="Spectral centroid & rolloff (active frames)",
                 xlabel="Active frame", ylabel="Hz")
    ax[1, 0].legend()

    img2 = librosa.display.specshow(f["mfcc"], x_axis="frames", ax=ax[1, 1])
    ax[1, 1].set(title="MFCCs", ylabel="Coefficient")
    fig.colorbar(img2, ax=ax[1, 1])

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_features.png", dpi=130)
    plt.close(fig)

def plot_comparison(all_feats: dict, out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, f in all_feats.items():
        ax[0].semilogx(f["freqs"][1:], f["envelope_db"][1:], label=name)
        ax[1].plot(f["mfcc_mean"], marker="o", label=name)
    ax[0].set(title="Average spectral envelope", xlabel="Frequency (Hz)",
              ylabel="Power (dB)", xlim=(30, SR / 2))
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7)
    ax[1].set(title="Mean MFCCs", xlabel="Coefficient", ylabel="Value")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=130)
    plt.close(fig)

# --------------------------------------------------------------------------
# Batch / CLI
# --------------------------------------------------------------------------
def collect_files(inputs) -> list[Path]:
    """Expand files and folders (recursively) into a list of audio files."""
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTS)
        elif p.exists():
            files.append(p)
        else:
            print(f"[warn] not found: {item}")
    return files

def unique_name(path: Path, used: set) -> str:
    """Name for outputs. Stems from the pipeline are all called 'guitar.wav', so
    prefix with the parent folder (the song name) when needed."""
    name = path.stem
    if name in used or name.lower() in {"guitar", "stem", "other"}:
        name = f"{path.parent.name}_{path.stem}"
    used.add(name)
    return name

def main():
    ap = argparse.ArgumentParser(
        description="Extract tone features (and conditioning reports) from audio files.")
    ap.add_argument("inputs", nargs="+", help="audio files and/or folders")
    ap.add_argument("--out", default="features", help="output directory")
    ap.add_argument("--sr", type=int, default=SR, help=f"analysis sample rate (default {SR})")
    ap.add_argument("--n-mfcc", type=int, default=N_MFCC)
    ap.add_argument("--condition", choices=["file", "stream"], default="file",
                    help="file = whole-file DC removal; stream = causal DC blocker "
                         "(matches what the Seed3 can do in real time)")
    ap.add_argument("--normalize", choices=["peak", "rms", "none"], default="peak")
    ap.add_argument("--report", action="store_true",
                    help="print level / DC / clipping / noise-floor report on the raw audio")
    ap.add_argument("--report-only", action="store_true",
                    help="only print the conditioning report; skip feature extraction")
    ap.add_argument("--plot", action="store_true", help="write per-file and comparison plots")
    ap.add_argument("--expect-len", type=int, default=None,
                    help="fail if the feature vector length differs (e.g. match the RF dataset)")
    args = ap.parse_args()

    files = collect_files(args.inputs)
    if not files:
        raise SystemExit("No audio files found.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    used: set = set()
    rows, all_feats = [], {}
    for path in files:
        name = unique_name(path, used)
        print(f"Processing {name} ...")
        y_raw = load_audio(path, args.sr)

        if args.report or args.report_only:
            print_report(name, conditioning_report(y_raw, args.sr))
        if args.report_only:
            continue

        f = extract_from_array(y_raw, sr=args.sr, mode=args.condition,
                               normalize=args.normalize, n_mfcc=args.n_mfcc)
        vec = f["vector"]
        if args.expect_len is not None and len(vec) != args.expect_len:
            raise SystemExit(f"Feature vector has {len(vec)} values but --expect-len is "
                             f"{args.expect_len}. Edit build_feature_vector().")

        np.savez(out_dir / f"{name}_features.npz",
                 feature_vector=vec, feature_names=np.array(f["names"]),
                 freqs=f["freqs"], envelope_db=f["envelope_db"],
                 sr=args.sr, condition=args.condition, normalize=args.normalize)
        rows.append([name] + [f"{v:.8g}" for v in vec])
        all_feats[name] = f
        print(f"  {len(vec)} features, {f['n_active_frames']} active frames, "
              f"centroid {f['centroid_mean']:.0f} Hz, rolloff {f['rolloff_mean']:.0f} Hz")

        if args.plot:
            plot_song(name, f, out_dir)

    if rows:
        with open(out_dir / "features.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name"] + feature_names(args.n_mfcc))
            w.writerows(rows)
        if args.plot and len(all_feats) > 1:
            plot_comparison(all_feats, out_dir)
        print(f"Done. Outputs in {out_dir.resolve()}")

if __name__ == "__main__":
    main()