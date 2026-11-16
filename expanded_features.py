
"""94 audio descriptors for Model 2."""

import numpy as np
import librosa
from scipy.stats import skew

SR = 22050
HOP_LENGTH = 512

STATS = ("p10", "p50", "p90", "std")

FEATURE_COLUMNS = (
    [
        "crest_factor",
        "lfo_energy",
        "lfo_peak_freq",
        "delay_peak",
        "flux_mean",
        "flux_max",
        "flux_skew",
        "harm_perc_ratio",
    ]
    + [
        f"{name}_{stat}"
        for name in ("rms", "zcr", "centroid", "rolloff", "flatness")
        for stat in STATS
    ]
    + [
        f"contrast_{i}_{stat}"
        for stat in ("mean", "std")
        for i in range(7)
    ] 
    + [
        f"mfcc_{i}_{stat}"
        for stat in ("mean", "std")
        for i in range(1, 14)
    ]
    + [f"mfcc_{i}_delta_mean" for i in range(1, 14)]
    + [f"mfcc_{i}_delta2_mean" for i in range(1, 14)]
)

assert len(FEATURE_COLUMNS) == 94


def extract_features(y, sr=SR):
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Handle empty, silent, or invalid recordings
    if (
        len(y) == 0
        or not np.all(np.isfinite(y))
        or np.max(np.abs(y)) < 1e-10
    ):
        return dict.fromkeys(FEATURE_COLUMNS, 0.0)

    # 1. Dynamics and compression
    rms = librosa.feature.rms(
        y=y, hop_length=HOP_LENGTH
    )[0]

    crest = float(
        np.max(np.abs(y))
        / (np.sqrt(np.mean(y ** 2)) + 1e-8)
    )

    # 2. Envelope modulation spectrum
    centered = rms - np.mean(rms)

    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(
        len(centered),
        d=HOP_LENGTH / sr
    )

    mask = (frequencies >= 0.5) & (frequencies <= 15.0)

    if np.any(mask):
        selected = spectrum[mask]

        lfo_energy = float(
            np.max(selected) / (np.sum(spectrum) + 1e-8)
        )

        lfo_peak_freq = float(
            frequencies[mask][np.argmax(selected)]
        )
    else:
        lfo_energy = 0.0
        lfo_peak_freq = 0.0

    # 3. Envelope autocorrelation
    if np.sum(centered ** 2) > 1e-12:
        autocorr = librosa.autocorrelate(centered)

        min_lag = max(
            1, int(np.ceil(0.1 * sr / HOP_LENGTH))
        )

        max_lag = min(
            int(np.floor(0.8 * sr / HOP_LENGTH)),
            len(autocorr) - 1
        )

        if max_lag >= min_lag:
            delay_peak = float(
                np.max(
                    autocorr[min_lag:max_lag + 1]
                ) / (autocorr[0] + 1e-8)
            )
        else:
            delay_peak = 0.0
    else:
        delay_peak = 0.0

    # 4. Transients and onset strength
    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    flux_mean = float(np.mean(onset)) if len(onset) else 0.0
    flux_max = float(np.max(onset)) if len(onset) else 0.0

    if len(onset) >= 3 and np.std(onset) > 1e-10:
        flux_skew = float(skew(onset))
    else:
        flux_skew = 0.0

    # 5. Harmonic and percussive energy
    harmonic, percussive = librosa.effects.hpss(y)

    harm_perc_ratio = float(
        np.mean(harmonic ** 2)
        / (np.mean(percussive ** 2) + 1e-8)
    )

    # 6. Spectral characteristics
    zcr = librosa.feature.zero_crossing_rate(
        y=y, hop_length=HOP_LENGTH
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y, hop_length=HOP_LENGTH
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )

    # 7. MFCCs and their derivatives
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        hop_length=HOP_LENGTH
    )

    # Choose a valid odd window for short recordings
    width = min(
        9,
        mfcc.shape[1]
        if mfcc.shape[1] % 2
        else mfcc.shape[1] - 1
    )

    if width >= 3:
        delta = librosa.feature.delta(
            mfcc, width=width
        )

        delta2 = librosa.feature.delta(
            mfcc, order=2, width=width
        )
    else:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)

    # 8. Summarize all features
    vals = [
        crest,
        lfo_energy,
        lfo_peak_freq,
        delay_peak,
        flux_mean,
        flux_max,
        flux_skew,
        harm_perc_ratio,
    ]

    def get_stats(arr):
        return [
            np.percentile(arr, 10),
            np.percentile(arr, 50),
            np.percentile(arr, 90),
            np.std(arr),
        ]

    for feat in (rms, zcr, centroid, rolloff, flatness):
        vals.extend(get_stats(feat))

    vals.extend(np.mean(contrast, axis=1))
    vals.extend(np.std(contrast, axis=1))

    vals.extend(np.mean(mfcc, axis=1))
    vals.extend(np.std(mfcc, axis=1))

    vals.extend(np.mean(delta, axis=1))
    vals.extend(np.mean(delta2, axis=1))

    # Replace NaN and infinity with zero
    vals = np.nan_to_num(
        np.asarray(vals, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    assert len(vals) == len(FEATURE_COLUMNS)

    return dict(zip(FEATURE_COLUMNS, vals.tolist()))
