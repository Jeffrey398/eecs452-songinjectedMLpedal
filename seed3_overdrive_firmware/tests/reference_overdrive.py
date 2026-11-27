"""
reference_overdrive.py - Python reference of the Seed3 Wiener-Hammerstein overdrive.

Must stay identical to overdrive_dsp.h (verify_overdrive_dsp.py checks that). The knob ->
model mapping (map_knobs) is a PLACEHOLDER; when the real pedal model is fitted, update
this function and MapKnobs() in overdrive_dsp.h together.
"""
import numpy as np
from scipy.signal import lfilter

BUTTER_Q = 0.7071067811865476
DC_R = 0.998


def map_knobs(drive, tone, level):
    return dict(
        h1_hp_hz=80.0, h1_mid_hz=720.0, h1_mid_db=4.0, h1_mid_q=0.8,
        nl_gain=1.0 + 49.0 * drive * drive,
        nl_bias=0.10 * drive,
        h2_lp_hz=800.0 * 2.0 ** (3.5 * tone),
        out_gain=10.0 ** ((level - 0.5) * 18.0 / 20.0),
    )


def _hp(fs, f0, q):
    w = 2 * np.pi * f0 / fs; cw = np.cos(w); al = np.sin(w) / (2 * q)
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]); a = np.array([1 + al, -2 * cw, 1 - al])
    return b / a[0], a / a[0]


def _lp(fs, f0, q):
    w = 2 * np.pi * f0 / fs; cw = np.cos(w); al = np.sin(w) / (2 * q)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2]); a = np.array([1 + al, -2 * cw, 1 - al])
    return b / a[0], a / a[0]


def _peak(fs, f0, gain_db, q):
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / fs; cw = np.cos(w); al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * cw, 1 - al * A]); a = np.array([1 + al / A, -2 * cw, 1 - al / A])
    return b / a[0], a / a[0]


def overdrive(x, sr, drive, tone, level):
    """Full effect (engaged). x: mono float array, returns same length."""
    p = map_knobs(drive, tone, level)
    u = np.asarray(x, dtype=np.float64)
    u = lfilter(*_hp(sr, p["h1_hp_hz"], BUTTER_Q), u)
    u = lfilter(*_peak(sr, p["h1_mid_hz"], p["h1_mid_db"], p["h1_mid_q"]), u)
    g, b = p["nl_gain"], p["nl_bias"]
    v = (np.tanh(g * u + b) - np.tanh(b)) / np.tanh(g)
    w = lfilter(*_lp(sr, p["h2_lp_hz"], BUTTER_Q), v)
    y = lfilter([1.0, -1.0], [1.0, -DC_R], w) * p["out_gain"]
    return np.clip(y, -1.0, 1.0)


def test_signal(sr=48000, seconds=6.0, seed=0):
    """Plucked-string-like test signal."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds); out = np.zeros(n); t = np.arange(int(sr * 1.2)) / sr; pos = 0
    while pos < n - sr // 2:
        f = rng.choice([82.4, 110.0, 146.8, 196.0, 246.9, 329.6])
        note = sum((1.0 / h) * np.sin(2 * np.pi * f * h * t) * np.exp(-t * (2 + h)) for h in range(1, 12))
        note += 0.05 * rng.standard_normal(len(t)) * np.exp(-t * 30)
        seg = out[pos:pos + len(note)]; seg += rng.uniform(0.3, 0.9) * note[:len(seg)]
        pos += int(sr * rng.uniform(0.4, 0.9))
    return 0.9 * out / np.max(np.abs(out))
