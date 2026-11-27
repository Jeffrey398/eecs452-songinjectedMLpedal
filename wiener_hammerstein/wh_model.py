# The WH model learns by comparing its processed output against the actual pedal's output
import numpy as np
from scipy.signal import lfilter

SR = 48000

PARAMETER_NAMES = [
    "pre_hp",
    "pre_lp",
    "gain",
    "bias",
    "post_lp",
    "output_gain",
]

# TODO: adjust for better performance?
INITIAL_PARAMS = np.array([
    80.0,       # Pre-filter HP cutoff
    8000.0,     # Pre-filter LP cutoff
    5.0,        # Nonlinear gain
    0.0,        # Clipping bias
    4000.0,     # Post-filter LP cutoff
    1.0,        # Output gain
])

LOWER_BOUNDS = np.array([
    10.0,
    500.0,
    0.1,
    -2.0,
    200.0,
    0.001,
])

UPPER_BOUNDS = np.array([
    500.0,
    20000.0,
    1000.0,
    2.0,
    20000.0,
    20.0,
])


def lowpass(x, cutoff, sr=SR):
    alpha = np.exp(-2.0 * np.pi * cutoff / sr)

    b = [1.0 - alpha]
    a = [1.0, -alpha]

    return lfilter(b, a, x)


def highpass(x, cutoff, sr=SR):
    return x - lowpass(x, cutoff, sr)


def waveshaper(x, gain, bias):
    return (
        np.tanh(gain * x + bias)
        - np.tanh(bias)
    )


def process(x, params, sr=SR):

    (
        pre_hp,
        pre_lp,
        gain,
        bias,
        post_lp,
        output_gain,
    ) = params

    x = np.asarray(x, dtype=np.float64)

    # Stage 1: Wiener pre-filter
    u = highpass(x, pre_hp, sr)
    u = lowpass(u, pre_lp, sr)

    # Stage 2: Static nonlinear distortion
    v = waveshaper(u, gain, bias)

    # Stage 3: Hammerstein post-filter
    y = lowpass(v, post_lp, sr)

    # Match measured output amplitude
    y = output_gain * y

    return y
