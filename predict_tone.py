
import argparse
import joblib
import librosa
import numpy as np

from expanded_features import extract_features

parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
parser.add_argument("--port", default=None)
args = parser.parse_args()

# Load trained Model 2
bundle = joblib.load("guitar_tone_model_v2.joblib")

model = bundle["model"]
feature_names = bundle["features"]
target_names = bundle["targets"]
sr = bundle["sample_rate"]

# Analyze a five-second excerpt
y, _ = librosa.load(
    args.audio_file,
    sr=sr,
    mono=True
)

n = int(5 * sr)
if len(y) < n:
    raise ValueError("Recording must be at least 5 seconds")

# Select the loudest non-overlapping 5-second excerpt
starts = list(range(0, len(y) - n + 1, n))
start = max(
    starts,
    key=lambda s: np.mean(y[s:s+n] ** 2)
)
clip = y[start:start+n]

# Extract features in the saved training order
features = extract_features(clip, sr)
X = np.array(
    [[features[name] for name in feature_names]],
    dtype=float
)

# Predict all six DSP parameters
prediction = model.predict(X)[0]
settings = dict(zip(target_names, prediction))

for name, value in settings.items():
    print(f"{name}: {value:.3f}")

# Serial format:
# bass,mid,treble,drive,threshold,ratio
message = ",".join(
    f"{settings[name]:.4f}"
    for name in target_names
) + "\n"

print("Serial message:", message.strip())

if args.port:
    import serial
    with serial.Serial(args.port, 115200, timeout=2) as ser:
        ser.write(message.encode("ascii"))
        ser.flush()
        print("Settings sent")
