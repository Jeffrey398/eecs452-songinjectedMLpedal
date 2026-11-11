
import librosa
import numpy as np
from pathlib import Path
import pandas as pd

# Folder containing guitar audio recordings
audio_folder = Path(__file__).resolve().parent / "trainingData/genre"
""


audio_files = list(audio_folder.rglob("*.flac"))

for file in audio_files:
    genre = file.parent.name
    print(f"Genre: {genre}, Song: {file.name}")

def extract_features(audio_path):
    # Load audio at a consistent sample rate
    y, sr = librosa.load(str(audio_path), sr=22050)

    # Spectral centroid: brightness
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Spectral rolloff: high-frequency energy distribution
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # RMS: signal energy
    rms = librosa.feature.rms(y=y)

    # Spectral flatness: noise-like vs tonal characteristics
    flatness = librosa.feature.spectral_flatness(y=y)

    # MFCC: spectral characteristics
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Summarize features over time
    features = [
        np.mean(centroid),
        np.std(centroid),
        np.mean(rolloff),
        np.mean(rms),
        np.std(rms),
        np.mean(flatness),
    ]

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    return np.array(features)


def main():
    X = []
    filenames = []

    if not audio_folder.is_dir():
        raise FileNotFoundError(
            f"Audio folder not found: {audio_folder}"
        )

    # Process every supported audio file
    for file in sorted(audio_folder.iterdir()):
        if file.suffix.lower() not in [".wav", ".aiff", ".aif", ".mp3", ".flac"]:
            continue

        print(f"Processing: {file.name}")

        try:
            features = extract_features(file)
            X.append(features)
            filenames.append(file.name)
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    if not X:
        print("No audio files successfully processed.")
        return

    # Convert list of feature vectors into a matrix
    X = np.vstack(X)

    print("\nFeature matrix shape:", X.shape)
    print("Number of recordings:", X.shape[0])
    print("Features per recording:", X.shape[1])

    # Save features for future model training
    np.save("X_features.npy", X)
    np.save("filenames.npy", np.array(filenames))

    print("\nFeature extraction complete!")


if __name__ == "__main__":
    audio_folder = Path(__file__).resolve().parent / "trainingData/genre"

    all_features = []
    filenames = []

    for file in sorted(audio_folder.rglob("*")):
        if file.suffix.lower() in [".wav", ".aiff", ".aif", ".mp3", ".flac"]:
            print(f"Processing: {file.name}")

            try:
                features = extract_features(file)

                all_features.append(features)
                filenames.append(file.name)

            except Exception as e:
                print(f"Error: {e}")

    if all_features:
        X = np.vstack(all_features)

        feature_names = [
            "centroid_mean",
            "centroid_std",
            "rolloff_mean",
            "rms_mean",
            "rms_std",
            "flatness_mean"
        ]

        feature_names += [
            f"mfcc_{i}_mean" for i in range(1, 14)
        ]

        feature_names += [
            f"mfcc_{i}_std" for i in range(1, 14)
        ]

        df = pd.DataFrame(X, columns=feature_names)
        df.insert(0, "filename", filenames)

        df.to_csv("song_features.csv", index=False)

        print(f"Processed {len(filenames)} songs")
        print("Feature matrix shape:", X.shape)
        print("Saved song_features.csv")
    else:
        print("No songs processed.")
