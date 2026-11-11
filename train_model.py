"""Train and evaluate a multi-output Random Forest on labeled_training_data.csv."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = (["centroid_mean", "centroid_std", "rolloff_mean", "rms_mean", "rms_std", "flatness_mean"]
                   + [f"mfcc_{i}_mean" for i in range(1, 14)]
                   + [f"mfcc_{i}_std" for i in range(1, 14)])
TARGET_COLUMNS = ["bass_gain_db", "mid_gain_db", "treble_gain_db", "drive", "compression_threshold_db", "compression_ratio"]


def main():
    df = pd.read_csv(ROOT / "labeled_training_data.csv")
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    Y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    groups = df["source_file"].to_numpy()
    if len(np.unique(groups)) < 3:
        raise ValueError("Need at least 3 distinct original recordings for a meaningful group split")
    
    # splits into train and test sets while ensuring that all examples from the same source recording are in the same set. 
    # This prevents data leakage and ensures that the model is evaluated on unseen recordings.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, Y, groups=groups))

    # 250 decision trees, minimum of 2 samples per leaf, use all available CPU cores, and set a random seed for reproducibility.
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=42)
    model.fit(X[train_idx], Y[train_idx])
    pred = model.predict(X[test_idx])


    print(f"Train examples: {len(train_idx)} | Test examples: {len(test_idx)}")
    print(f"Train source recordings: {len(set(groups[train_idx]))} | Test source recordings: {len(set(groups[test_idx]))}")
    print("\nParameter                       MAE        R²")
    print("------------------------------------------------")


    for i, name in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(Y[test_idx, i], pred[:, i])
        r2 = r2_score(Y[test_idx, i], pred[:, i])
        print(f"{name:<30} {mae:>7.3f}   {r2:>7.3f}")
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS, "sample_rate": 22050}, ROOT / "guitar_tone_model.joblib")
    
    
    print("\nSaved guitar_tone_model.joblib")
    print("Note: parameter recovery from global spectral features may be ambiguous; negative R² is possible.")


if __name__ == "__main__":
    main()
