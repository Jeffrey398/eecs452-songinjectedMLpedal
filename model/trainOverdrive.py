"""Train Drive/Tone/Level regression using actual recorded pedal examples."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from ML_Pedal.model.expanded_features import FEATURE_COLUMNS, SR

TARGET_COLUMNS = ["drive", "tone"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("overdrive_training_data.csv"))
    p.add_argument("--output", type=Path, default=Path("overdrive_model.joblib"))
    args = p.parse_args()
    df = pd.read_csv(args.dataset)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    Y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    groups = (df["start_seconds"] // 30).astype(int).to_numpy() # prevents same musical passage across different pedal configurations form being in both training and testing
    if len(set(groups)) < 5:
        raise ValueError("At least five distinct recordings are recommended for grouped evaluation")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Non-finite feature or label detected")
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train, test = next(split.split(X, Y, groups))
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2,random_state=42, n_jobs=-1)
    model.fit(X[train], Y[train])
    predicted = model.predict(X[test])
    print(f"Training recordings: {len(set(groups[train]))}, test recordings: {len(set(groups[test]))}")
    print(f"Training windows: {len(train)}, test windows: {len(test)}")
    for i, name in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(Y[test, i], predicted[:, i])
        r2 = r2_score(Y[test, i], predicted[:, i])
        print(f"{name}: MAE={mae:.3f} ({mae * 100:.1f} knob percentage points), R²={r2:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS,
                 "targets": TARGET_COLUMNS, "sample_rate": SR,
                 "window_seconds": 5.0,
                 "pedals": sorted(df["pedal"].unique().tolist()),
                 "version": "overdrive-recordings-v1"}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
