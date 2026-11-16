
"""Train and evaluate the 94-feature Random Forest."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)

from expanded_features import FEATURE_COLUMNS, SR
from generate_training_data import TARGET_COLUMNS

ROOT = Path(__file__).resolve().parent


def main():

    # Load expanded dataset
    df = pd.read_csv(
        ROOT / "labeled_training_data_v2.csv"
    )

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=float
    )

    Y = df[TARGET_COLUMNS].to_numpy(
        dtype=float
    )

    groups = df["source_file"].to_numpy()

    # Check for invalid values
    if (
        not np.all(np.isfinite(X))
        or not np.all(np.isfinite(Y))
    ):
        raise ValueError(
            "Dataset contains non-finite values"
        )

    # Split by original recording
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, Y, groups=groups)
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # Train Model 2
    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, Y_train)

    # Evaluate
    predictions = model.predict(X_test)

    print(
        f"Train examples: {len(train_idx)} | "
        f"Test examples: {len(test_idx)}"
    )

    print(
        f"Train source recordings: "
        f"{len(set(groups[train_idx]))} | "
        f"Test source recordings: "
        f"{len(set(groups[test_idx]))}"
    )

    print(f"Feature count: {len(FEATURE_COLUMNS)}")

    print(
        f'{"Parameter":32s} '
        f'{"MAE":>9s} {"R²":>9s}'
    )

    print("-" * 52)

    for i, name in enumerate(TARGET_COLUMNS):

        mae = mean_absolute_error(
            Y_test[:, i],
            predictions[:, i]
        )

        r2 = r2_score(
            Y_test[:, i],
            predictions[:, i]
        )

        print(
            f"{name:32s} {mae:9.3f} {r2:9.3f}"
        )

    # Save Model 2
    output_path = ROOT / "guitar_tone_model_v2.joblib"

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
            "sample_rate": SR,
            "version": 2,
        },
        output_path
    )

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
