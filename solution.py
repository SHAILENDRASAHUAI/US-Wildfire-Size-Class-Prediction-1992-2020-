"""
Wildfire Size Class Prediction (1992-2020)
==========================================
Predicts the fire size class (A-G) for US wildfires using LightGBM.

Usage:
    python solution.py

Expects train.csv and test.csv in the current directory.
Outputs submission.csv with predicted probabilities for each class.

Metric: Weighted multi-class log loss (lower is better).
Class weights are inverse-frequency: rarer classes (F, G) penalized more.
"""

import warnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "submission.csv"
RANDOM_STATE = 42
N_FOLDS = 5
TARGET = "fire_size_class"
CLASS_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
SUBMISSION_COLS = ["id"] + [f"class_{c}" for c in CLASS_LABELS]

# LightGBM hyper-parameters tuned for imbalanced multi-class log-loss
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 7,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",   # handles severe class imbalance
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def parse_discovery_time(series: pd.Series) -> pd.Series:
    """Convert 'HHMM' string to hour-of-day float; NaN for missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    # HHMM -> hour + minute/60
    hour = (numeric // 100).clip(0, 23)
    minute = (numeric % 100).clip(0, 59)
    result = hour + minute / 60.0
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features used by the model."""
    df = df.copy()

    # --- Time of discovery ---
    df["discovery_hour"] = parse_discovery_time(df["discovery_time"])
    df["discovery_hour_missing"] = df["discovery_hour"].isna().astype(np.int8)
    df["discovery_hour"] = df["discovery_hour"].fillna(-1)

    # Cyclical encoding for periodic time features
    df["doy_sin"] = np.sin(2 * np.pi * df["discovery_doy"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["discovery_doy"] / 366)
    df["month_sin"] = np.sin(2 * np.pi * df["discovery_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["discovery_month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["discovery_dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["discovery_dow"] / 7)
    df["hour_sin"] = np.where(
        df["discovery_hour"] >= 0,
        np.sin(2 * np.pi * df["discovery_hour"] / 24),
        0.0,
    )
    df["hour_cos"] = np.where(
        df["discovery_hour"] >= 0,
        np.cos(2 * np.pi * df["discovery_hour"] / 24),
        0.0,
    )

    # 5-year period bucket so the model can pick up long-term shifts
    df["period_5yr"] = ((df["fire_year"] - 1990) // 5).astype(np.int8)

    # Geographic rounding to create coarse grid cells
    df["lat_bin"] = (df["latitude"] // 1).astype(np.int16)
    df["lon_bin"] = (df["longitude"] // 1).astype(np.int16)

    # Interaction: month × cause
    df["month_cause"] = (
        df["discovery_month"].astype(str) + "_" + df["nwcg_general_cause"].fillna("Unknown")
    )

    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode categoricals; unseen test values map to -1."""
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].fillna("__missing__").astype(str))
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        test[col] = (
            test[col].fillna("__missing__").astype(str).map(mapping).fillna(-1).astype(int)
        )
    return train, test


# ---------------------------------------------------------------------------
# Weighted log-loss (evaluation metric matching the competition scorer)
# ---------------------------------------------------------------------------

def weighted_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_counts: np.ndarray,
) -> float:
    """
    Compute inverse-frequency weighted multi-class log-loss.

    weight_c = N / (n_classes * count_c)
    """
    n = len(y_true)
    n_classes = y_pred.shape[1]
    weights = n / (n_classes * class_counts)

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    loss = 0.0
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            loss += weights[c] * (-np.log(y_pred[mask, c]).mean())

    return loss / weights.sum()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data …")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"  train shape: {train.shape}")
    print(f"  test  shape: {test.shape}")

    # Encode target
    label_order = {c: i for i, c in enumerate(CLASS_LABELS)}
    y = train[TARGET].map(label_order).astype(np.int8).values
    class_counts = np.array([int((y == i).sum()) for i in range(len(CLASS_LABELS))])
    print("  class distribution:")
    for i, c in enumerate(CLASS_LABELS):
        pct = 100 * class_counts[i] / len(y)
        print(f"    {c}: {class_counts[i]:>8,d}  ({pct:.2f}%)")

    # Feature engineering
    print("\nEngineering features …")
    train = engineer_features(train)
    test = engineer_features(test)

    # Categorical columns to encode
    cat_cols = ["nwcg_general_cause", "owner_descr", "state", "month_cause"]
    train, test = encode_categoricals(train, test, cat_cols)

    # Feature matrix
    drop_cols = [TARGET, "id", "discovery_time", "fire_year"]
    feature_cols = [c for c in train.columns if c not in drop_cols]
    print(f"  {len(feature_cols)} features: {feature_cols}")

    X = train[feature_cols].values
    X_test = test[feature_cols].values
    test_ids = test["id"].values

    # ---------------------------------------------------------------------------
    # Cross-validated LightGBM training with OOF predictions
    # ---------------------------------------------------------------------------
    print(f"\nTraining LightGBM with {N_FOLDS}-fold CV …")
    oof_preds = np.zeros((len(X), len(CLASS_LABELS)), dtype=np.float64)
    test_preds = np.zeros((len(X_test), len(CLASS_LABELS)), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    lgb_feature_cols = feature_cols  # used for LightGBM Dataset

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n  Fold {fold}/{N_FOLDS}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr,
            feature_name=lgb_feature_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            reference=dtrain,
            free_raw_data=False,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params={k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"},
            train_set=dtrain,
            num_boost_round=LGBM_PARAMS["n_estimators"],
            valid_sets=[dval],
            callbacks=callbacks,
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / N_FOLDS

        fold_loss = weighted_log_loss(y_val, oof_preds[val_idx], class_counts)
        print(f"  Fold {fold} weighted log-loss: {fold_loss:.5f}")

    # Overall OOF score
    oof_loss = weighted_log_loss(y, oof_preds, class_counts)
    print(f"\nOverall OOF weighted log-loss: {oof_loss:.5f}")

    # ---------------------------------------------------------------------------
    # Generate submission
    # ---------------------------------------------------------------------------
    print("\nWriting submission …")
    submission = pd.DataFrame(test_preds, columns=[f"class_{c}" for c in CLASS_LABELS])
    submission.insert(0, "id", test_ids)
    submission = submission[SUBMISSION_COLS]

    # Sanity check: probabilities sum to ~1
    row_sums = submission[[f"class_{c}" for c in CLASS_LABELS]].sum(axis=1)
    assert (np.abs(row_sums - 1.0) < 1e-6).all(), "Probabilities do not sum to 1!"

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}  ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
