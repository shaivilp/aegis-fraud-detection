"""Load the IEEE-CIS fraud data, split it by time and score predictions.

Every script in this repository uses these functions, so every model sees the same
split and is scored the same way. The Kaggle files are not in the repository. They
live in ieee_data/, which git ignores (see download_data.py).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IEEE_DATA_DIR", ROOT / "ieee_data"))

ID, TIME, TARGET = "TransactionID", "TransactionDT", "isFraud"
SPLITS = ("train", "val", "test")
VAL_START, TEST_START = 0.70, 0.85  # shares of the rows, in time order
PRECISION_TARGET = 0.50  # at least half of the flagged transactions should be fraud

# The columns Kaggle's data description lists as categorical. Several are stored as
# numbers, but those numbers are codes, not quantities.
CATEGORICAL = (
    ["ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    + [f"card{i}" for i in range(1, 7)]
    + [f"M{i}" for i in range(1, 10)]
    + [f"id_{i}" for i in range(12, 39)]
)
NOT_FEATURES = {ID, TIME, TARGET, "has_identity"}


def _read_csv(name):
    """Read one Kaggle file with compact column types, so it fits in memory."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_data.py first.")
    dtypes = {}
    for column in pd.read_csv(path, nrows=0).columns:
        if column in CATEGORICAL:
            dtypes[column] = "category"
        elif column in (ID, TIME):
            dtypes[column] = "int32"
        elif column == TARGET:
            dtypes[column] = "int8"
        else:
            dtypes[column] = "float32"
    return pd.read_csv(path, dtype=dtypes)


def load_data():
    """All labeled transactions, joined with identity data where it exists, in time order."""
    transactions = _read_csv("train_transaction.csv")
    identity = _read_csv("train_identity.csv")
    transactions["has_identity"] = transactions[ID].isin(identity[ID])
    df = transactions.merge(identity, on=ID, how="left", validate="one_to_one")
    return df.sort_values([TIME, ID], ignore_index=True)


def split_times(times):
    """The first timestamp of validation and of test.

    Cutting on timestamps keeps transactions with the same timestamp in the same split.
    """
    ordered = np.sort(times)
    return ordered[int(len(ordered) * VAL_START)], ordered[int(len(ordered) * TEST_START)]


def load_splits():
    """(train, val, test): the first 70% of the time-ordered rows, the next 15%, the last 15%."""
    df = load_data()
    val_start, test_start = split_times(df[TIME].to_numpy())
    train = df[df[TIME] < val_start].reset_index(drop=True)
    val = df[(df[TIME] >= val_start) & (df[TIME] < test_start)].reset_index(drop=True)
    test = df[df[TIME] >= test_start].reset_index(drop=True)
    return train, val, test


def choose_threshold(y_true, y_score, precision_target=PRECISION_TARGET):
    """The lowest score threshold that reaches the target precision. Call this on validation only."""
    y_score = np.asarray(y_score, dtype=float)
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    reaches_target = np.flatnonzero(precision[:-1] >= precision_target)
    if len(reaches_target) == 0:
        return float(y_score.max()) + 1.0  # no threshold works, so flag nothing
    return float(thresholds[reaches_target[0]])


def evaluate(y_true, y_score, threshold):
    """PR-AUC, ROC-AUC, and precision and recall at a threshold chosen on validation."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    flagged = y_score >= threshold
    true_positives = int((flagged & (y_true == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
        "precision_target": PRECISION_TARGET,
        "precision_at_threshold": true_positives / max(int(flagged.sum()), 1),
        "recall_at_threshold": true_positives / max(int(y_true.sum()), 1),
        "flagged": int(flagged.sum()),
        "rows": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),  # also the PR-AUC of random guessing
    }
