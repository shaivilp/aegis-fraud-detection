"""Check the data before modeling.

    python data_check.py

Prints the number of examples per class and split, saves every number to
results/data_check.json, and writes samples.md with three examples of each class.
The samples come from the training split only, so the test rows stay unread.
"""
import pandas as pd

from fraud import DATA_DIR, ID, NOT_FEATURES, ROOT, SPLITS, TARGET, TIME, load_splits
from mustcite import save_results

SECONDS_PER_DAY = 86400
CARD_COLUMNS = [f"card{i}" for i in range(1, 7)]
CLASS_NAMES = {0: "not fraud", 1: "fraud"}
# A few readable columns only: no transaction ids and no card or address codes.
SAMPLE_COLUMNS = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]


def card_key(df):
    """All six card columns joined into one string: an approximate id for a card."""
    key = df[CARD_COLUMNS[0]].astype(object).fillna("na").astype(str)
    for column in CARD_COLUMNS[1:]:
        key = key + "|" + df[column].astype(object).fillna("na").astype(str)
    return key


def split_summary(df, features):
    fraud = int(df[TARGET].sum())
    return {
        "rows": len(df),
        "not_fraud": len(df) - fraud,
        "fraud": fraud,
        "fraud_rate": fraud / len(df),
        "first_day": float(df[TIME].min() / SECONDS_PER_DAY),
        "last_day": float(df[TIME].max() / SECONDS_PER_DAY),
        "share_with_identity": float(df["has_identity"].mean()),
        "share_of_feature_values_missing": float(df[features].isna().mean().mean()),
    }


def format_value(value):
    return "missing" if pd.isna(value) else str(value)


def write_samples(train, features, path):
    lines = [
        "# Sample transactions",
        "",
        "Three transactions of each class, drawn at random (seed 0) from the training split by `data_check.py`.",
        "Only a few readable columns are shown. The full data is not in this repository.",
        "`day` counts days from Kaggle's reference point. `missing` is the share of this row's features with no value.",
        "",
        "| Label | day | " + " | ".join(SAMPLE_COLUMNS) + " | has identity data | missing |",
        "|---|---:|" + "---|" * len(SAMPLE_COLUMNS) + "---|---:|",
    ]
    for label, name in CLASS_NAMES.items():
        for _, row in train[train[TARGET] == label].sample(3, random_state=0).iterrows():
            values = [f"{row['TransactionAmt']:.2f}"] + [format_value(row[c]) for c in SAMPLE_COLUMNS[1:]]
            missing = row[features].isna().mean()
            lines.append(f"| {label} ({name}) | {row[TIME] / SECONDS_PER_DAY:.1f} | " + " | ".join(values)
                         + f" | {'yes' if row['has_identity'] else 'no'} | {missing:.0%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


print(f"Reading the data from {DATA_DIR}")
splits = dict(zip(SPLITS, load_splits()))
train, val, test = splits["train"], splits["val"], splits["test"]
features = [c for c in train.columns if c not in NOT_FEATURES]

# The split must be clean before anything else is trusted.
ids = {name: set(df[ID]) for name, df in splits.items()}
checks = {
    "ids_unique": sum(len(s) for s in ids.values()) == sum(len(df) for df in splits.values()),
    "splits_disjoint": not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]),
    "time_order": bool(train[TIME].max() < val[TIME].min() and val[TIME].max() < test[TIME].min()),
    "both_classes_in_every_split": all(df[TARGET].nunique() == 2 for df in splits.values()),
}
assert all(checks.values()), checks

# Facts from the training split only.
by_class = train.groupby(TARGET)
by_identity = train.groupby("has_identity")[TARGET].mean()
train_keys = card_key(train).unique()
metrics = {
    "total_rows": sum(len(df) for df in splits.values()),
    "feature_columns": len(features),
    "checks": checks,
    "splits": {name: split_summary(df, features) for name, df in splits.items()},
    "train_share_with_identity_by_class": {CLASS_NAMES[k]: float(v) for k, v in by_class["has_identity"].mean().items()},
    "train_fraud_rate_with_identity": float(by_identity[True]),
    "train_fraud_rate_without_identity": float(by_identity[False]),
    "train_median_amount_by_class": {CLASS_NAMES[k]: float(v) for k, v in by_class["TransactionAmt"].median().items()},
    "train_fraud_rate_by_product": {str(k): float(v) for k, v in
                                    train.groupby("ProductCD", observed=True)[TARGET].mean().items()},
    # Rows whose card key was already seen in training. This uses features only, no labels.
    "val_share_of_rows_with_card_seen_in_train": float(card_key(val).isin(train_keys).mean()),
    "test_share_of_rows_with_card_seen_in_train": float(card_key(test).isin(train_keys).mean()),
}

print("\n| Class | Train | Validation | Test |\n|---|---:|---:|---:|")
for key, name in (("not_fraud", "Not fraud (0)"), ("fraud", "Fraud (1)"), ("rows", "Total")):
    print(f"| {name} | " + " | ".join(f"{metrics['splits'][s][key]:,}" for s in SPLITS) + " |")
print()

write_samples(train, features, ROOT / "samples.md")
save_results("data_check", metrics, {"split": "time order: first 70% train, next 15% val, last 15% test",
                                     "card_key": "card1 to card6 joined", "sample_seed": 0})
