"""Level 0: class-weighted logistic regression.

    python level0_logreg.py           full run: writes the validation and test results
    python level0_logreg.py --smoke   quick code check: small sample, validation only, saves nothing

The settings below are fixed before the run and are not tuned. Everything that learns from
data (medians, scaling, categories, the model) is fitted on the training split only.
The alert threshold is chosen on validation and reused unchanged on test.
"""
import argparse
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud import CATEGORICAL, NOT_FEATURES, TARGET, choose_threshold, evaluate, load_splits
from mustcite import save_results

MAX_LEVELS = 64  # categorical columns with more values than this are left for Level 1 embeddings
C = 1.0
MAX_ITER = 1000
SMOKE_ROWS = 40_000


def fill_missing_category(X):
    """Treat a missing category as its own value."""
    return X.astype(object).where(X.notna(), "missing")


parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

train, val, test = load_splits()
if args.smoke:
    train = train.sample(SMOKE_ROWS, random_state=0)
    test = None  # a code check never touches the test split

categorical = [c for c in train.columns if c in CATEGORICAL]
low_cardinality = [c for c in categorical if train[c].nunique() <= MAX_LEVELS]
high_cardinality = [c for c in categorical if c not in low_cardinality]
numeric = [c for c in train.columns if c not in NOT_FEATURES and c not in categorical]

features = ColumnTransformer([
    ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("categorical", Pipeline([
        ("fill", FunctionTransformer(fill_missing_category)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ]), low_cardinality),
], sparse_threshold=0)
model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=MAX_ITER)),
])

start = time.time()
model.fit(train[numeric + low_cardinality], train[TARGET])
seconds = time.time() - start
classifier = model.named_steps["classifier"]
iterations = int(classifier.n_iter_[0])
print(f"fitted in {seconds:.0f}s, {iterations} iterations, {classifier.coef_.shape[1]} input columns")


def predict(df):
    return model.predict_proba(df[numeric + low_cardinality])[:, 1]


val_score = predict(val)
threshold = choose_threshold(val[TARGET], val_score)
val_metrics = evaluate(val[TARGET], val_score, threshold)

if args.smoke:
    print("smoke test, nothing saved:", {k: round(v, 4) for k, v in val_metrics.items()})
else:
    config = {
        "model": "logistic regression", "C": C, "class_weight": "balanced", "solver": "lbfgs",
        "max_iter": MAX_ITER, "iterations_used": iterations, "converged": iterations < MAX_ITER,
        "fit_seconds": round(seconds, 1), "train_rows": len(train),
        "numeric_columns": len(numeric), "one_hot_columns": low_cardinality,
        "dropped_high_cardinality_columns": high_cardinality, "max_levels": MAX_LEVELS,
        "input_columns_after_encoding": int(classifier.coef_.shape[1]),
        "numeric_preprocessing": "median imputation, standardization, fitted on train",
        "threshold_chosen_on": "validation",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    save_results("level0_logreg_val", val_metrics, config)
    save_results("level0_logreg_test", evaluate(test[TARGET], predict(test), threshold), config)
