"""Stage 4 - Detractor risk model.

Purpose: rank the drivers of a one/two-star review, and quantify how much of
the outcome is explained by things the marketplace controls. The model is
evidence for the recommendations; it is not the deliverable.

Two models are fitted deliberately:

  * `post_hoc`  - includes the realised delivery outcome. Answers "what
                  explains a bad review after the fact?"
  * `at_risk`   - uses only what is knowable at checkout (category, seller
                  history, distance, promise, basket). Answers "can we flag a
                  risky order before it ships?" This is the one an operations
                  team could actually deploy.

Evaluation is a time-based split, not a random one: training on future orders
to predict past ones would inflate the score and would not survive review.

Run:  python -m src.model
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src import config as C
from src.analysis import analysis_base, load_facts

RANDOM_STATE = 42

NUMERIC_COMMON = [
    "item_total", "freight_total", "freight_ratio", "n_items", "n_sellers",
    "n_categories", "installments", "distance_km", "promised_days",
    "avg_weight_g", "avg_photos", "avg_description_length", "purchase_hour",
]
CATEGORICAL_COMMON = [
    "main_category", "customer_state", "seller_state", "payment_type",
    "purchase_dow",
]
SELLER_HISTORY = ["seller_prior_orders", "seller_prior_late_rate",
                  "seller_prior_score"]
OUTCOME_FEATURES = ["delivery_days", "delay_days", "handling_days",
                    "transit_days"]


# --------------------------------------------------------------------------
# Leakage-safe seller history
# --------------------------------------------------------------------------
def add_seller_history(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding, strictly-prior seller statistics.

    A seller's lifetime average score cannot be used to predict an order that
    is itself part of that average. These columns are shifted so each row sees
    only the seller's earlier orders.
    """
    # Stable sort with an explicit tiebreaker. 566 orders share a purchase
    # timestamp, and the default quicksort orders those ties differently from
    # run to run - which shifts the expanding means below, and moved the test
    # AUC by ~0.004 between otherwise identical builds.
    df = df.sort_values(["purchased_at", "order_id"], kind="mergesort").copy()
    g = df.groupby("main_seller_id", observed=True)
    df["seller_prior_orders"] = g.cumcount()
    df["seller_prior_late_rate"] = (
        g["is_late"].apply(lambda s: s.shift().expanding().mean()).reset_index(
            level=0, drop=True)
    )
    df["seller_prior_score"] = (
        g["review_score"].apply(lambda s: s.shift().expanding().mean()).reset_index(
            level=0, drop=True)
    )
    return df


def make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Gradient boosting on mixed types.

    HistGradientBoosting handles NaNs natively, which matters here: missing is
    informative (a seller with no history is a new seller) and imputing it
    would erase that signal.
    """
    pre = ColumnTransformer(
        [
            ("cat",
             OrdinalEncoder(handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1),
             categorical),
            ("num", "passthrough", numeric),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=list(range(len(categorical))),
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def time_split(df: pd.DataFrame, holdout_frac: float = 0.25):
    """Train on the earlier orders, test on the most recent ones."""
    cutoff = df.purchased_at.quantile(1 - holdout_frac)
    return df[df.purchased_at < cutoff], df[df.purchased_at >= cutoff], cutoff


def fit_and_score(df: pd.DataFrame, numeric, categorical, label: str) -> dict:
    features = categorical + numeric
    train, test, cutoff = time_split(df)

    X_tr, y_tr = train[features], train.is_bad_review.astype(int)
    X_te, y_te = test[features], test.is_bad_review.astype(int)

    pipe = make_pipeline(numeric, categorical)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]

    # Baseline: always predict the training base rate.
    base_rate = y_tr.mean()

    metrics = {
        "model": label,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "cutoff": str(cutoff),
        "test_bad_rate": float(y_te.mean()),
        "roc_auc": float(roc_auc_score(y_te, p)),
        "pr_auc": float(average_precision_score(y_te, p)),
        "pr_auc_baseline": float(y_te.mean()),
        "brier": float(brier_score_loss(y_te, p)),
        "brier_baseline": float(brier_score_loss(y_te, np.full(len(y_te), base_rate))),
    }

    # Capture rate: if operations can only review the riskiest 10% of orders,
    # what share of eventual detractors does that 10% contain?
    order = np.argsort(-p)
    for frac in (0.05, 0.10, 0.20):
        k = int(len(p) * frac)
        metrics[f"detractor_capture_top{int(frac * 100)}pct"] = float(
            y_te.iloc[order[:k]].sum() / y_te.sum()
        )

    # Permutation importance on a sample, for runtime.
    sample = test.sample(min(12_000, len(test)), random_state=RANDOM_STATE)
    imp = permutation_importance(
        pipe, sample[features], sample.is_bad_review.astype(int),
        n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    importance = (
        pd.DataFrame({"feature": features,
                      "importance": imp.importances_mean,
                      "std": imp.importances_std})
        .sort_values("importance", ascending=False)
    )
    importance.insert(0, "model", label)
    return {"metrics": metrics, "importance": importance, "pipeline": pipe,
            "test": test.assign(risk=p)}


def main() -> None:
    df = analysis_base(load_facts())
    df = add_seller_history(df)
    df = df[df.is_bad_review.notna()]

    results = {}

    results["post_hoc"] = fit_and_score(
        df,
        NUMERIC_COMMON + SELLER_HISTORY + OUTCOME_FEATURES,
        CATEGORICAL_COMMON,
        "post_hoc (includes realised delivery)",
    )
    results["at_risk"] = fit_and_score(
        df,
        NUMERIC_COMMON + SELLER_HISTORY,
        CATEGORICAL_COMMON,
        "at_risk (checkout-time features only)",
    )

    metrics = pd.DataFrame([r["metrics"] for r in results.values()])
    importance = pd.concat([r["importance"] for r in results.values()])

    metrics.to_csv(C.TABLE_DIR / "model_metrics.csv", index=False)
    importance.to_csv(C.TABLE_DIR / "model_importance.csv", index=False)

    # Risk decile calibration for the at-risk model, used in the report.
    t = results["at_risk"]["test"]
    dec = (
        t.assign(decile=pd.qcut(t.risk, 10, labels=False, duplicates="drop"))
        .groupby("decile")
        .agg(orders=("order_id", "size"),
             predicted=("risk", "mean"),
             actual=("is_bad_review", "mean"),
             avg_score=("review_score", "mean"))
        .reset_index()
    )
    dec.to_csv(C.TABLE_DIR / "model_calibration.csv", index=False)

    with open(C.TABLE_DIR / "model_summary.json", "w") as fh:
        json.dump(metrics.to_dict(orient="records"), fh, indent=2)

    print(metrics[["model", "roc_auc", "pr_auc", "pr_auc_baseline",
                   "detractor_capture_top10pct", "brier",
                   "brier_baseline"]].round(4).to_string(index=False))
    print("\nTop drivers (at-risk model):")
    print(importance[importance.model.str.startswith("at_risk")]
          .head(10)[["feature", "importance"]].round(4).to_string(index=False))
    print("\nTop drivers (post-hoc model):")
    print(importance[importance.model.str.startswith("post_hoc")]
          .head(8)[["feature", "importance"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
