"""Untrained reference estimators for the proposed comparison."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier


def build_models(seed: int = 42) -> dict[str, object]:
    """Return fresh estimators; fit and tune only inside training partitions.

    The length reference learns at most two split levels from n_chars, rather
    than choosing thresholds after looking at test performance.
    """
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "length": Pipeline([
            ("select_length", ColumnTransformer([("length", "passthrough", ["n_chars"])])),
            ("tree", DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=seed)),
        ]),
        "logistic": Pipeline([
            ("scale", RobustScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }
