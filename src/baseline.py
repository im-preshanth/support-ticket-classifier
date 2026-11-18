"""TF-IDF + logistic regression baseline.

Usage:
    python -m src.baseline
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline, make_pipeline

from src.data import load_tickets, split


def build_baseline() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def main() -> None:
    train_df, _, test_df = split(load_tickets())
    model = build_baseline().fit(train_df["text"], train_df["queue"])
    pred = model.predict(test_df["text"])

    print(f"accuracy    {accuracy_score(test_df['queue'], pred):.3f}")
    print(f"macro_f1    {f1_score(test_df['queue'], pred, average='macro'):.3f}")
    print(f"weighted_f1 {f1_score(test_df['queue'], pred, average='weighted'):.3f}")
    print(classification_report(test_df["queue"], pred, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
