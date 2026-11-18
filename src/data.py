"""Load, clean and split the customer support ticket dataset.

The same cleaning rules and seed are used by the notebook, the baseline
script and the fine-tuning run, so every model sees identical splits.
"""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "Tobi-Bueck/customer-support-tickets"
SEED = 42
MIN_PER_CLASS = 50


def load_tickets(language: str = "en", min_per_class: int = MIN_PER_CLASS) -> pd.DataFrame:
    """Return a cleaned DataFrame with columns `text` and `queue`."""
    raw = load_dataset(DATASET_ID, split="train").to_pandas()
    df = raw[raw["language"] == language].copy()
    df["text"] = (df["subject"].fillna("") + ". " + df["body"].fillna("")).str.strip(". ").str.strip()
    df = df[(df["text"].str.len() > 10) & df["queue"].notna()].drop_duplicates(subset=["text"])

    counts = df["queue"].value_counts()
    df = df[df["queue"].isin(counts[counts >= min_per_class].index)]
    return df[["text", "queue"]].reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = SEED):
    """Stratified 80 / 10 / 10 train, validation and test split."""
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["queue"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["queue"], random_state=seed)
    return train_df, val_df, test_df


def label_maps(df: pd.DataFrame):
    labels = sorted(df["queue"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return labels, label2id, id2label
