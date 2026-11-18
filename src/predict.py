"""Classify new tickets with the fine-tuned model.

The model is loaded from MODEL_ID: a local folder (default ./model, the
unzipped ticket_model.zip from the notebook) or a Hugging Face Hub repo id.

Usage:
    python -m src.predict "I was charged twice for my subscription this month"
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

from transformers import pipeline

MODEL_ID = os.getenv("MODEL_ID", "model")


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID, top_k=None)


def predict(text: str, top_k: int = 3) -> list[dict]:
    """Return the top-k queues with probabilities, highest first."""
    scores = get_classifier()(text, truncation=True, max_length=256)
    scores = scores[0] if scores and isinstance(scores[0], list) else scores
    return sorted(scores, key=lambda s: s["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    ticket = " ".join(sys.argv[1:]) or "My laptop will not connect to the office VPN since this morning."
    for row in predict(ticket):
        print(f"{row['score']:.3f}  {row['label']}")
