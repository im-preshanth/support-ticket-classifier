"""Gradio demo: paste a support ticket and see the predicted queue.

Run locally:  MODEL_ID=model python app/app.py
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.predict import predict  # noqa: E402

EXAMPLES = [
    "I was charged twice for my subscription this month and need a refund.",
    "Our dashboard has been down for all users since 9 AM. Is there an outage?",
    "The item I received is damaged. How do I return it for a replacement?",
    "Can you send pricing for 50 licenses and set up a demo call next week?",
]


def classify(text: str) -> dict:
    if not text.strip():
        return {}
    return {row["label"]: float(row["score"]) for row in predict(text, top_k=5)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=6, label="Support ticket (subject and body)"),
    outputs=gr.Label(num_top_classes=5, label="Predicted queue"),
    examples=EXAMPLES,
    title="Support Ticket Routing Classifier",
    description="Fine-tuned BERT that routes English support tickets to one of 10 queues. "
                "Trained on a synthetic public dataset; see the model card for limitations.",
)

if __name__ == "__main__":
    demo.launch()
