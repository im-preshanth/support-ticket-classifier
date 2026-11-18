---
language: en
license: apache-2.0
base_model: bert-base-uncased
pipeline_tag: text-classification
datasets:
  - Tobi-Bueck/customer-support-tickets
metrics:
  - f1
  - accuracy
---

# Support Ticket Routing Classifier (BERT)

`bert-base-uncased` fine-tuned to route English customer support tickets to one of 10 queues.

## Intended use

Suggesting the destination queue for an incoming support ticket (subject + body), ideally as a top-3 suggestion for a human triage agent. Not intended for automated decisions on real customers without retraining and evaluation on in-house data.

## Labels

Billing and Payments, Customer Service, General Inquiry, Human Resources, IT Support, Product Support, Returns and Exchanges, Sales and Pre-Sales, Service Outages and Maintenance, Technical Support.

## Training data

English subset of `Tobi-Bueck/customer-support-tickets` (synthetic). 23,745 tickets after cleaning and de-duplication; stratified 80/10/10 split with seed 42 (18,996 / 2,374 / 2,375).

## Training procedure

- Max sequence length 256, learning rate 3e-5, 700 warmup steps, weight decay 0.01
- Batch size 16, 6 epochs, fp16, Google Colab T4 GPU (about 19 minutes)
- Cross-entropy loss with inverse-frequency class weights
- Best epoch selected by validation macro-F1 (0.595 at epoch 6)

## Evaluation (held-out test set, 2,375 tickets)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + LogReg baseline | 0.504 | 0.507 | 0.505 |
| This model | 0.531 | 0.579 | 0.528 |

Strongest queues: Billing and Payments (F1 0.799), Human Resources (0.731). Weakest: Product Support (0.424), General Inquiry (0.452).

## Limitations

- Trained on synthetic tickets; performance on real tickets is unknown.
- Technical Support, IT Support and Product Support overlap heavily and are frequently confused with each other.
- English only. Single training run and seed.

## How to use

```python
from transformers import pipeline

clf = pipeline("text-classification", model="<hf-username>/support-ticket-classifier", top_k=3)
clf("I was charged twice for my subscription this month and need a refund.")
```
