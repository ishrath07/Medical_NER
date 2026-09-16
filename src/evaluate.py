"""
src/evaluate.py

Phase 4 — Evaluate the fine-tuned model on the held-out TEST set,
computing entity-level (not token-level) precision/recall/F1 per type.
Writes a report to results/eval_report.md for the portfolio writeup.
"""

import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from collections import defaultdict

from preprocess import load_data, get_label_list
from train import tokenize_and_align_labels, get_entities, precision_score, recall_score, f1_score, classification_report

MODEL_DIR = "models/biobert-ncbi-disease-subset"
TEST_SUBSET_SIZE = 150  # keep test eval quick too; bump up if you want fuller numbers
REPORT_PATH = "results/eval_report.md"


def main():
    dataset = load_data()
    label_list = get_label_list(dataset)

    test_ds = dataset["test"].shuffle(seed=42).select(range(TEST_SUBSET_SIZE))
    print(f"Evaluating on test subset: {len(test_ds)} examples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

    test_tok = test_ds.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="tmp_eval", use_cpu=True, report_to=[]),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    raw_predictions = trainer.predict(test_tok)
    predictions = np.argmax(raw_predictions.predictions, axis=2)
    labels = raw_predictions.label_ids

    true_predictions = [
        [label_list[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    report_text = classification_report(true_labels, true_predictions)
    overall_p = precision_score(true_labels, true_predictions)
    overall_r = recall_score(true_labels, true_predictions)
    overall_f1 = f1_score(true_labels, true_predictions)

    print(f"\nOverall — Precision: {overall_p:.3f}  Recall: {overall_r:.3f}  F1: {overall_f1:.3f}")

    # --- Write markdown report ---
    with open(REPORT_PATH, "w") as f:
        f.write("# Evaluation Report — Medical NER (BioBERT / NCBI-disease)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Test examples evaluated: {len(test_ds)}\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(f"| Metric | Score |\n|---|---|\n")
        f.write(f"| Precision | {overall_p:.3f} |\n")
        f.write(f"| Recall | {overall_r:.3f} |\n")
        f.write(f"| F1 | {overall_f1:.3f} |\n\n")
        f.write("## Per-Entity-Type Breakdown\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n")

    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()