"""
src/train.py

Phase 3 — Fine-tune BioBERT for disease NER on a SUBSET of NCBI-disease.
Designed to run on CPU in a reasonable time (~5-15 min) for demo purposes.
"""

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from collections import defaultdict

from preprocess import load_data, get_label_list

MODEL_CHECKPOINT = "dmis-lab/biobert-base-cased-v1.1"
OUTPUT_DIR = "models/biobert-ncbi-disease-subset"

# --- Subset sizes: tune these based on how much time you have ---
TRAIN_SUBSET_SIZE = 500
EVAL_SUBSET_SIZE = 100


# --- Pure-Python entity-level F1 (no seqeval dependency) ---
def get_entities(seq):
    entities = []
    entity_type, start = None, None
    for i, tag in enumerate(seq + ["O"]):
        if tag.startswith("B-"):
            if entity_type is not None:
                entities.append((entity_type, start, i - 1))
            entity_type, start = tag[2:], i
        elif tag.startswith("I-") and entity_type == tag[2:]:
            continue
        else:
            if entity_type is not None:
                entities.append((entity_type, start, i - 1))
            entity_type, start = None, None
    return entities


def precision_score(true_labels, pred_labels):
    tp, fp = 0, 0
    for t, p in zip(true_labels, pred_labels):
        ts, ps = set(get_entities(t)), set(get_entities(p))
        tp += len(ts & ps)
        fp += len(ps - ts)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(true_labels, pred_labels):
    tp, fn = 0, 0
    for t, p in zip(true_labels, pred_labels):
        ts, ps = set(get_entities(t)), set(get_entities(p))
        tp += len(ts & ps)
        fn += len(ts - ps)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(true_labels, pred_labels):
    p, r = precision_score(true_labels, pred_labels), recall_score(true_labels, pred_labels)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def classification_report(true_labels, pred_labels):
    tp_d, fp_d, fn_d = defaultdict(int), defaultdict(int), defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        ts, ps = set(get_entities(t)), set(get_entities(p))
        for e in ts & ps:
            tp_d[e[0]] += 1
        for e in ps - ts:
            fp_d[e[0]] += 1
        for e in ts - ps:
            fn_d[e[0]] += 1
    lines = [f"{'type':15}{'precision':>10}{'recall':>10}{'f1':>10}"]
    for label in sorted(set(tp_d) | set(fp_d) | set(fn_d)):
        tp, fp, fn = tp_d[label], fp_d[label], fn_d[label]
        p_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        lines.append(f"{label:15}{p_:10.3f}{r_:10.3f}{f1_:10.3f}")
    report = "\n".join(lines)
    print(report)
    return report


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def build_compute_metrics(label_list):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for p, l in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for p, l in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        classification_report(true_labels, true_predictions)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    return compute_metrics


def main():
    print("Loading dataset...")
    dataset = load_data()
    label_list = get_label_list(dataset)
    print(f"Labels: {label_list}")

    # --- Take a subset for fast CPU training ---
    train_ds = dataset["train"].shuffle(seed=42).select(range(TRAIN_SUBSET_SIZE))
    eval_ds = dataset["validation"].shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
    print(f"Using subset: train={len(train_ds)}, eval={len(eval_ds)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print("Tokenizing...")
    train_tok = train_ds.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True
    )
    eval_tok = eval_ds.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True
    )

    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(label_list)
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(label_list),
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()