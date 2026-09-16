"""
src/preprocess.py

Phase 2 — Data preparation for the Medical NER fine-tuning pipeline.

Loads the NCBI-disease dataset (already in BIO-tagged token classification
format: 'tokens' + 'ner_tags' per example), caches it locally to data/, and
exposes a clean loader for Phase 3 (fine-tuning) to import.

Note: requires datasets==2.19.0 (pinned) since this dataset still uses the
older loading-script format, which datasets>=3.0 no longer supports.
"""

import os
from datasets import load_dataset, load_from_disk, DatasetDict

DATA_DIR = "data/ncbi_disease_processed"


def load_data() -> DatasetDict:
    """
    Load the NCBI-disease dataset, either from local disk cache (if already
    processed) or by downloading fresh from HuggingFace and caching it.
    """
    if os.path.exists(DATA_DIR):
        print(f"Loading cached dataset from {DATA_DIR} ...")
        dataset = load_from_disk(DATA_DIR)
    else:
        print("No local cache found — downloading ncbi_disease ...")
        dataset = load_dataset("ncbi/ncbi_disease", trust_remote_code=True)
        os.makedirs("data", exist_ok=True)
        dataset.save_to_disk(DATA_DIR)
        print(f"Saved processed dataset to {DATA_DIR}")
    return dataset


def get_label_list(dataset: DatasetDict) -> list[str]:
    """Return the BIO label names, e.g. ['O', 'B-Disease', 'I-Disease']."""
    return dataset["train"].features["ner_tags"].feature.names


def inspect(dataset: DatasetDict, label_list: list[str]) -> None:
    """Print basic stats and a sample so you can sanity-check the data."""
    print("\n--- Dataset summary ---")
    print("Labels:", label_list)
    print("Train size:     ", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))
    print("Test size:      ", len(dataset["test"]))

    print("\n--- Sample example (train[0]) ---")
    example = dataset["train"][0]
    print("Tokens:  ", example["tokens"])
    print("NER tags:", example["ner_tags"])

    # Show the tags as readable labels, not just integer ids
    readable = [label_list[tag] for tag in example["ner_tags"]]
    print("Readable:", list(zip(example["tokens"], readable)))


def main():
    dataset = load_data()
    label_list = get_label_list(dataset)
    inspect(dataset, label_list)
    return dataset, label_list


if __name__ == "__main__":
    main()