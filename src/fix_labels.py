"""
src/fix_labels.py

One-time fix: patch the saved model's config.json so it uses real label
names (O, B-Disease, I-Disease) instead of generic LABEL_0/1/2. No
retraining needed — this only affects how the pipeline interprets output.
"""

import json

MODEL_DIR = "models/biobert-ncbi-disease-subset"
CONFIG_PATH = f"{MODEL_DIR}/config.json"

LABEL_LIST = ["O", "B-Disease", "I-Disease"]

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

config["id2label"] = {str(i): label for i, label in enumerate(LABEL_LIST)}
config["label2id"] = {label: i for i, label in enumerate(LABEL_LIST)}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print("Patched id2label/label2id in config.json")
print(config["id2label"])