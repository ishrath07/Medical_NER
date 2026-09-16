# Medical / Legal Entity Extractor (NER)

A domain-tuned Named Entity Recognition system for extracting disease mentions from unstructured clinical text, with an interactive Streamlit demo for uploading documents and viewing entities highlighted inline.

## Problem

Clinics and small legal teams manually extract dates, names, drug names, or clauses from unstructured documents — a slow, error-prone process. This project targets the medical side: automatically identifying disease mentions in clinical notes.

## Build

- **Base model:** [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) — BERT pretrained on biomedical literature (PubMed abstracts + PMC full-text articles)
- **Dataset:** [NCBI-disease](https://huggingface.co/datasets/ncbi/ncbi_disease) — 5,433 train / 924 validation / 941 test sentences, BIO-tagged for disease mentions (`O`, `B-Disease`, `I-Disease`)
- **Fine-tuning:** HuggingFace `transformers` `Trainer` API, token classification, trained on CPU using a 500-example training subset (3 epochs, batch size 8) to keep iteration fast for this demo. Full-dataset training is a documented next step.
- **Demo:** Streamlit app — upload a `.txt` or `.pdf` clinical document, run inference, and view entities highlighted inline via spaCy's `displacy` renderer, plus a structured entity table.

## Demo

![Demo screenshot](assets/demo_screenshot.png)

Upload a clinical `.txt` or `.pdf` document and disease entities are highlighted inline, with a structured table view below.

## Metrics

Entity-level precision/recall/F1 (not token-level), evaluated on a 150-example held-out test subset:

| Metric | Score |
|---|---|
| Precision | 0.821 |
| Recall | 0.782 |
| F1 | 0.801 |

Full breakdown in [`results/eval_report.md`](results/eval_report.md).

*Note: these numbers reflect training on a 500-example subset of NCBI-disease for demo/portfolio purposes rather than the full 5,433-example training set. Training on the full dataset is expected to improve recall further and is a natural next step.*

## Project Structure

```
Medical_NER/
├── assets/
│   └── demo_screenshot.png        # Demo screenshot for README
├── data/                          # Cached datasets (gitignored, regenerated via preprocess.py)
├── models/                        # Fine-tuned model weights (gitignored — see below)
├── results/
│   └── eval_report.md             # Entity-level evaluation report
├── src/
│   ├── preprocess.py              # Loads & caches NCBI-disease dataset
│   ├── baseline_ner.py            # Phase 1: pretrained scispaCy baseline
│   ├── train.py                   # Fine-tunes BioBERT on NCBI-disease subset
│   ├── evaluate.py                # Entity-level F1 evaluation on test set
│   └── fix_labels.py              # One-time utility: patches model config label names
├── app.py                         # Streamlit demo
├── requirements.txt
└── README.md
```

## Setup & Usage

```bash
# 1. Clone and set up environment
git clone https://github.com/ishrath07/Medical_NER.git
cd Medical_NER
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Prepare data
python src/preprocess.py

# 3. Train the model (fine-tunes BioBERT on a subset; ~10-15 min on CPU)
python src/train.py

# 4. Evaluate
python src/evaluate.py

# 5. Run the demo
streamlit run app.py
```

**Note on model weights:** trained model weights are not included in this repository (kept out via `.gitignore` due to size). Run `train.py` to reproduce the model locally, or contact me for a pre-trained checkpoint.

## Tech Stack

Python · HuggingFace `transformers` & `datasets` · PyTorch · spaCy (`displacy`) · Streamlit · pypdf

## Future Improvements

- Train on the full NCBI-disease dataset (not just the 500-example subset) for higher recall
- Extend to additional entity types (Chemical, Drug) using a multi-type biomedical NER dataset
- Add a legal-domain NER track (LegalNER dataset + `nlpaueb/legal-bert-base-uncased`)
- Deploy the Streamlit demo publicly via Streamlit Community Cloud
