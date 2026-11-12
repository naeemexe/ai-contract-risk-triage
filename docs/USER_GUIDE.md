# User Guide

How to run the contract risk triage pipeline, read its outputs, and adapt it. For every command-line option and Python function, see the [API Reference](API.md).

## Contents

1. [Who this is for](#1-who-this-is-for)
2. [Setup](#2-setup)
3. [Five-minute tour (no GPU)](#3-five-minute-tour-no-gpu)
4. [Full pipeline](#4-full-pipeline)
5. [Reading the outputs](#5-reading-the-outputs)
6. [Customizing risk rules](#6-customizing-risk-rules)
7. [Scoring your own contracts](#7-scoring-your-own-contracts)
8. [Running the tests](#8-running-the-tests)
9. [Reproducing the reported results](#9-reproducing-the-reported-results)
10. [Troubleshooting](#10-troubleshooting)

## 1. Who this is for

- **ML practitioners** who want to reproduce or extend the clause classifier.
- **Reviewers and analysts** who want a ranked list of contracts to review first, with reasons.

## 2. Setup

```bash
git clone https://github.com/naeemexe/ai-contract-risk-triage.git
cd ai-contract-risk-triage
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root. The full CUAD dataset ships in `data/cuad/`.

| Hardware | What you can run |
|---|---|
| Laptop CPU | Quickstart notebook, TF-IDF baseline, risk scoring, tests |
| Apple Silicon (MPS) or NVIDIA GPU | Full DistilRoBERTa training (~10 min/epoch) |
| Google Colab T4 (free) | Full training via `notebooks/naeem_distilroberta_finetune_colab.ipynb` |

## 3. Five-minute tour (no GPU)

```bash
jupyter notebook notebooks/quickstart_demo.ipynb
```

Run all cells. The notebook loads the 10-contract sample, trains the TF-IDF baseline, scores each contract's risk, and explains the top-ranked contract line by line.

Or, from the terminal:

```bash
python risk/score_contracts.py --source gold --test data/sample/sample_contracts.json --out risk/outputs/sample
```

## 4. Full pipeline

| Step | Command | Output |
|---|---|---|
| 1. Prepare data | `python training/prepare_dataset.py` | `training/data/` |
| 2. Train | `python training/train_distilroberta.py --epochs 3 --pos-weight-mode sqrt --out training/outputs/sqrt_weighted` | `training/outputs/sqrt_weighted/` |
| 3. Evaluate | `python training/evaluate.py --runs training/outputs/sqrt_weighted` | `training/outputs/evaluation/` |
| 4. Score risk | `python risk/score_contracts.py --source model --run training/outputs/sqrt_weighted` | `risk/outputs/model_sqrt_weighted/` |

Step 1 fails on purpose if any contract text is not covered by its chunks, so silent data loss cannot happen.

For a fast end-to-end check before a long run, add `--limit 64 --epochs 1` to step 2.

## 5. Reading the outputs

### Training (`training/outputs/<run>/`)

| File | Meaning |
|---|---|
| `best_model/` | Model and tokenizer from the epoch with the best validation macro F1. Load with `AutoModelForSequenceClassification.from_pretrained`. |
| `history.csv` | Train loss and validation F1 per epoch. Use it to spot overfitting. |
| `predictions_{val,test}.npz` | Per-chunk probabilities, labels, and masks for `evaluate.py`. |
| `chunk_level_metrics.csv`, `contract_level_metrics.csv` | Per-category precision, recall, and F1 at threshold 0.5. |

### Evaluation (`training/outputs/evaluation/`)

| File | Meaning |
|---|---|
| `run_summary.csv` | Micro and macro F1 per run, threshold scheme, and level. Start here. |
| `per_category_<run>.csv` | Precision, recall, and F1 for each of the 41 categories. |
| `thresholds_<run>.csv` | Tuned decision threshold for each category. |
| `tfidf_comparison.csv` | Side-by-side F1 with the TF-IDF baseline. |
| `error_analysis_<run>.csv` | Top false positives and false negatives per category, with the chunk text. |

**Chunk level vs. contract level.** Contract level asks "does this contract contain the clause?" and is what triage needs. Chunk level asks "is the clause in this chunk?" and is harder.

### Risk scoring (`risk/outputs/<source>/`)

| File | Meaning |
|---|---|
| `contract_scores.csv` | Contracts ranked by score, with bucket (Low / Medium / High) and a `reasons` column listing every point. |
| `clause_register.csv` | One row per detected clause: risk level, confidence, points, and evidence text. |
| `sensitivity.csv` | How many contracts change bucket if cutoffs or weights move. Low numbers mean a stable ranking. |
| `model_vs_gold.csv` | (Model source only) Rank correlation and bucket agreement with the gold-label ranking. |

Example `reasons` value:

```text
Exclusivity (High, p=1.00) +3.00; No cap on liability +2.00; Anti-Assignment (Medium, p=1.00) +1.00
```

## 6. Customizing risk rules

All weights live in data files, so you can change scoring without editing code.

- **Category risk levels:** edit the `initial_risk` column in `evaluation/risk_categories.csv` (`Low`, `Medium`, or `High`).
- **Weights, floors, buckets, combinations:** edit `risk/risk_rules.json`.

| Key | Effect |
|---|---|
| `level_weights` | Points per detected clause at each risk level (default High 3, Medium 1, Low 0.5). |
| `severity_floor` | A High-risk clause detected with at least `min_confidence` pushes the contract to at least `floor_bucket`. |
| `missing_protections` | Points added when a protective clause is absent, such as no Cap On Liability. |
| `combinations` | Points added when all listed clauses appear together. |
| `buckets` | Top `high_top_pct`% of training scores are High, the next `medium_next_pct`% are Medium. |
| `evidence_rules` | Regex backstops that qualify or rescue model detections (model source only). |

After editing, re-run `risk/score_contracts.py` and check `sensitivity.csv`.

## 7. Scoring your own contracts

The scripts read CUAD's SQuAD-style JSON. To score new contracts:

1. Put each contract's plain text in a `context` field, using the same structure as `data/sample/sample_contracts.json`. The `qas` list can hold the 41 questions with empty `answers`.
2. Chunk and predict with the trained model: run `training/prepare_dataset.py --test your_contracts.json`, then reuse the prediction code in `train_distilroberta.py`, or load `best_model/` directly.
3. Score: `python risk/score_contracts.py --source model --run <run_dir> --chunks training/data/test_chunks.csv`.

This flow is research code, not a hardened service. Review results before acting on them.

## 8. Running the tests

```bash
python -m unittest tests.test_score_contracts
```

The tests cover the risk scorer's evidence rules and scoring logic. They run in under a second and need no model.

## 9. Reproducing the reported results

```bash
python training/prepare_dataset.py
python training/train_distilroberta.py --epochs 3 --out training/outputs/baseline
python training/train_distilroberta.py --epochs 3 --pos-weight-mode sqrt --out training/outputs/sqrt_weighted
python training/evaluate.py --runs training/outputs/baseline training/outputs/sqrt_weighted
```

The seed defaults to 42. Expect small differences across hardware (MPS vs. CUDA). Reference numbers are in `training/results/distilroberta_baseline/`.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| `FileNotFoundError: training/data/...` | Run `training/prepare_dataset.py` first. |
| CUDA out of memory | Lower `--batch-size` to 8. |
| Training is very slow | You are on CPU. Use Colab with a T4 GPU, or `--limit 64` for a smoke test. |
| `--run is required with --source model` | Pass the run directory from step 2. |
| Thresholds file not found | Run `training/evaluate.py` on that run first, or pass `--threshold-scheme 0.5`. |
