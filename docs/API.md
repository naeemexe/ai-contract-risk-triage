# API Reference

Command-line interfaces and main Python functions for each pipeline stage. For a task-oriented walkthrough, see the [User Guide](USER_GUIDE.md). All paths default to locations relative to the repository root.

- [`training/prepare_dataset.py`](#trainingprepare_datasetpy)
- [`training/train_distilroberta.py`](#trainingtrain_distilrobertapy)
- [`training/evaluate.py`](#trainingevaluatepy)
- [`risk/score_contracts.py`](#riskscore_contractspy)
- [Data formats](#data-formats)

---

## `training/prepare_dataset.py`

Chunks every contract with the paragraph-aware chunker (480 tokens, 64-token overlap), labels each chunk, and tokenizes it for DistilRoBERTa.

### CLI

| Option | Default | Description |
|---|---|---|
| `--train` | `data/cuad/train_separate_questions.json` | CUAD training JSON |
| `--test` | `data/cuad/test.json` | CUAD test JSON |
| `--tokenizer` | `distilroberta-base` | Hugging Face tokenizer name |
| `--out` | `training/data` | Output directory |
| `--max-length` | `512` | Maximum sequence length |
| `--min-train-positives` | `10` | Drop categories with fewer positive training chunks than this |

### Outputs

`train_chunks.csv`, `test_chunks.csv` (chunk text and labels) · `train_tokenized.pt`, `test_tokenized.pt` (`input_ids`, `labels`, `label_mask`, contract titles) · `labels.json` (ordered category list = model output heads) · `dataset_summary.csv` · `chunking_summary.csv`

### Labeling rule

| Situation | Label |
|---|---|
| Clause span fully inside the chunk | `1` |
| Clause span only partly inside the chunk | masked (ignored in loss and metrics) |
| No clause span in the chunk | `0` |

### Functions

| Function | Description |
|---|---|
| `check_chunks(chunks_df, dataset, chunker)` | Asserts the chunks cover every contract with no gaps; returns annotated-span coverage. Raises if text is dropped. |
| `encode_split(chunks_df, tokenizer, categories, max_length)` | Tokenizes chunk text and builds the label and mask tensors for one split. |
| `split_labels(value)` | Parses a stored label string into a list of category names. |
| `load_chunker()` | Imports the Week 3 chunker module (`Week 3/tokenizer_chunking_experiment.py`). |

---

## `training/train_distilroberta.py`

Fine-tunes a transformer for multi-label clause classification with masked binary cross-entropy.

### CLI

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `training/data` | Output of `prepare_dataset.py` |
| `--out` | `training/outputs` | Run directory |
| `--model` | `distilroberta-base` | Hugging Face model name |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `3e-5` | Learning rate (AdamW) |
| `--weight-decay` | `0.01` | Weight decay |
| `--warmup-ratio` | `0.1` | Fraction of steps used for linear warmup |
| `--threshold` | `0.5` | Decision threshold for metrics written during training |
| `--val-frac` | `0.1` | Fraction of training contracts held out for validation (split by contract) |
| `--seed` | `42` | Random seed |
| `--pos-weight-mode` | `none` | Positive-class weighting: `none`, `raw` (neg/pos), or `sqrt` (√(neg/pos)) |
| `--pos-weight-cap` | `30.0` | Upper bound on any category's positive weight |
| `--use-pos-weight` | off | Deprecated alias for `--pos-weight-mode raw` |
| `--limit` | none | Use only the first N chunks (smoke test) |

### Outputs

`best_model/` (model and tokenizer from the best validation macro-F1 epoch) · `predictions_val.npz`, `predictions_test.npz` · `history.csv` · `chunk_level_metrics.csv` · `contract_level_metrics.csv`

### Functions

| Function | Description |
|---|---|
| `masked_bce(logits, labels, label_mask, pos_weight=None)` | Binary cross-entropy averaged over unmasked labels only, with optional per-category positive weights. |
| `predict(model, loader, device, use_amp)` | Runs inference and returns sigmoid probabilities, labels, and masks. |
| `per_category_metrics(probs, labels, masks, categories, threshold)` | Precision, recall, F1, and support per category, ignoring masked entries. |
| `micro_f1(probs, labels, masks, threshold)` | Micro-averaged F1 over all unmasked labels. |
| `contract_level(probs, labels, masks, titles)` | Aggregates chunks to contracts: score = max chunk probability, label = any chunk contains the clause. |
| `make_collate(tokenizer)` | Returns a collate function that pads a batch dynamically. |
| `set_seed(seed)` | Seeds Python, NumPy, and PyTorch. |

### Loading a trained model

```python
import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

run = "training/outputs/sqrt_weighted/best_model"
tokenizer = AutoTokenizer.from_pretrained(run)
model = AutoModelForSequenceClassification.from_pretrained(run).eval()
categories = json.load(open("training/data/labels.json"))

text = "Neither party may assign this Agreement without the prior written consent of the other party."
with torch.no_grad():
    probs = torch.sigmoid(model(**tokenizer(text, return_tensors="pt", truncation=True)).logits)[0]
print(sorted(zip(categories, probs.tolist()), key=lambda x: -x[1])[:3])
```

---

## `training/evaluate.py`

Tunes per-category thresholds on validation data, evaluates on test data, compares with TF-IDF, and exports an error analysis.

### CLI

| Option | Default | Description |
|---|---|---|
| `--runs` | required | One or more run directories from `train_distilroberta.py` |
| `--data-dir` | `training/data` | Output of `prepare_dataset.py` |
| `--out` | `training/outputs/evaluation` | Output directory |
| `--min-val-positives` | `5` | Categories with fewer validation positives keep threshold 0.5 |
| `--top-n-errors` | `5` | False positives and false negatives exported per category |

### Outputs

`thresholds_<run>.csv` · `per_category_<run>.csv` · `run_summary.csv` · `tfidf_comparison.csv` · `error_analysis_<run>.csv`

### Functions

| Function | Description |
|---|---|
| `tune_thresholds(val, categories, min_pos)` | Picks the F1-maximizing threshold in 0.05–0.95 per category, separately at chunk and contract level. |
| `metrics(probs, labels, masks, categories, thresholds)` | Per-category precision, recall, and F1 at the given thresholds. |
| `micro(probs, labels, masks, thresholds, cols)` | Micro F1 over a subset of categories. |
| `macro(df, names=None)` | Macro F1, skipping categories with no test positives. |
| `error_analysis(test, thresholds, categories, chunks, top_n)` | Highest-confidence false positives and false negatives per category, with chunk text. |
| `f1_at(y_true, scores, t)` | F1 at threshold `t`. |
| `load(run, split)` | Loads `predictions_<split>.npz` from a run directory. |

---

## `risk/score_contracts.py`

Turns detected clauses into an explainable risk score and a Low / Medium / High bucket for each contract.

### CLI

| Option | Default | Description |
|---|---|---|
| `--source` | required | `gold` (CUAD annotations, confidence 1) or `model` (saved predictions) |
| `--run` | none | Run directory; required when `--source model` |
| `--thresholds` | `training/outputs/evaluation/thresholds_<run>.csv` | Tuned thresholds file |
| `--threshold-scheme` | `tuned` | `tuned` or `0.5` |
| `--chunks` | `training/data/test_chunks.csv` | Chunk text, used for evidence and evidence rules |
| `--train` | `data/cuad/train_separate_questions.json` | Training contracts, used to set bucket cutoffs |
| `--test` | `data/cuad/test.json` | Contracts to score (gold source) |
| `--rubric` | `evaluation/risk_categories.csv` | Category risk levels |
| `--rules` | `risk/risk_rules.json` | Weights, floors, combinations, buckets, evidence rules |
| `--out` | `risk/outputs/<source>` | Output directory |

### Outputs

`contract_scores.csv` · `clause_register.csv` · `sensitivity.csv` · `model_vs_gold.csv` and `evidence_rules_check.csv` (model source only)

### Functions

| Function | Returns | Description |
|---|---|---|
| `load_gold(path)` | `{title: {category: entry}}` | Detected clauses from CUAD annotations. `entry` has `confidence`, `evidence`, `source`. |
| `load_model(run, thresholds_path, chunks_path, scheme, evidence_rules)` | `{title: {category: entry}}` | Detected clauses from model predictions, after thresholds and evidence rules. |
| `score_contract(found, levels, rules)` | `(score, reasons, severity_floor, register_rows)` | Scores one contract with the four signals. |
| `score_all(contracts, levels, rules)` | `(scores_df, register_df)` | Scores many contracts. |
| `run(contracts, train_gold, levels, rules)` | `(scores_df, register_df, (medium_cut, high_cut))` | Scores contracts and assigns buckets using cutoffs from training contracts. |
| `cutoffs(train_scores, rules)` | `(medium, high)` | Percentile cutoffs from training scores. |
| `assign_buckets(df, cuts, rules)` | `pd.Series` | Bucket per contract, applying the severity floor. |
| `sensitivity(contracts, train_gold, levels, rules, base)` | `pd.DataFrame` | Bucket changes and rank correlation under perturbed cutoffs and weights. |
| `norm(name)` | `str` | Lowercases and strips spaces, for matching category names. |

### Python example

```python
import json
import pandas as pd
from risk import score_contracts as sc

rubric = pd.read_csv("evaluation/risk_categories.csv")
levels = {sc.norm(c): (lvl, c) for c, lvl in zip(rubric["category"], rubric["initial_risk"])}
rules = json.load(open("risk/risk_rules.json"))

train = sc.load_gold("data/cuad/train_separate_questions.json")
sample = sc.load_gold("data/sample/sample_contracts.json")
scores, register, cuts = sc.run(sample, train, levels, rules)
print(scores.sort_values("score", ascending=False)[["contract_title", "risk_bucket", "score"]])
```

---

## Data formats

**CUAD JSON** (`data/cuad/*.json`, `data/sample/sample_contracts.json`): SQuAD v2 style.

```text
{"data": [{"title": str,
           "paragraphs": [{"context": str,            # full contract text
                           "qas": [{"id": str,
                                    "question": str,   # names the category in quotes
                                    "answers": [{"text": str, "answer_start": int}],
                                    "is_impossible": bool}]}]}]}
```

**Risk rubric** (`evaluation/risk_categories.csv`): `category`, `initial_risk` (`Low` / `Medium` / `High`), `justification`.

**Risk rules** (`risk/risk_rules.json`): see [User Guide §6](USER_GUIDE.md#6-customizing-risk-rules).
