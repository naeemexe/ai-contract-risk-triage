# Sample Dataset

A small slice of CUAD for trying the project without the full 125 MB dataset: **10 contracts** from the official CUAD **test** split (8K–29K characters each, covering sponsorship, distribution, licensing, consulting, co-branding, and non-compete agreements).

## Files

| File | Format | Description |
|---|---|---|
| `sample_contracts.json` | CUAD / SQuAD v2 JSON | Full contract text plus 41 questions (one per clause category) with expert answer spans. Drop-in replacement for `data/cuad/test.json`. |
| `sample_clause_labels.csv` | CSV, 410 rows | One row per (contract, category). |

### `sample_clause_labels.csv` columns

| Column | Type | Meaning |
|---|---|---|
| `contract_title` | string | CUAD contract identifier |
| `category` | string | One of the 41 CUAD clause categories |
| `present` | 0/1 | 1 if experts annotated at least one span for this category |
| `answer_count` | int | Number of annotated spans |
| `first_answer_text` | string | First annotated span, truncated to 300 characters |

## How to use it

```bash
# Risk-score the sample (bucket cutoffs still come from the 408 training contracts)
python risk/score_contracts.py --source gold --test data/sample/sample_contracts.json --out risk/outputs/sample
```

```python
import json, pandas as pd
contracts = json.load(open("data/sample/sample_contracts.json"))["data"]
labels = pd.read_csv("data/sample/sample_clause_labels.csv")
print(labels.pivot_table(index="contract_title", columns="category", values="present").sum())
```

The walkthrough in [`notebooks/quickstart_demo.ipynb`](../../notebooks/quickstart_demo.ipynb) uses this data.

## Source and license

CUAD v1, The Atticus Project, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The contracts are public SEC filings.
