# US Wildfire Size Class Prediction (1992–2020)

Predict which size class (A–G) a US wildfire ends up in, given features
known at discovery time.

## Problem

| | |
|---|---|
| **Target** | `fire_size_class` — ordinal label A (< 0.25 ac) through G (≥ 5000 ac) |
| **Metric** | Weighted multi-class log-loss (inverse-frequency weights; lower = better) |
| **Baseline** | Uniform 1/7 predictions ≈ 1.946 |
| **Challenge** | Classes A + B account for ~86 % of fires; G is only 0.21 % |

## Data

Place `train.csv` and `test.csv` (from the USDA Fire Program Analysis database)
in the project root before running.

| File | Rows | Description |
|---|---|---|
| `train.csv` | 1,842,852 | Features + `fire_size_class` label |
| `test.csv` | 460,714 | Features only — generate predictions for these |

### Feature columns

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique fire ID |
| `fire_year` | int | Year discovered (1992–2020) |
| `discovery_doy` | int | Day of year (1–366) |
| `discovery_month` | int | Month (1–12) |
| `discovery_dow` | int | Day of week (0 = Mon, 6 = Sun) |
| `nwcg_general_cause` | string | Cause bucket (13 categories) |
| `latitude` | float | Fire latitude |
| `longitude` | float | Fire longitude |
| `owner_descr` | string | Land owner (USFS, BLM, STATE, PRIVATE, …) |
| `state` | string | US state (two-letter code) |
| `discovery_time` | string | Time of discovery HHMM (~34 % missing) |

## Solution

`solution.py` implements an end-to-end pipeline:

1. **Feature engineering** — cyclical time encodings, geographic grid cells,
   decade trend, interaction features, and `discovery_time` parsing with a
   missing-indicator flag.
2. **LightGBM** multi-class model with `class_weight="balanced"` to handle
   severe class imbalance.
3. **5-fold stratified cross-validation** — OOF predictions for honest
   evaluation; test predictions averaged across folds.
4. **Submission** — `submission.csv` with columns
   `id, class_A, class_B, class_C, class_D, class_E, class_F, class_G`
   (probabilities summing to 1 per row).

## Quick start

```bash
pip install -r requirements.txt
python solution.py
```

The script prints per-fold and overall OOF weighted log-loss and writes
`submission.csv` when finished.

## Submission format

```
id,class_A,class_B,class_C,class_D,class_E,class_F,class_G
1234,0.45,0.35,0.12,0.04,0.02,0.01,0.01
```
