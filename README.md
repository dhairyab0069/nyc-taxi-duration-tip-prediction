# NYC Taxi: Trip Duration Regression and Tip Classification

Two supervised learning tasks on real NYC TLC yellow taxi records, each solved with
a ladder of models from a simple baseline to gradient boosting and a neural network.

| Task | Type | Target | Main metric |
|------|------|--------|-------------|
| `duration` | Regression | Trip duration in minutes | MAE (minutes) |
| `tip` | Classification | Whether a card-paid trip is tipped | AUROC |

## Questions

1. How much do gradient boosting and a neural network gain over a strong group-median baseline?
2. How well do models trained on one month hold up on the next month (temporal shift)?
3. Which features drive each prediction?

## Models

| Name | What it is |
|------|------------|
| `baseline` | Median duration per route (fallback: pickup zone, then global). Tip rate per pickup zone. |
| `linear` | Ridge / logistic regression on scaled numeric and one-hot categorical features |
| `hgb` | scikit-learn histogram gradient boosting with cross-fitted target encoding for zones and routes |
| `lightgbm` | LightGBM with native categorical zones and early stopping |
| `mlp` | PyTorch MLP with learned embeddings for zones, hour, weekday, rate code, vendor |

## Protocol

- Data: January 2024 for training and validation (90/10 split), February 2024 for testing
- Cleaning: drops trips under 1 minute or over 3 hours, zero or over 100 miles, non-positive fares, zone IDs outside 1–265, and records stamped outside the month
- Duration target is modeled as log1p(minutes) and evaluated in minutes
- Tip task uses card payments only, since cash tips are not recorded
- Leakage checks: the duration task never sees fare or duration; the tip task never sees tip or total amount (enforced in tests)

## Setup and run

```bash
pip install -e ".[dev]"
pytest
bash scripts/download_data.sh
python scripts/run.py --task duration
python scripts/run.py --task tip
```

Override config values from the command line:

```bash
python scripts/run.py --task tip --set models=[hgb,lightgbm] data.n_train=300000
```

On SLURM: `sbatch scripts/slurm_setup.sh` once (environment, data, tests), then `sbatch scripts/slurm_run.sh`. Results go to `runs/<task>/results.json`.

## Results

**Duration (test month, MAE in minutes)**

| Model | Val MAE | Test MAE | R² |
|-------|---------|----------|----|
| baseline | 4.06 | 4.24 | 0.652 |
| linear | 3.38 | 3.52 | 0.766 |
| hgb | 2.75 | 2.91 | 0.832 |
| lightgbm | 2.61 | 2.77 | 0.841 |
| mlp | 2.88 | 3.11 | 0.805 |

**Tip (test month)**

| Model | Val AUROC | Test AUROC | Avg precision |
|-------|-----------|------------|---------------|
| baseline | 0.686 | 0.682 | 0.972 |
| linear | 0.751 | 0.745 | 0.978 |
| hgb | 0.755 | 0.747 | 0.979 |
| lightgbm | 0.752 | 0.747 | 0.978 |
| mlp | 0.744 | 0.735 | 0.977 |

Single run, seed 0. About 95% of card trips are tipped, so average precision mostly reflects the base rate; AUROC is the main tip metric.

## Limitations

- The duration task uses the recorded trip distance, which a real ETA system would replace with planned route distance.
- One train month and one test month; more months would show whether the shift is seasonal.

## Structure

```
configs/default.yaml      data months, sample sizes, model hyperparameters
src/taxi/data.py          loading and cleaning
src/taxi/features.py      feature engineering and leakage rules
src/taxi/models.py        baseline, linear, HGB, LightGBM behind one interface
src/taxi/mlp.py           PyTorch MLP with embeddings
src/taxi/metrics.py       regression and classification metrics
scripts/run.py            train all models, evaluate on val and next month
tests/                    cleaning, leakage, and model sanity checks on synthetic data
logs/                     SLURM logs from the reported run (see logs/README.md)
runs/                     results.json for each task
```
