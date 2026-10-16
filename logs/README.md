# Run logs

SLURM job output from the run reported in the main README (Northeastern Explorer, October 2026).
Full metrics for every model are in `runs/duration/results.json` and `runs/tip/results.json`.

| Log | Job | What it contains |
|---|---|---|
| `setup_11134875.log` | `scripts/slurm_setup.sh` | Environment install, January and February 2024 data download, `pytest` (4 passed) |
| `run_11134876.log` | `scripts/slurm_run.sh` | Both tasks: per-model validation and test metrics, MLP training epochs, fit times, summary tables |
| `run_11139812.log` | `scripts/slurm_run.sh` | Rerun that also saves test-month predictions for the figures; every metric matches the first run to within 3e-7 |
