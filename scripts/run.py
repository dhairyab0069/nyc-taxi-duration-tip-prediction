"""Train every model on one month and test on the next.

Examples:
    python scripts/run.py --task duration
    python scripts/run.py --task tip --set models=[hgb,lightgbm]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from taxi import TASKS
from taxi.data import load_month
from taxi.features import task_frame
from taxi.metrics import evaluate
from taxi.models import build_model
from taxi.utils import load_config, save_json, set_seed

KEY_METRIC = {"duration": "mae_min", "tip": "auroc"}
CONTEXT = ["trip_distance", "hour", "dow", "PULocationID", "DOLocationID"]


def save_predictions(path, task, Xte, yte, preds):
    """Test-month predictions next to a few context columns, for plot_results.py.
    Duration is stored in minutes; tip as P(tipped)."""
    to_out = np.expm1 if task == "duration" else (lambda v: v)
    df = Xte[CONTEXT].copy()
    df["y_true"] = to_out(yte)
    for name, p in preds.items():
        df[name] = to_out(p)
    df.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    d = cfg["data"]

    train_df = load_month(d["dir"], d["train_month"], d["n_train"], cfg["seed"])
    test_df = load_month(d["dir"], d["test_month"], d["n_test"], cfg["seed"])
    X, y = task_frame(train_df, args.task)
    Xte, yte = task_frame(test_df, args.task)
    stratify = y if TASKS[args.task] == "classification" else None
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=d["val_frac"], random_state=cfg["seed"],
                                          stratify=stratify)
    Xtr, Xva = Xtr.reset_index(drop=True), Xva.reset_index(drop=True)
    print(f"task {args.task} ({TASKS[args.task]})  train {len(Xtr)}  val {len(Xva)}  "
          f"test {len(Xte)} [{d['test_month']}]")

    results, preds = {}, {}
    for name in cfg["models"]:
        print(f"\n== {name}")
        t0 = time.time()
        model = build_model(name, args.task, cfg, cfg["seed"]).fit(Xtr, ytr, Xva, yva)
        fit_s = time.time() - t0
        preds[name] = model.predict(Xte)
        results[name] = {
            "val": evaluate(args.task, yva, model.predict(Xva)),
            "test": evaluate(args.task, yte, preds[name]),
            "fit_seconds": round(fit_s, 1),
        }
        if hasattr(model, "feature_importance"):
            results[name]["feature_importance_gain"] = model.feature_importance()
        k = KEY_METRIC[args.task]
        print(f"   {k}: val {results[name]['val'][k]:.4f}  test {results[name]['test'][k]:.4f}"
              f"  ({fit_s:.0f}s)")

    k = KEY_METRIC[args.task]
    print(f"\n{'model':10s} {'val ' + k:>16s} {'test ' + k:>16s}")
    for name, r in results.items():
        print(f"{name:10s} {r['val'][k]:16.4f} {r['test'][k]:16.4f}")

    out = os.path.join(cfg["out_dir"], args.task, "results.json")
    save_json({"config": cfg, "results": results}, out)
    save_predictions(os.path.join(cfg["out_dir"], args.task, "test_predictions.parquet"), args.task, Xte, yte, preds)
    print(f"\nsaved {out} and test_predictions.parquet")


if __name__ == "__main__":
    main()
