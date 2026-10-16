import numpy as np
import pandas as pd

from taxi.data import clean
from taxi.features import task_frame
from taxi.metrics import evaluate
from taxi.models import build_model
from fake import fake_trips

CFG = {"hgb": {"max_iter": 50, "learning_rate": 0.1, "max_leaf_nodes": 15}}


def test_clean_filters_bad_trips():
    df = fake_trips(500)
    df.loc[0, "trip_distance"] = 0
    df.loc[1, "tpep_dropoff_datetime"] = df.loc[1, "tpep_pickup_datetime"]
    df.loc[2, "tpep_pickup_datetime"] = pd.Timestamp("2023-12-31 23:00")
    out = clean(df, "2024-01")
    assert len(out) <= 497
    assert out.duration_min.between(1, 180).all()
    assert out.RatecodeID.max() <= 6 and out.passenger_count.notna().all()


def test_tip_task_has_no_leakage_and_card_only():
    df = clean(fake_trips(1000))
    X, y = task_frame(df, "tip")
    assert "tip_amount" not in X.columns and "tipped" not in X.columns
    assert len(X) == (df.payment_type == 1).sum()
    assert set(np.unique(y)) <= {0, 1}


def test_duration_task_uses_no_post_trip_info():
    X, _ = task_frame(clean(fake_trips(500)), "duration")
    assert "fare_amount" not in X.columns and "duration_min" not in X.columns


def test_models_beat_baseline_on_signal():
    for task, better in [("duration", lambda a, b: a < b), ("tip", lambda a, b: a > b)]:
        X, y = task_frame(clean(fake_trips(4000, seed=1)), task)
        Xt, yt = task_frame(clean(fake_trips(2000, seed=2)), task)
        key = "mae_min" if task == "duration" else "auroc"
        scores = {}
        for name in ["baseline", "linear", "hgb"]:
            m = build_model(name, task, CFG).fit(X, y, Xt, yt)
            scores[name] = evaluate(task, yt, m.predict(Xt))[key]
        assert better(scores["hgb"], scores["baseline"]), (task, scores)
        assert better(scores["linear"], scores["baseline"]), (task, scores)
