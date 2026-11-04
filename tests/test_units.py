import numpy as np
import pandas as pd
import pytest
from fake import fake_trips

from taxi.data import clean
from taxi.features import build_features, task_frame
from taxi.metrics import classification_metrics, evaluate, regression_metrics
from taxi.models import GroupBaseline, build_model

SMALL = {
    "lightgbm": {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15, "early_stopping_rounds": 5},
    "mlp": {"hidden": [16], "dropout": 0.0, "emb_dim": 4, "epochs": 2, "batch_size": 256, "lr": 0.01, "patience": 2},
}


def test_regression_metrics_undo_log_and_are_exact_on_perfect_input():
    minutes = np.array([5.0, 10.0, 20.0, 40.0])
    m = regression_metrics(np.log1p(minutes), np.log1p(minutes))
    assert m["mae_min"] == pytest.approx(0) and m["r2"] == pytest.approx(1)
    off = regression_metrics(np.log1p(minutes), np.log1p(minutes + 2))
    assert off["mae_min"] == pytest.approx(2.0)


def test_classification_metrics_on_known_scores():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auroc"] == 1.0 and m["accuracy"] == 1.0 and m["positive_rate"] == 0.5
    assert classification_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["auroc"] == 0.0


def test_evaluate_dispatches_by_task():
    y = np.log1p(np.array([3.0, 6.0]))
    assert "mae_min" in evaluate("duration", y, y)
    assert "auroc" in evaluate("tip", np.array([0, 1]), np.array([0.2, 0.7]))


def test_features_are_cyclic_and_route_is_unique_per_pair():
    df = clean(fake_trips(500))
    X = build_features(df, "duration")
    assert np.allclose(X.tod_sin**2 + X.tod_cos**2, 1)
    assert (X.route == X.PULocationID * 1000 + X.DOLocationID).all()
    assert X.is_weekend.isin([0, 1]).all() and len(X) == len(df)
    assert "fare_amount" in build_features(df, "tip").columns


def test_duration_target_is_log1p_minutes():
    df = clean(fake_trips(300))
    _, y = task_frame(df, "duration")
    assert np.allclose(np.expm1(y), df.duration_min)


def test_group_baseline_falls_back_from_route_to_zone_to_global():
    X = pd.DataFrame({"PULocationID": [1, 1, 2], "DOLocationID": [5, 6, 7]})
    X["route"] = X.PULocationID * 1000 + X.DOLocationID
    model = GroupBaseline("duration").fit(X, np.array([1.0, 3.0, 10.0]))
    query = pd.DataFrame({"PULocationID": [1, 1, 99], "DOLocationID": [5, 9, 9]})
    query["route"] = query.PULocationID * 1000 + query.DOLocationID
    pred = model.predict(query)
    assert pred[0] == 1.0      # known route
    assert pred[1] == 2.0      # unseen route, known pickup zone (median of 1 and 3)
    assert pred[2] == 3.0      # unseen zone, global median


@pytest.mark.parametrize("name", ["lightgbm", "mlp"])
@pytest.mark.parametrize("task", ["duration", "tip"])
def test_lightgbm_and_mlp_run_end_to_end_on_synthetic_data(name, task):
    pytest.importorskip("lightgbm" if name == "lightgbm" else "torch")
    X, y = task_frame(clean(fake_trips(1500, seed=1)), task)
    Xt, yt = task_frame(clean(fake_trips(500, seed=2)), task)
    pred = build_model(name, task, SMALL).fit(X, y, Xt, yt).predict(Xt)
    assert pred.shape == (len(Xt),) and np.isfinite(pred).all()
    if task == "tip":
        assert ((pred >= 0) & (pred <= 1)).all()
