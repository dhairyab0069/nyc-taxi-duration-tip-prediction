"""Models with one shared interface: fit(Xtr, ytr, Xva, yva) and predict(X).

predict returns log1p(minutes) for regression and P(tipped) for classification.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from taxi.features import LOW_CAT, ZONE, numeric_cols


class GroupBaseline:
    """Regression: median per route, falling back to pickup zone, then global.
    Classification: tip rate per pickup zone, falling back to global."""

    def __init__(self, task):
        self.task = task

    def fit(self, X, y, Xva=None, yva=None):
        s = pd.Series(y, index=X.index)
        if self.task == "duration":
            self.route = s.groupby(X.route).median()
            self.zone = s.groupby(X.PULocationID).median()
            self.glob = float(np.median(y))
        else:
            self.zone = s.groupby(X.PULocationID).mean()
            self.glob = float(np.mean(y))
        return self

    def predict(self, X):
        out = X.PULocationID.map(self.zone)
        if self.task == "duration":
            out = X.route.map(self.route).fillna(out)
        return out.fillna(self.glob).to_numpy()


class LinearModel:
    def __init__(self, task):
        self.task = task

    def fit(self, X, y, Xva=None, yva=None):
        pre = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols(self.task)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), LOW_CAT + ZONE),
        ])
        est = Ridge(alpha=1.0) if self.task == "duration" else LogisticRegression(max_iter=1000)
        self.pipe = Pipeline([("pre", pre), ("est", est)]).fit(X, y)
        return self

    def predict(self, X):
        if self.task == "duration":
            return self.pipe.predict(X)
        return self.pipe.predict_proba(X)[:, 1]


class HGBModel:
    """sklearn histogram gradient boosting, with cross-fitted target encoding
    for high-cardinality zone and route ids."""

    def __init__(self, task, cfg, seed=0):
        self.task, self.cfg, self.seed = task, cfg, seed

    def fit(self, X, y, Xva=None, yva=None):
        target_type = "continuous" if self.task == "duration" else "binary"
        pre = ColumnTransformer([
            ("te", TargetEncoder(target_type=target_type, random_state=self.seed), ZONE + ["route"]),
            ("rest", "passthrough", numeric_cols(self.task) + LOW_CAT),
        ])
        Est = HistGradientBoostingRegressor if self.task == "duration" else HistGradientBoostingClassifier
        est = Est(max_iter=self.cfg["max_iter"], learning_rate=self.cfg["learning_rate"],
                  max_leaf_nodes=self.cfg["max_leaf_nodes"], early_stopping=True,
                  validation_fraction=0.1, random_state=self.seed)
        self.pipe = Pipeline([("pre", pre), ("est", est)]).fit(X, y)
        return self

    def predict(self, X):
        if self.task == "duration":
            return self.pipe.predict(X)
        return self.pipe.predict_proba(X)[:, 1]


class LightGBMModel:
    """LightGBM with native categorical handling for zone ids."""

    def __init__(self, task, cfg, seed=0):
        self.task, self.cfg, self.seed = task, cfg, seed
        self.cols = numeric_cols(task) + LOW_CAT + ZONE

    def _prep(self, X):
        X = X[self.cols].copy()
        for c in ZONE:
            X[c] = pd.Categorical(X[c], categories=range(1, 266))
        return X

    def fit(self, X, y, Xva, yva):
        import lightgbm as lgb
        Est = lgb.LGBMRegressor if self.task == "duration" else lgb.LGBMClassifier
        self.model = Est(n_estimators=self.cfg["n_estimators"], learning_rate=self.cfg["learning_rate"],
                         num_leaves=self.cfg["num_leaves"], random_state=self.seed, verbose=-1)
        self.model.fit(self._prep(X), y, eval_set=[(self._prep(Xva), yva)],
                       callbacks=[lgb.early_stopping(self.cfg["early_stopping_rounds"], verbose=False)])
        return self

    def predict(self, X):
        if self.task == "duration":
            return self.model.predict(self._prep(X))
        return self.model.predict_proba(self._prep(X))[:, 1]

    def feature_importance(self):
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return dict(sorted(zip(self.cols, gain.tolist()), key=lambda kv: -kv[1]))


def build_model(name, task, cfg, seed=0):
    if name == "baseline":
        return GroupBaseline(task)
    if name == "linear":
        return LinearModel(task)
    if name == "hgb":
        return HGBModel(task, cfg["hgb"], seed)
    if name == "lightgbm":
        return LightGBMModel(task, cfg["lightgbm"], seed)
    if name == "mlp":
        from taxi.mlp import MLPModel  # torch is imported only when needed
        return MLPModel(task, cfg["mlp"], seed)
    raise ValueError(f"unknown model {name}")
