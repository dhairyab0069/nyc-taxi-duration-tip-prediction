"""Feature engineering shared by every model.

Leakage rules:
  duration task uses only what is known at pickup (plus planned trip distance).
  tip task may also use fare and duration, which are known before the tip,
  but never tip_amount or total_amount.
"""
import numpy as np
import pandas as pd

ZONE = ["PULocationID", "DOLocationID"]
LOW_CAT = ["hour", "dow", "ratecode", "vendor"]
BASE_NUM = ["tod_sin", "tod_cos", "trip_distance", "log_distance",
            "passenger_count", "is_weekend", "same_zone"]
TIP_NUM = ["fare_amount", "duration_min", "fare_per_mile"]


def numeric_cols(task):
    return BASE_NUM + (TIP_NUM if task == "tip" else [])


def build_features(df, task):
    t = df.tpep_pickup_datetime
    minutes = t.dt.hour * 60 + t.dt.minute
    X = pd.DataFrame({
        "hour": t.dt.hour.astype(int),
        "dow": t.dt.dayofweek.astype(int),
        "tod_sin": np.sin(2 * np.pi * minutes / 1440),
        "tod_cos": np.cos(2 * np.pi * minutes / 1440),
        "trip_distance": df.trip_distance,
        "log_distance": np.log1p(df.trip_distance),
        "passenger_count": df.passenger_count,
        "ratecode": df.RatecodeID.astype(int),
        "vendor": df.VendorID.astype(int),
        "PULocationID": df.PULocationID.astype(int),
        "DOLocationID": df.DOLocationID.astype(int),
    })
    X["is_weekend"] = (X.dow >= 5).astype(int)
    X["same_zone"] = (X.PULocationID == X.DOLocationID).astype(int)
    X["route"] = X.PULocationID * 1000 + X.DOLocationID
    if task == "tip":
        X["fare_amount"] = df.fare_amount
        X["duration_min"] = df.duration_min
        X["fare_per_mile"] = df.fare_amount / df.trip_distance
    return X.reset_index(drop=True)


def task_frame(df, task):
    """Return (features, target) for a task.

    duration: target is log1p(minutes), so errors are relative and long trips do not dominate.
    tip: only card payments (payment_type 1), because cash tips are not recorded.
    """
    if task == "tip":
        df = df[df.payment_type == 1]
    X = build_features(df, task)
    if task == "duration":
        y = np.log1p(df.duration_min.to_numpy())
    else:
        y = df.tipped.to_numpy()
    return X, y
