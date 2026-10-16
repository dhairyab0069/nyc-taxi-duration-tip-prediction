"""Loading and cleaning NYC TLC yellow taxi trip records."""
import os

import numpy as np
import pandas as pd

COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "RatecodeID", "PULocationID", "DOLocationID",
    "payment_type", "fare_amount", "tip_amount",
]


def month_path(data_dir, month):
    return os.path.join(data_dir, f"yellow_tripdata_{month}.parquet")


def clean(df, month=None):
    """Drop implausible trips and add derived targets."""
    df = df.copy()
    df["duration_min"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    keep = (
        df.duration_min.between(1, 180)
        & (df.trip_distance > 0) & (df.trip_distance <= 100)
        & (df.fare_amount > 0)
        & df.PULocationID.between(1, 265) & df.DOLocationID.between(1, 265)
    )
    if month is not None:  # raw files contain a few trips stamped in other months
        year, mon = map(int, month.split("-"))
        keep &= (df.tpep_pickup_datetime.dt.year == year) & (df.tpep_pickup_datetime.dt.month == mon)
    df = df[keep].copy()
    df["passenger_count"] = df.passenger_count.fillna(1).clip(0, 6)
    df["RatecodeID"] = df.RatecodeID.fillna(1).clip(1, 6)   # 99 = unknown, folded into 6
    df["VendorID"] = df.VendorID.clip(0, 7)
    df["tipped"] = (df.tip_amount > 0).astype(np.int64)
    return df.reset_index(drop=True)


def load_month(data_dir, month, n=None, seed=0):
    df = pd.read_parquet(month_path(data_dir, month), columns=COLUMNS)
    df = clean(df, month)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df
