import numpy as np
import pandas as pd


def fake_trips(n=3000, seed=0, month="2024-01"):
    """Synthetic trips with a learnable signal, for tests only."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{month}-01")
    pickup = start + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, n), unit="m")
    dist = rng.gamma(2.0, 1.5, n) + 0.1
    hour = pickup.hour.to_numpy()
    rush = ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
    minutes = 3 + dist * (3 + 2 * rush) + rng.normal(0, 2, n).clip(-2, 10)
    fare = 3 + 2.5 * dist
    tipped = rng.random(n) < (0.3 + 0.4 * (fare > 10))
    return pd.DataFrame({
        "VendorID": rng.choice([1, 2], n),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + pd.to_timedelta(minutes, unit="m"),
        "passenger_count": rng.choice([1.0, 2.0, np.nan], n),
        "trip_distance": dist,
        "RatecodeID": rng.choice([1.0, 2.0, 99.0], n),
        "PULocationID": rng.integers(1, 266, n),
        "DOLocationID": rng.integers(1, 266, n),
        "payment_type": rng.choice([1, 2], n, p=[0.75, 0.25]),
        "fare_amount": fare,
        "tip_amount": np.where(tipped, fare * 0.2, 0.0),
    })
