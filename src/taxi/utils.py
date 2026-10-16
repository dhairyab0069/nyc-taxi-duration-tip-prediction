import json
import os
import random

import numpy as np
import yaml


def load_config(path, overrides=()):
    """Load YAML and apply dotted overrides like data.n_train=200000."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
