"""Tests for browser-safe JSON helpers."""

import json
import math

import pandas as pd

from experiments.listening.json_util import json_safe


def test_json_safe_converts_nan_to_null():
    payload = {"pool_id": float("nan"), "program": 0}
    encoded = json.dumps(json_safe(payload), allow_nan=False)
    assert json.loads(encoded) == {"pool_id": None, "program": 0}


def test_json_safe_converts_pandas_na():
    payload = {"pool_id": pd.NA, "items": [pd.NA, "dry"]}
    encoded = json.dumps(json_safe(payload), allow_nan=False)
    assert json.loads(encoded) == {"pool_id": None, "items": [None, "dry"]}


def test_json_safe_preserves_finite_numbers():
    payload = {"init_noise_level": 0.45, "track": 0}
    encoded = json.dumps(json_safe(payload), allow_nan=False)
    assert json.loads(encoded) == payload


def test_json_safe_rejects_inf():
    payload = {"value": math.inf}
    encoded = json.dumps(json_safe(payload), allow_nan=False)
    assert json.loads(encoded) == {"value": None}
