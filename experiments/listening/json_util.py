"""JSON helpers for browser-safe API responses."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively replace NaN/NA/Inf with ``null`` for strict JSON encoding."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
