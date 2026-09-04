"""Stable display helpers for Experiment 2 validation diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_diagnostic_value(value: object) -> str:
    """Return a deterministic display string without changing source values."""
    value = _python_scalar(value)
    if value is pd.NA:
        return "pd.NA"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
    return repr(value)
