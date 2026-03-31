"""
Outlier detection utilities for KG-enhanced PharmaDrishti modeling.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def detect_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    method: str = "iqr",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    outlier_flags = pd.DataFrame(index=df.index)

    for column in columns:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            outlier_flags[f"{column}_outlier"] = False
            continue

        if method.lower() == "iqr":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_flags[f"{column}_outlier"] = (series < lower) | (series > upper)
        elif method.lower() == "zscore":
            std = valid.std(ddof=0)
            if std == 0:
                outlier_flags[f"{column}_outlier"] = False
            else:
                z_scores = (series - valid.mean()) / std
                outlier_flags[f"{column}_outlier"] = z_scores.abs() > z_threshold
        else:
            raise ValueError("method must be either 'iqr' or 'zscore'")

    outlier_flags["any_outlier"] = outlier_flags.any(axis=1)
    return outlier_flags.fillna(False)


def remove_or_flag_outliers(
    df: pd.DataFrame,
    outlier_flags: pd.DataFrame,
    strategy: str = "flag",
) -> pd.DataFrame:
    result = df.copy()

    if "any_outlier" not in outlier_flags.columns:
        outlier_flags = outlier_flags.copy()
        outlier_flags["any_outlier"] = outlier_flags.any(axis=1)

    if strategy == "remove":
        return result.loc[~outlier_flags["any_outlier"]].copy()

    if strategy == "flag":
        for col in outlier_flags.columns:
            result[col] = outlier_flags[col].astype(bool)
        return result

    raise ValueError("strategy must be either 'remove' or 'flag'")
