"""
Shared XGBoost configuration for PharmaDrishti.

Keeps the same model family while adding regularization and early-stopping-ready
defaults to improve generalization.
"""

from __future__ import annotations

from xgboost import XGBRegressor


DEFAULT_XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 8,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "colsample_bylevel": 0.75,
    "min_child_weight": 3,
    "gamma": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "random_state": 42,
    "objective": "reg:squarederror",
    "verbosity": 0,
    "n_jobs": -1,
}


def build_xgb_regressor(**overrides) -> XGBRegressor:
    params = dict(DEFAULT_XGB_PARAMS)
    params.update(overrides)
    return XGBRegressor(**params)
