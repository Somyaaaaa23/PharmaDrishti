"""
Decision Layer — PharmaDrishti

Post-model adjustment layer that refines raw XGBoost predictions using
domain-specific factors that are better handled as explicit rules than
learned patterns:

  final_score = model_prediction
              × competition_penalty(competitor_count)
              × patent_factor(patent_remaining_years)
              × pricing_sensitivity(price_to_income_ratio)

All adjustment factors are in (0, 1], so the final score is always ≤ the
raw model prediction.  This makes the layer conservative and interpretable.

Confidence Intervals
--------------------
Bootstrap-based 90% confidence intervals are computed by adding calibrated
noise to the raw prediction and re-applying the decision layer N times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Tunable constants ─────────────────────────────────────────────────────────
COMPETITION_SATURATION = 10.0   # competitor count at which penalty saturates
PATENT_HALF_LIFE       = 10.0   # years — patent protection half-life
PRICING_THRESHOLD      = 0.20   # price/income ratio at which penalty = 0.5 (raised from 0.05)
PRICING_STEEPNESS      = 15.0   # sigmoid steepness (reduced from 40 — less aggressive)
N_BOOTSTRAP            = 200    # bootstrap samples for confidence intervals
BOOTSTRAP_NOISE_STD    = 0.03   # noise std for bootstrap sampling


@dataclass
class DecisionLayerResult:
    """Container for decision layer outputs."""
    final_score: np.ndarray
    competition_penalty: np.ndarray
    patent_factor: np.ndarray
    pricing_penalty: np.ndarray
    ci_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_upper: np.ndarray = field(default_factory=lambda: np.array([]))


def _competition_penalty(competitor_count: np.ndarray) -> np.ndarray:
    """
    Exponential decay penalty based on number of competitors.
    0 competitors → 1.0 (no penalty)
    COMPETITION_SATURATION competitors → ~0.05
    """
    k = -np.log(0.05) / COMPETITION_SATURATION
    return np.exp(-k * np.clip(competitor_count, 0.0, None))


def _patent_factor(patent_remaining_years: np.ndarray) -> np.ndarray:
    """
    Regulatory protection boost.
    0 years remaining → 0.5 (no protection, moderate penalty)
    Many years remaining → approaches 1.0 (full protection, no penalty)
    """
    decay = np.exp(-np.clip(patent_remaining_years, 0.0, None) / PATENT_HALF_LIFE)
    return 0.5 + 0.5 * (1.0 - decay)


def _pricing_sensitivity(price_to_income_ratio: np.ndarray) -> np.ndarray:
    """
    Sigmoid pricing penalty.
    Low ratio (affordable) → 1.0
    High ratio (expensive) → approaches 0.0
    """
    return 1.0 / (1.0 + np.exp(
        PRICING_STEEPNESS * (np.clip(price_to_income_ratio, 0.0, None) - PRICING_THRESHOLD)
    ))


def apply_decision_layer(
    raw_predictions: np.ndarray | pd.Series,
    competitor_count: np.ndarray | pd.Series | float = 3.0,
    patent_remaining_years: np.ndarray | pd.Series | float = 0.0,
    price_to_income_ratio: np.ndarray | pd.Series | float = 0.05,
    compute_ci: bool = False,
    rng: np.random.Generator | None = None,
) -> DecisionLayerResult:
    """
    Apply post-model decision layer adjustments.

    Parameters
    ----------
    raw_predictions : array-like
        Raw XGBoost model predictions in [0, 1].
    competitor_count : array-like or scalar
        Number of competing products in the market.
    patent_remaining_years : array-like or scalar
        Years of patent protection remaining.
    price_to_income_ratio : array-like or scalar
        Medicine batch price / annual income.
    compute_ci : bool
        Whether to compute bootstrap confidence intervals.
    rng : np.random.Generator, optional
        For reproducible bootstrap sampling.

    Returns
    -------
    DecisionLayerResult
    """
    if rng is None:
        rng = np.random.default_rng(42)

    preds = np.asarray(raw_predictions, dtype=float).ravel()
    n = len(preds)

    def _to_array(x: np.ndarray | pd.Series | float) -> np.ndarray:
        if np.isscalar(x):
            return np.full(n, float(x))
        return np.asarray(x, dtype=float).ravel()

    comp  = _to_array(competitor_count)
    pat   = _to_array(patent_remaining_years)
    pti   = _to_array(price_to_income_ratio)

    comp_pen  = _competition_penalty(comp)
    pat_fac   = _patent_factor(pat)
    price_pen = _pricing_sensitivity(pti)

    final = np.clip(preds * comp_pen * pat_fac * price_pen, 0.0, 1.0)

    ci_lower = np.array([])
    ci_upper = np.array([])

    if compute_ci:
        bootstrap_finals = np.zeros((N_BOOTSTRAP, n))
        for i in range(N_BOOTSTRAP):
            noisy = np.clip(preds + rng.normal(0.0, BOOTSTRAP_NOISE_STD, size=n), 0.0, 1.0)
            bootstrap_finals[i] = np.clip(noisy * comp_pen * pat_fac * price_pen, 0.0, 1.0)
        ci_lower = np.percentile(bootstrap_finals, 5,  axis=0)
        ci_upper = np.percentile(bootstrap_finals, 95, axis=0)

    return DecisionLayerResult(
        final_score=final,
        competition_penalty=comp_pen,
        patent_factor=pat_fac,
        pricing_penalty=price_pen,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def extract_decision_inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Extract the three decision-layer inputs from a DataFrame.
    Falls back to sensible defaults when columns are absent.

    Note: when KG columns are absent (baseline mode), competition and
    patent factors default to neutral (1.0 effect) so the decision layer
    acts as a pass-through and does not distort predictions.
    """
    def _col(name: str, default: float) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy()
        return np.full(len(df), default)

    # Use neutral defaults when KG data is absent:
    #   competitor_count = 0  → competition_penalty = 1.0 (no penalty)
    #   patent_remaining = 20 → patent_factor ≈ 1.0 (full protection)
    has_kg = "competitor_count_q50" in df.columns
    comp_default  = 3.0 if has_kg else 0.0
    patent_default = 0.0 if has_kg else 20.0

    return {
        "competitor_count":       _col("competitor_count_q50", comp_default),
        "patent_remaining_years": _col("patent_remaining_years_q50", patent_default),
        "price_to_income_ratio":  _col("price_to_income_ratio", 0.05),
    }
