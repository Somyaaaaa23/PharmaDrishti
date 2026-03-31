"""
Feature Fusion Layer — PharmaDrishti

Combines persona features, medicine features, suitability score (as auxiliary
signal), and knowledge-graph features into a single structured feature vector
ready for XGBoost training.

Key design decisions
--------------------
* suitability_score is kept as an INPUT feature, not the target.
* Highly correlated features (|r| > 0.95 with any other feature) are dropped
  to prevent multicollinearity and reduce overfitting.
* Temporal features (approval_age, patent/exclusivity decay) are added here.
* All numeric columns are robust-scaled (median / IQR) before returning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Column groups ─────────────────────────────────────────────────────────────

# Columns that identify rows — never used as features
IDENTIFIER_COLS = {"persona_id", "medicine_id", "drug_node_id"}

# The new target — must NOT appear in X
TARGET_COLS = {"adoption_score"}

# Old target — now an input feature (auxiliary signal)
AUXILIARY_SIGNAL_COLS = {"suitability_score"}

# Columns that directly reconstruct the suitability formula → leakage risk
# (they were the formula inputs, not independent signals)
# NOTE: price, log_price, ratio columns are kept — they are valid independent
# signals for adoption_score and are no longer leaky with the new target.
LEAKAGE_COLS = {
    "disease_match",
    "affordability",
    "manufacturing_match",
    "risk_score",
    "market_pricing_score",
    "manufacturing",
    "side_effect_risk",
}

# Outlier flag columns added by outlier_detector — not features
OUTLIER_SUFFIX = "_outlier"


def _drop_high_correlation(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove one column from each pair with |Pearson r| > threshold.
    Samples up to 50k rows for speed on large datasets."""
    sample = df.sample(min(len(df), 50_000), random_state=42) if len(df) > 50_000 else df
    numeric = sample.select_dtypes(include=[np.number])
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop, errors="ignore")


def _robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Median / IQR robust scaling, clipped to [-8, 8]. Uses numpy for speed on large datasets."""
    out = df.copy()
    arr = out.select_dtypes(include=[np.number]).values.astype(float)
    # Clip at 1st/99th percentile using a sample for speed
    sample_idx = np.random.default_rng(42).choice(len(arr), min(len(arr), 100_000), replace=False)
    sample = arr[sample_idx]
    p01 = np.nanpercentile(sample, 1, axis=0)
    p99 = np.nanpercentile(sample, 99, axis=0)
    arr = np.clip(arr, p01, p99)
    median = np.nanmedian(arr[sample_idx], axis=0)
    q25 = np.nanpercentile(arr[sample_idx], 25, axis=0)
    q75 = np.nanpercentile(arr[sample_idx], 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    arr = np.clip((arr - median) / scale, -8.0, 8.0)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = arr
    return out


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal decay features from patent / exclusivity remaining years."""
    PATENT_HALF_LIFE      = 10.0
    EXCLUSIVITY_HALF_LIFE = 5.0

    if "patent_remaining_years_q50" in df.columns:
        yrs = pd.to_numeric(df["patent_remaining_years_q50"], errors="coerce").fillna(0.0)
        df["patent_decay_factor"] = np.exp(-yrs / PATENT_HALF_LIFE)

    if "exclusivity_remaining_years_q50" in df.columns:
        yrs = pd.to_numeric(df["exclusivity_remaining_years_q50"], errors="coerce").fillna(0.0)
        df["exclusivity_decay_factor"] = np.exp(-yrs / EXCLUSIVITY_HALF_LIFE)

    if "approval_age_q50" in df.columns:
        age = pd.to_numeric(df["approval_age_q50"], errors="coerce").fillna(5.0)
        df["market_maturity"] = 1.0 - np.exp(-age / 5.0)

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-feature interaction terms."""
    if {"suitability_score", "competitor_count_q50"}.issubset(df.columns):
        comp = pd.to_numeric(df["competitor_count_q50"], errors="coerce").fillna(3.0)
        df["suit_x_competition"] = df["suitability_score"] * np.exp(-comp / 10.0)

    if {"suitability_score", "patent_decay_factor"}.issubset(df.columns):
        df["suit_x_patent_protection"] = df["suitability_score"] * (1.0 - df["patent_decay_factor"])

    if {"income", "competitor_count_q50"}.issubset(df.columns):
        comp = pd.to_numeric(df["competitor_count_q50"], errors="coerce").fillna(3.0)
        df["income_competition_pressure"] = np.log1p(
            pd.to_numeric(df["income"], errors="coerce").fillna(0.0)
        ) / (1.0 + comp)

    return df


def build_fused_feature_matrix(
    df: pd.DataFrame,
    drop_high_corr: bool = True,
    corr_threshold: float = 0.95,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Build the final fused feature matrix X and target y.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset containing persona features, medicine features,
        suitability_score, KG features, and adoption_score (if available).
    drop_high_corr : bool
        Whether to remove highly correlated feature pairs.
    corr_threshold : float
        Correlation threshold for dropping features.
    scale : bool
        Whether to apply robust scaling.

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series | None  — adoption_score target (None if not in df)
    """
    work = df.copy()

    # ── Extract target before any transformations ─────────────────────────────
    y = None
    if "adoption_score" in work.columns:
        y = work["adoption_score"].copy()

    # ── Add engineered features ───────────────────────────────────────────────
    work = _add_temporal_features(work)
    work = _add_interaction_features(work)

    # ── Determine columns to keep ─────────────────────────────────────────────
    exclude = (
        IDENTIFIER_COLS
        | TARGET_COLS
        | LEAKAGE_COLS
        | {c for c in work.columns if c.endswith(OUTLIER_SUFFIX) or c == "any_outlier"}
    )

    feature_cols = [
        c for c in work.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(work[c])
    ]

    X = work[feature_cols].copy()

    # ── One-hot encode low-cardinality ordinal columns ────────────────────────
    # Skip one-hot for large datasets (>500k rows) — use ordinal directly for speed
    cat_cols = [
        c for c in [
            "city_tier_encoded",
            "persona_disease_encoded",
            "medicine_disease_encoded",
            "income_level_encoded",
            "digital_adoption_encoded",
            "gender_encoded",
        ]
        if c in X.columns
    ]
    if cat_cols and len(X) <= 500_000:
        dummies = pd.get_dummies(
            X[cat_cols].astype("int64").astype("category"),
            prefix=cat_cols,
            dtype=float,
        )
        X = pd.concat([X, dummies], axis=1)

    # ── Soft disease similarity ───────────────────────────────────────────────
    if {"persona_disease_encoded", "medicine_disease_encoded"}.issubset(X.columns):
        dist = (X["persona_disease_encoded"] - X["medicine_disease_encoded"]).abs()
        X["disease_similarity_soft"] = np.exp(-dist / 4.0)

    if {"chronic_condition_count", "disease_similarity_soft"}.issubset(X.columns):
        X["burden_x_similarity"] = (
            np.sqrt(X["chronic_condition_count"].clip(lower=0.0))
            * X["disease_similarity_soft"]
        )

    # ── Healthcare engagement proxy ───────────────────────────────────────────
    if {"last_doctor_visit_months", "digital_adoption_encoded"}.issubset(X.columns):
        recency = np.exp(-X["last_doctor_visit_months"] / 12.0)
        X["healthcare_engagement"] = (recency * (1 + X["digital_adoption_encoded"] / 3.0)).clip(0.0, 2.0)

    if {"monthly_healthcare_spend", "income"}.issubset(X.columns):
        X["health_spend_pressure"] = (
            X["monthly_healthcare_spend"] / np.maximum(X["income"] / 12.0, 1.0)
        ).clip(0.0, 2.0)

    if {"insurance_sum_assured", "income", "chronic_condition_count"}.issubset(X.columns):
        X["insurance_adequacy"] = (
            X["insurance_sum_assured"]
            / np.maximum(X["income"] * X["chronic_condition_count"].clip(lower=1), 1.0)
        ).clip(0.0, 10.0)

    # ── Polynomial / nonlinear features on key signals ────────────────────────
    if "suitability_score" in X.columns:
        X["suitability_sq"]   = X["suitability_score"] ** 2
        X["suitability_sqrt"] = np.sqrt(X["suitability_score"].clip(lower=0.0))

    if "price_to_income_ratio" in X.columns:
        X["log_price_to_income"] = np.log1p(X["price_to_income_ratio"].clip(lower=0.0))

    if {"suitability_score", "price_to_income_ratio"}.issubset(X.columns):
        X["suit_x_affordability"] = X["suitability_score"] * (
            1.0 / (1.0 + X["price_to_income_ratio"].clip(lower=0.0) * 40.0)
        )

    if {"suitability_score", "city_tier_encoded"}.issubset(X.columns):
        tier_access = X["city_tier_encoded"].map({1: 1.0, 2: 0.78, 3: 0.56}).fillna(0.67)
        X["suit_x_access"] = X["suitability_score"] * tier_access

    if {"chronic_condition_count", "suitability_score"}.issubset(X.columns):
        X["burden_x_suit"] = (
            (X["chronic_condition_count"].clip(lower=0.0) / 5.0).clip(upper=1.0)
            * X["suitability_score"]
        )

    # ── Fill NaN ──────────────────────────────────────────────────────────────
    X = X.fillna(0.0)

    # ── Drop zero-variance and NaN-correlation columns ────────────────────────
    # Remove columns with zero variance (constants after encoding)
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]

    # ── Drop highly correlated features ──────────────────────────────────────
    if drop_high_corr and len(X) > 1:
        X = _drop_high_correlation(X, threshold=corr_threshold)

    # ── Robust scaling ────────────────────────────────────────────────────────
    if scale:
        X = _robust_scale(X)

    return X, y
