"""
Training feature selection utilities for PharmaDrishti.

Excludes columns that directly reconstruct the synthetic target formula,
which would otherwise create leakage-like behavior in model training.

Target variable: adoption_score (NOT suitability_score).
suitability_score is kept as an auxiliary input feature.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


IDENTIFIER_COLUMNS = {"persona_id", "medicine_id", "drug_node_id"}
# adoption_score is the new target; suitability_score is now an input feature
TARGET_COLUMNS = {"adoption_score"}
# These columns either ARE the suitability formula components or can be
# algebraically combined to reconstruct them — keeping them gives R² ≈ 1.0.
LEAKAGE_PRONE_COLUMNS = {
    # Direct formula inputs that reconstruct suitability_score
    "disease_match",
    "affordability",
    "manufacturing_match",
    "risk_score",
    "market_pricing_score",
    # Raw sub-components that feed the formula
    "manufacturing",       # used in manufacturing_match directly
    "side_effect_risk",    # used in risk_score directly
    # NOTE: price, log_price, ratio columns are intentionally kept —
    # they are independent signals for adoption_score (not leaky anymore)
}


def select_model_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns: list[str] = []
    for column in df.columns:
        if column in IDENTIFIER_COLUMNS or column in TARGET_COLUMNS:
            continue
        if column in LEAKAGE_PRONE_COLUMNS:
            continue
        if column.endswith("_outlier") or column == "any_outlier":
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        feature_columns.append(column)
    return feature_columns


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = select_model_feature_columns(df)
    matrix = df[feature_columns].copy()

    categorical_like_columns = [
        column
        for column in [
            "city_tier_encoded",
            "persona_disease_encoded",
            "medicine_disease_encoded",
            "income_level_encoded",
            "digital_adoption_encoded",
            "gender_encoded",
        ]
        if column in matrix.columns
    ]
    if categorical_like_columns:
        dummies = pd.get_dummies(
            matrix[categorical_like_columns].astype("int64").astype("category"),
            prefix=categorical_like_columns,
            dtype=float,
        )
        matrix = pd.concat([matrix, dummies], axis=1)

    if {"persona_disease_encoded", "medicine_disease_encoded"}.issubset(matrix.columns):
        disease_distance = (
            matrix["persona_disease_encoded"] - matrix["medicine_disease_encoded"]
        ).abs()
        matrix["disease_similarity_soft"] = np.exp(-disease_distance / 4.0)

    if {"chronic_condition_count", "disease_similarity_soft"}.issubset(matrix.columns):
        matrix["burden_x_similarity"] = (
            np.sqrt(matrix["chronic_condition_count"].clip(lower=0.0))
            * matrix["disease_similarity_soft"]
        )

    if {"insurance_sum_assured", "income", "chronic_condition_count"}.issubset(matrix.columns):
        matrix["insurance_adequacy"] = (
            matrix["insurance_sum_assured"]
            / np.maximum(matrix["income"] * matrix["chronic_condition_count"].clip(lower=1), 1.0)
        ).clip(0.0, 10.0)

    if {"last_doctor_visit_months", "digital_adoption_encoded"}.issubset(matrix.columns):
        recency = np.exp(-matrix["last_doctor_visit_months"] / 12.0)
        matrix["healthcare_engagement"] = (
            recency * (1 + matrix["digital_adoption_encoded"] / 3.0)
        ).clip(0.0, 2.0)

    if {"monthly_healthcare_spend", "income"}.issubset(matrix.columns):
        matrix["health_spend_pressure"] = (
            matrix["monthly_healthcare_spend"] / np.maximum(matrix["income"] / 12.0, 1.0)
        ).clip(0.0, 2.0)

    numeric_columns = matrix.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        values = pd.to_numeric(matrix[column], errors="coerce").fillna(0.0)
        low = values.quantile(0.01)
        high = values.quantile(0.99)
        clipped = values.clip(low, high)
        median = float(clipped.median())
        q1 = float(clipped.quantile(0.25))
        q3 = float(clipped.quantile(0.75))
        scale = max(q3 - q1, 1e-6)
        matrix[column] = ((clipped - median) / scale).clip(-8.0, 8.0)

    # Use adoption_score as target if available, else fall back to suitability_score
    if "adoption_score" in df.columns:
        return matrix, df["adoption_score"].copy()
    return matrix, df["suitability_score"].copy()
