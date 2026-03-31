"""
Regulatory and market data loading utilities for PharmaDrishti.

Loads FDA Orange Book standardized datasets and NPPA market data,
with light schema normalization for downstream mapping and KG steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _coerce_year(value) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return float(text[:4])
    if text.isdigit():
        return float(text)
    return float("nan")


def load_nppa_market_data(base_dir: str | Path = ".") -> pd.DataFrame:
    base = Path(base_dir)
    file_path = base / "nppaipdms.csv"
    df = pd.read_csv(file_path)

    if len(df.columns) >= 6:
        df = df.iloc[:, :6].copy()
        df.columns = [
            "id",
            "year",
            "medicine_name",
            "formulation",
            "notification",
            "price",
        ]

    df["medicine_name"] = df["medicine_name"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["price_numeric"] = (
        df["price"].astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    )
    return df


def load_regulatory_data(base_dir: str | Path = ".") -> Dict[str, pd.DataFrame]:
    base = Path(base_dir) / "standardized_data"

    products = pd.read_csv(base / "products_standardized.csv")
    patents = pd.read_csv(base / "patents_standardized.csv")
    exclusivity = pd.read_csv(base / "exclusivity_standardized.csv")
    reference = pd.read_csv(base / "reference_standardized.csv")

    products = products.copy()
    patents = patents.copy()
    exclusivity = exclusivity.copy()
    reference = reference.copy()

    products["appl_no"] = pd.to_numeric(products.get("appl_no"), errors="coerce")
    products["approval_year"] = products.get("approval_date", pd.Series(dtype=float)).map(
        _coerce_year
    )

    patents["appl_no"] = pd.to_numeric(patents.get("appl_no"), errors="coerce")
    patents["patent_expiry_year"] = patents.get(
        "patent_expire_date_text", pd.Series(dtype=float)
    ).map(_coerce_year)

    exclusivity["appl_no"] = pd.to_numeric(exclusivity.get("appl_no"), errors="coerce")
    exclusivity["exclusivity_expiry_year"] = exclusivity.get(
        "exclusivity_date", pd.Series(dtype=float)
    ).map(_coerce_year)

    return {
        "products": products,
        "patents": patents,
        "exclusivity": exclusivity,
        "reference": reference,
    }
