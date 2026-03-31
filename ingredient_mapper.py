"""
Ingredient mapping between NPPA medicine names and FDA products.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd


SALT_TOKENS = {
    "hcl",
    "hydrochloride",
    "sodium",
    "potassium",
    "calcium",
    "phosphate",
    "sulphate",
    "sulfate",
    "acetate",
    "nitrate",
    "maleate",
    "tartrate",
    "citrate",
}


def normalize_ingredient_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s;,+/-]", " ", text)
    text = text.replace(";", " ").replace("+", " ").replace("/", " ")
    tokens = [token for token in text.split() if token not in SALT_TOKENS]
    return " ".join(tokens).strip()


def extract_ingredient_from_medicine_name(medicine_name: str) -> str:
    name = str(medicine_name)
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\b\d+\s?(mg|mcg|g|ml|iu|%)\b", " ", name, flags=re.IGNORECASE)
    return normalize_ingredient_text(name)


def _best_fuzzy_match(query: str, candidates: pd.Series) -> tuple[str | None, float]:
    if not query:
        return None, 0.0
    best_name = None
    best_score = 0.0
    for candidate in candidates.dropna().unique():
        score = SequenceMatcher(None, query, str(candidate)).ratio()
        if score > best_score:
            best_score = score
            best_name = str(candidate)
    return best_name, best_score


def _resolve_product_match(
    extracted: str,
    exact_lookup: pd.DataFrame,
    products: pd.DataFrame,
    candidates: pd.Series,
    fuzzy_threshold: float,
) -> tuple[object | None, str | None, str, float]:
    if extracted in exact_lookup.index:
        mapped = exact_lookup.loc[extracted]
        return mapped, extracted, "exact", 1.0

    matched_ingredient, score = _best_fuzzy_match(extracted, candidates)
    if matched_ingredient and score >= fuzzy_threshold:
        mapped = products[products["normalized_ingredient"] == matched_ingredient].iloc[0]
        return mapped, matched_ingredient, "fuzzy", float(score)

    return None, None, "none", 0.0


def map_nppa_to_products(
    nppa_df: pd.DataFrame,
    products_df: pd.DataFrame,
    fuzzy_threshold: float = 0.78,
) -> pd.DataFrame:
    products = products_df.copy()
    nppa = nppa_df.copy()

    products["normalized_ingredient"] = products["ingredient"].map(normalize_ingredient_text)
    nppa["extracted_ingredient"] = nppa["medicine_name"].map(extract_ingredient_from_medicine_name)

    exact_lookup = (
        products.sort_values("appl_no")
        .dropna(subset=["normalized_ingredient"])
        .drop_duplicates(subset=["normalized_ingredient"])
        .set_index("normalized_ingredient")
    )

    rows = []
    candidates = products["normalized_ingredient"]
    for _, record in nppa.iterrows():
        extracted = record["extracted_ingredient"]
        mapped, matched_ingredient, match_type, match_score = _resolve_product_match(
            extracted=extracted,
            exact_lookup=exact_lookup,
            products=products,
            candidates=candidates,
            fuzzy_threshold=fuzzy_threshold,
        )

        rows.append(
            {
                "nppa_id": record.get("id"),
                "nppa_year": record.get("year"),
                "medicine_name": record.get("medicine_name"),
                "price_numeric": record.get("price_numeric"),
                "extracted_ingredient": extracted,
                "matched_ingredient": matched_ingredient,
                "match_type": match_type,
                "match_score": match_score,
                "appl_no": mapped.get("appl_no") if mapped is not None else pd.NA,
                "approval_year": mapped.get("approval_year") if mapped is not None else pd.NA,
                "manufacturer": mapped.get("applicant_full_name") if mapped is not None else pd.NA,
                "trade_name": mapped.get("trade_name") if mapped is not None else pd.NA,
            }
        )

    return pd.DataFrame(rows)
