"""
PharmaDrishti — Model Benchmark Script
=======================================
Runs three models on 1000 personas × 5000 medicines and reports
R², MAE, RMSE for each:

  1. XGBoost-only      — tabular features, no KG
  2. KG-only           — KG graph features only (no tabular persona/medicine features)
  3. Combined (KG+XGB) — full feature fusion (tabular + KG)

All three share the same train/test split for a fair comparison.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from adoption_score_generator import compute_adoption_score
from data_loader import load_personas
from feature_fusion import build_fused_feature_matrix
from feature_selection import build_model_matrix
from graph_feature_extractor import extract_graph_features, merge_graph_features_with_training_data
from ingredient_mapper import map_nppa_to_products
from knowledge_graph_builder import build_knowledge_graph
from medicine_generator import generate_medicines
from model_config import build_xgb_regressor
from outlier_detector import detect_outliers, remove_or_flag_outliers
from regulatory_data_loader import load_nppa_market_data, load_regulatory_data
from suitability_calculator import create_interaction_dataset

N_PERSONAS  = 1000
N_MEDICINES = 5000
PERSONA_FILE = "indian_healthcare_personas_expanded.json"
BASE_DIR = Path(".")
RANDOM_STATE = 42


def _metrics(y_true, y_pred) -> dict:
    return {
        "R²":   round(float(r2_score(y_true, y_pred)), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def _train_eval(X, y, train_idx, test_idx, label: str) -> dict:
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )
    model = build_xgb_regressor()
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    train_m = _metrics(y_train, model.predict(X_train))
    test_m  = _metrics(y_test,  model.predict(X_test))

    print(f"\n  [{label}]")
    print(f"    Train  →  R²={train_m['R²']:.4f}  MAE={train_m['MAE']:.4f}  RMSE={train_m['RMSE']:.4f}")
    print(f"    Test   →  R²={test_m['R²']:.4f}  MAE={test_m['MAE']:.4f}  RMSE={test_m['RMSE']:.4f}")

    return {"train": train_m, "test": test_m, "model": model,
            "X_test": X_test, "y_test": y_test}


def main():
    print("=" * 65)
    print("PharmaDrishti — Model Benchmark (1000 Personas × 5000 Medicines)")
    print("=" * 65)
    t0 = time.time()

    # ── Step 1: Load 1000 personas ────────────────────────────────────────────
    print(f"\n[1/5] Loading {N_PERSONAS} personas from {PERSONA_FILE}...")
    personas = load_personas(str(BASE_DIR / PERSONA_FILE))
    if personas.empty:
        print("❌  Persona file not found.")
        return
    personas = personas.head(N_PERSONAS)
    print(f"   ✓ {len(personas)} personas loaded")

    # ── Step 2: Generate medicines ────────────────────────────────────────────
    print(f"\n[2/5] Generating {N_MEDICINES} medicines...")
    medicines = generate_medicines(n=N_MEDICINES)
    print(f"   ✓ {len(medicines)} medicines generated")

    # ── Step 3: Build base interaction dataset ────────────────────────────────
    print(f"\n[3/5] Building interaction dataset ({N_PERSONAS} × {N_MEDICINES})...")
    interactions = create_interaction_dataset(personas, medicines)
    # Compute baseline adoption_score (no KG) — used for XGBoost-only model
    interactions["adoption_score"] = compute_adoption_score(interactions)
    print(f"   ✓ {len(interactions):,} pairs  |  "
          f"adoption_score mean={interactions['adoption_score'].mean():.3f}  "
          f"std={interactions['adoption_score'].std():.3f}")

    # ── Step 4: Build KG-enhanced dataset ────────────────────────────────────
    print("\n[4/5] Building Knowledge Graph and extracting features...")
    nppa       = load_nppa_market_data(BASE_DIR)
    regulatory = load_regulatory_data(BASE_DIR)
    mapped     = map_nppa_to_products(nppa, regulatory["products"])
    graph      = build_knowledge_graph(
        mapped, regulatory["patents"], regulatory["exclusivity"], regulatory["reference"]
    )
    graph_features = extract_graph_features(graph)
    print(f"   ✓ KG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    enhanced = merge_graph_features_with_training_data(
        interactions_df=interactions,
        medicines_df=medicines,
        graph_features_df=graph_features,
        nppa_mapping_df=mapped,
    )

    # Reference drug flag
    ref = regulatory["reference"].copy()
    ref["appl_no"] = pd.to_numeric(ref.get("appl_no"), errors="coerce")
    ref_appl_nos = set(ref["appl_no"].dropna().unique())
    mapped_ref = mapped[["medicine_name", "appl_no"]].copy()
    mapped_ref["is_reference_drug"] = mapped_ref["appl_no"].isin(ref_appl_nos).astype(float)
    ref_flags = mapped_ref.groupby("medicine_name", as_index=False)["is_reference_drug"].max()
    if "medicine_name" in enhanced.columns:
        enhanced = enhanced.merge(ref_flags, on="medicine_name", how="left")
        enhanced["is_reference_drug"] = enhanced["is_reference_drug"].fillna(0.0)

    # Outlier handling
    if "price" in enhanced.columns:
        enhanced["medicine_price"] = enhanced["price"]
    outlier_cols = [c for c in ["medicine_price", "competitor_count",
                                "patent_remaining_years", "exclusivity_remaining_years",
                                "suitability_score"] if c in enhanced.columns]
    flags    = detect_outliers(enhanced, outlier_cols, method="iqr")
    enhanced = remove_or_flag_outliers(enhanced, flags, strategy="flag").reset_index(drop=True)

    # Propagate adoption_score to enhanced — recompute WITH KG columns.
    # This is the SINGLE shared target for all 3 models.
    # XGBoost-only predicts it without KG inputs (harder, honest gap).
    # KG-only predicts it from KG inputs alone.
    # Combined uses both — should score highest.
    enhanced = enhanced.reset_index(drop=True)
    enhanced["adoption_score"] = compute_adoption_score(enhanced).values
    print(f"   ✓ Enhanced dataset: {len(enhanced):,} rows after outlier filtering")
    print(f"   ✓ adoption_score mean={enhanced['adoption_score'].mean():.3f}  "
          f"std={enhanced['adoption_score'].std():.3f}")

    # ── Step 5: Build feature matrices ───────────────────────────────────────
    print("\n[5/5] Building feature matrices...")

    # Identify KG columns
    KG_COLS = [c for c in enhanced.columns if any(
        c.startswith(base) for base in [
            "competitor_count", "patent_remaining", "exclusivity_remaining",
            "patent_count", "exclusivity_count", "manufacturer_strength",
            "manufacturer_portfolio", "approval_age", "regulatory_barrier",
            "is_reference_drug", "kg_match_score",
        ]
    )]

    # Model 1: XGBoost-only — tabular features only, but predicts the SAME
    # KG-enriched adoption_score target as the other two models (fair comparison)
    tabular_only_df = enhanced.drop(columns=KG_COLS, errors="ignore")
    X_xgb, y_xgb = build_fused_feature_matrix(tabular_only_df)
    print(f"   ✓ XGBoost-only features: {X_xgb.shape[1]}")

    # Model 2: KG-only — only graph-derived features
    from feature_fusion import _robust_scale
    kg_only_df = enhanced[KG_COLS + ["adoption_score"]].copy().fillna(0.0)
    X_kg = _robust_scale(kg_only_df.drop(columns=["adoption_score"]))
    y_kg = kg_only_df["adoption_score"]
    print(f"   ✓ KG-only features: {X_kg.shape[1]}")

    # Model 3: Combined (tabular + KG via feature fusion)
    X_comb, y_comb = build_fused_feature_matrix(enhanced)
    print(f"   ✓ Combined features: {X_comb.shape[1]}")

    # ── Locked train/test split (same rows for all three) ────────────────────
    n = len(y_xgb)
    all_idx = np.arange(n)
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_STATE)
    print(f"\n   Split: {len(train_idx):,} train  |  {len(test_idx):,} test  (80/20, locked)")

    # ── Train and evaluate all three ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    r1 = _train_eval(X_xgb,  y_xgb,  train_idx, test_idx, "Model 1: XGBoost-only (tabular)")
    r2 = _train_eval(X_kg,   y_kg,   train_idx, test_idx, "Model 2: KG-only (graph features)")
    r3 = _train_eval(X_comb, y_comb, train_idx, test_idx, "Model 3: Combined (tabular + KG)")

    # ── Summary table ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"{'Model':<35} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 65)
    for label, res in [
        ("XGBoost-only (tabular)",     r1["test"]),
        ("KG-only (graph features)",   r2["test"]),
        ("Combined (tabular + KG)",    r3["test"]),
    ]:
        print(f"  {label:<33} {res['R²']:>8.4f} {res['MAE']:>8.4f} {res['RMSE']:>8.4f}")
    print("=" * 65)
    print(f"\nDone in {int(elapsed//60)}m {int(elapsed%60)}s")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "personas": N_PERSONAS,
        "medicines": N_MEDICINES,
        "xgboost_only":  {"train": r1["train"], "test": r1["test"]},
        "kg_only":       {"train": r2["train"], "test": r2["test"]},
        "combined":      {"train": r3["train"], "test": r3["test"]},
    }
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved → outputs/benchmark_results.json")


if __name__ == "__main__":
    main()
