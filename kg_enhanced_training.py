"""
KG-enhanced training pipeline for PharmaDrishti.

Architecture:
  1. Build interaction dataset from personas × N medicines.
  2. Compute adoption_score target (NOT suitability_score).
     suitability_score is kept as an auxiliary input feature.
  3. Baseline model  → trained on tabular interaction features only.
  4. KG-enhanced model → trained on tabular features PLUS graph features
     derived from ALL 4 standardised regulatory CSVs:
       • products_standardized.csv   (drug/applicant metadata)
       • patents_standardized.csv    (patent expiry)
       • exclusivity_standardized.csv (exclusivity periods)
       • reference_standardized.csv  (reference drug flags)
  5. Both models use the SAME train/test split (locked index) for a fair
     apples-to-apples comparison of R², MAE, RMSE.
  6. Decision layer applied to KG-enhanced predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from adoption_score_generator import compute_adoption_score
from data_loader import load_personas
from dataset_scaling import ensure_persona_dataset
from decision_layer import apply_decision_layer, extract_decision_inputs
from feature_fusion import build_fused_feature_matrix
from feature_selection import build_model_matrix
from graph_feature_extractor import (
    extract_graph_features,
    merge_graph_features_with_training_data,
)
from ingredient_mapper import map_nppa_to_products
from knowledge_graph_builder import build_knowledge_graph
from medicine_generator import generate_medicines
from model_config import build_xgb_regressor
from outlier_detector import detect_outliers, remove_or_flag_outliers
from regulatory_data_loader import load_nppa_market_data, load_regulatory_data
from suitability_calculator import create_interaction_dataset
from visualization_generator import generate_visualizations


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_reference_features(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    mapped_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge reference-drug flags from reference_standardized.csv into df.
    Maps via appl_no from the NPPA→products mapping table.
    """
    ref = reference_df.copy()
    ref["appl_no"] = pd.to_numeric(ref.get("appl_no"), errors="coerce")

    # is_reference_drug: 1 if the drug appears in the reference list
    ref_appl_nos = set(ref["appl_no"].dropna().unique())
    mapped_ref = mapped_df[["medicine_name", "appl_no"]].copy()
    mapped_ref["is_reference_drug"] = mapped_ref["appl_no"].isin(ref_appl_nos).astype(float)

    # aggregate to medicine_name level (take max so any hit = 1)
    ref_flags = mapped_ref.groupby("medicine_name", as_index=False)["is_reference_drug"].max()

    if "medicine_name" in df.columns:
        df = df.merge(ref_flags, on="medicine_name", how="left")
        df["is_reference_drug"] = df["is_reference_drug"].fillna(0.0)
    else:
        df["is_reference_drug"] = 0.0

    return df


def _evaluate_on_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int = 42,
) -> tuple:
    """Train and evaluate on a pre-defined split so both models see same rows."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    x_fit, x_valid, y_fit, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state
    )
    model = build_xgb_regressor(random_state=random_state)
    model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)

    pred = model.predict(X_test)
    metrics = {
        "r2":   float(r2_score(y_test, pred)),
        "mae":  float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    return model, metrics, X_test, y_test, pred


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    base_dir: str | Path = ".",
    n_personas: int | None = None,
    n_medicines: int = 5000,
    outlier_method: str = "iqr",
    outlier_strategy: str = "flag",
    persona_target_count: int | None = None,
    persona_file: str = "indian_healthcare_personas.json",
    random_state: int = 42,
):
    base = Path(base_dir)
    output_dir = base / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ── 1. Build base interaction dataset (1000 personas × N medicines) ───────
    persona_path = base / persona_file
    if persona_target_count is not None and persona_target_count > 0:
        persona_path = ensure_persona_dataset(persona_target_count, persona_path)

    personas = load_personas(str(persona_path))
    if n_personas is not None and n_personas > 0:
        personas = personas.head(n_personas)

    medicines = generate_medicines(n=n_medicines)
    interactions = create_interaction_dataset(personas, medicines)

    print(f"[pipeline] Interaction dataset: {len(interactions):,} rows  "
          f"({len(personas)} personas × {len(medicines)} medicines)")

    # ── 1b. Compute adoption_score target ─────────────────────────────────────
    interactions["adoption_score"] = compute_adoption_score(interactions)
    print(f"[pipeline] adoption_score: mean={interactions['adoption_score'].mean():.3f}  "
          f"std={interactions['adoption_score'].std():.3f}")

    # ── 2. Load ALL 4 standardised regulatory CSVs ───────────────────────────
    nppa       = load_nppa_market_data(base)
    regulatory = load_regulatory_data(base)   # products, patents, exclusivity, reference

    print("[pipeline] Regulatory data loaded:")
    print(f"  NPPA market rows    : {len(nppa):,}")
    print(f"  products rows       : {len(regulatory['products']):,}")
    print(f"  patents rows        : {len(regulatory['patents']):,}")
    print(f"  exclusivity rows    : {len(regulatory['exclusivity']):,}")
    print(f"  reference rows      : {len(regulatory['reference']):,}")

    # ── 3. Build KG from products + patents + exclusivity ────────────────────
    mapped        = map_nppa_to_products(nppa, regulatory["products"])
    graph         = build_knowledge_graph(
        mapped,
        regulatory["patents"],
        regulatory["exclusivity"],
        regulatory["reference"],
    )
    graph_features = extract_graph_features(graph)

    print(f"[pipeline] KG nodes: {graph.number_of_nodes():,}  "
          f"edges: {graph.number_of_edges():,}")

    # ── 4. Merge graph features into interactions ─────────────────────────────
    enhanced = merge_graph_features_with_training_data(
        interactions_df=interactions,
        medicines_df=medicines,
        graph_features_df=graph_features,
        nppa_mapping_df=mapped,
    )

    # ── 5. Add reference-drug features (4th CSV) ──────────────────────────────
    enhanced = _add_reference_features(enhanced, regulatory["reference"], mapped)

    # ── 6. Outlier handling on enhanced dataset ───────────────────────────────
    if "price" in enhanced.columns:
        enhanced["medicine_price"] = enhanced["price"]
    outlier_columns = [c for c in [
        "medicine_price", "competitor_count",
        "patent_remaining_years", "exclusivity_remaining_years",
        "suitability_score",
    ] if c in enhanced.columns]
    outlier_flags = detect_outliers(enhanced, outlier_columns, method=outlier_method)
    enhanced      = remove_or_flag_outliers(enhanced, outlier_flags, strategy=outlier_strategy)
    enhanced      = enhanced.reset_index(drop=True)

    # ── 7. Build feature matrices ─────────────────────────────────────────────
    # Baseline: tabular features from the RAW interaction dataset
    # (subset to same rows that survived outlier filtering)
    interactions_aligned = interactions.loc[enhanced.index].reset_index(drop=True)
    baseline_x, baseline_y = build_model_matrix(interactions_aligned)

    # KG-enhanced: use feature fusion layer (persona + medicine + suitability + KG)
    # adoption_score is already in enhanced (propagated from interactions_aligned)
    if "adoption_score" not in enhanced.columns:
        enhanced["adoption_score"] = interactions_aligned["adoption_score"].values
    enhanced_x, enhanced_y = build_fused_feature_matrix(enhanced)

    print(f"[pipeline] Baseline feature matrix : {baseline_x.shape}")
    print(f"[pipeline] KG feature matrix       : {enhanced_x.shape}")

    # Both targets should be adoption_score from the same source rows
    assert np.allclose(baseline_y.values, enhanced_y.values, atol=1e-4), \
        "Target mismatch between baseline and KG datasets — check row alignment!"

    # ── 8. Locked train/test split (same rows for both models) ───────────────
    n = len(baseline_y)
    all_idx   = np.arange(n)
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=random_state)

    # ── 9. Train and evaluate BOTH models ────────────────────────────────────
    _, baseline_metrics, _, _, _ = _evaluate_on_split(
        baseline_x, baseline_y, train_idx, test_idx, random_state
    )
    enhanced_model, enhanced_metrics, x_test, y_test, pred = _evaluate_on_split(
        enhanced_x, enhanced_y, train_idx, test_idx, random_state
    )

    # ── 10. Persist artifacts ─────────────────────────────────────────────────
    model_dir = base / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(enhanced_model,          model_dir / "adoption_model_kg_enhanced.pkl")
    joblib.dump(list(enhanced_x.columns), model_dir / "feature_names_kg_enhanced.pkl")

    # ── 10b. Apply decision layer to KG-enhanced test predictions ────────────
    di = extract_decision_inputs(enhanced.iloc[test_idx])
    dl_result = apply_decision_layer(
        pred,
        competitor_count=di["competitor_count"],
        patent_remaining_years=di["patent_remaining_years"],
        price_to_income_ratio=di["price_to_income_ratio"],
        compute_ci=False,
    )
    from sklearn.metrics import r2_score as _r2, mean_absolute_error as _mae, mean_squared_error as _mse
    dl_metrics = {
        "r2":   float(_r2(y_test, dl_result.final_score)),
        "mae":  float(_mae(y_test, dl_result.final_score)),
        "rmse": float(np.sqrt(_mse(y_test, dl_result.final_score))),
    }
    enhanced_metrics["after_decision_layer"] = dl_metrics

    metrics = {"baseline": baseline_metrics, "kg_enhanced": enhanced_metrics}
    (output_dir / "kg_model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── 11. Visualisations ────────────────────────────────────────────────────
    plot_files = generate_visualizations(
        graph=graph,
        enhanced_model=enhanced_model,
        enhanced_feature_names=list(enhanced_x.columns),
        enhanced_predictions=pred,
        metrics=metrics,
        training_df=enhanced,
        output_dir=output_dir / "figures",
    )

    # ── 12. Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Metric':<10}  {'Baseline':>12}  {'KG-Enhanced':>12}  {'+ Decision':>12}")
    print("="*60)
    for k in ("r2", "mae", "rmse"):
        dl_val = dl_metrics.get(k, float("nan"))
        print(f"{k.upper():<10}  {baseline_metrics[k]:>12.6f}  "
              f"{enhanced_metrics[k]:>12.6f}  {dl_val:>12.6f}")
    print("="*60)

    print("\nGenerated Visualizations:")
    for name, path in plot_files.items():
        print(f"  - {name}: {path}")

    return {
        "baseline_metrics":  baseline_metrics,
        "enhanced_metrics":  enhanced_metrics,
        "plot_files":        plot_files,
        "x_test_shape":      x_test.shape,
        "y_test_size":       len(y_test),
    }


if __name__ == "__main__":
    run_pipeline(
        n_personas=1000,
        n_medicines=5000,
        persona_file="indian_healthcare_personas_expanded.json",
    )
