"""
PharmaDrishti — Improved Model Training Pipeline

Changes from baseline:
  1. adoption_score (not suitability_score) is the target variable.
  2. suitability_score is kept as an auxiliary input feature.
  3. Feature fusion layer combines persona + medicine + KG + suitability signal.
  4. Decision layer applies post-model competition / patent / pricing adjustments.
  5. 80/20 train-test split + 5-fold cross-validation.
  6. Full evaluation suite: R², MAE, RMSE, feature importance, SHAP, residuals,
     predicted-vs-actual, learning curve.
  7. Regularised XGBoost (alpha, lambda, early stopping).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split

from adoption_score_generator import compute_adoption_score
from data_loader import load_personas
from decision_layer import apply_decision_layer, extract_decision_inputs
from feature_fusion import build_fused_feature_matrix
from medicine_generator import generate_medicines
from model_config import build_xgb_regressor
from suitability_calculator import create_interaction_dataset

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Output directories ────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
PLOTS_DIR  = Path("outputs") / "training_plots"


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _plot_feature_importance(model, feature_names: list[str]) -> None:
    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi, x="importance", y="feature", ax=ax, color="#4C78A8")
    ax.set_title("XGBoost Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=200)
    plt.close(fig)
    print(f"   Saved: {PLOTS_DIR / 'feature_importance.png'}")


def _plot_shap(model, X_sample: pd.DataFrame) -> None:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {PLOTS_DIR / 'shap_summary.png'}")


def _plot_residuals(y_true: pd.Series, y_pred: np.ndarray) -> None:
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Residual distribution
    sns.histplot(residuals, bins=40, kde=True, ax=axes[0], color="#9467bd")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (actual − predicted)")

    # Predicted vs actual
    axes[1].scatter(y_true.values, y_pred, alpha=0.25, s=6, color="#1f77b4")
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    axes[1].set_title("Predicted vs Actual")
    axes[1].set_xlabel("Actual adoption_score")
    axes[1].set_ylabel("Predicted adoption_score")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "residuals.png", dpi=200)
    plt.close(fig)
    print(f"   Saved: {PLOTS_DIR / 'residuals.png'}")


def _plot_learning_curve(model_params: dict, X: pd.DataFrame, y: pd.Series) -> None:
    from xgboost import XGBRegressor
    lc_model = XGBRegressor(**model_params)
    sizes = np.linspace(0.1, 1.0, 7)
    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X, y, cv=5, scoring="r2",
        train_sizes=sizes, shuffle=True, random_state=42, n_jobs=-1,
    )
    t_mean, t_std = train_scores.mean(1), train_scores.std(1)
    v_mean, v_std = val_scores.mean(1),   val_scores.std(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, t_mean, "o-", label="Train R²",      color="#1f77b4")
    ax.plot(train_sizes, v_mean, "o-", label="Validation R²", color="#ff7f0e")
    ax.fill_between(train_sizes, t_mean - t_std, t_mean + t_std, alpha=0.15, color="#1f77b4")
    ax.fill_between(train_sizes, v_mean - v_std, v_mean + v_std, alpha=0.15, color="#ff7f0e")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R²")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "learning_curve.png", dpi=200)
    plt.close(fig)
    print(f"   Saved: {PLOTS_DIR / 'learning_curve.png'}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("PharmaDrishti — Improved Model Training Pipeline")
    print("=" * 65)
    _ensure_dirs()
    t0 = time.time()

    # ── 1. Load personas ──────────────────────────────────────────────────────
    print("\n[1/8] Loading personas...")
    personas = load_personas()
    if personas.empty:
        print("❌  No personas found. Run generate_personas.py first.")
        return
    print(f"   ✓ {len(personas):,} personas loaded")

    # ── 2. Generate medicines ─────────────────────────────────────────────────
    print("\n[2/8] Generating medicines...")
    medicines = generate_medicines(n=5000)
    print(f"   ✓ {len(medicines):,} medicine profiles generated")

    # ── 3. Create interaction dataset ─────────────────────────────────────────
    print("\n[3/8] Creating interaction dataset...")
    interactions = create_interaction_dataset(personas, medicines)
    print(f"   ✓ {len(interactions):,} interaction pairs  "
          f"({len(personas)} × {len(medicines)})")

    # ── 4. Compute adoption_score target ──────────────────────────────────────
    print("\n[4/8] Computing adoption_score target variable...")
    interactions["adoption_score"] = compute_adoption_score(interactions)
    print(f"   ✓ adoption_score range: "
          f"[{interactions['adoption_score'].min():.3f}, "
          f"{interactions['adoption_score'].max():.3f}]  "
          f"mean={interactions['adoption_score'].mean():.3f}")

    # ── 5. Feature fusion ─────────────────────────────────────────────────────
    print("\n[5/8] Building fused feature matrix...")
    X, y = build_fused_feature_matrix(interactions)
    print(f"   ✓ Feature matrix: {X.shape[1]} features × {X.shape[0]:,} rows")
    print(f"   ✓ Target (adoption_score): mean={y.mean():.3f}, std={y.std():.3f}")

    # ── 6. Train / test split ─────────────────────────────────────────────────
    print("\n[6/8] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    print(f"   ✓ Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ── 7. Train XGBoost ──────────────────────────────────────────────────────
    print("\n[7/8] Training XGBoost regressor...")
    model = build_xgb_regressor()
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    print("   ✓ Training complete")

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_m = _metrics(y_train, y_train_pred)
    test_m  = _metrics(y_test,  y_test_pred)

    print("\n   Train metrics:")
    print(f"     R²={train_m['r2']:.4f}  MAE={train_m['mae']:.4f}  RMSE={train_m['rmse']:.4f}")
    print("   Test metrics:")
    print(f"     R²={test_m['r2']:.4f}  MAE={test_m['mae']:.4f}  RMSE={test_m['rmse']:.4f}")

    # 5-fold cross-validation
    print("\n   Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        build_xgb_regressor(), X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    print(f"   CV R²: {cv_scores.round(4).tolist()}  "
          f"mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    # Decision layer — use .loc to correctly align test rows from interactions
    test_interactions = interactions.loc[X_test.index]
    di = extract_decision_inputs(test_interactions)
    dl_result = apply_decision_layer(
        y_test_pred,
        competitor_count=di["competitor_count"],
        patent_remaining_years=di["patent_remaining_years"],
        price_to_income_ratio=di["price_to_income_ratio"],
        compute_ci=False,
    )
    dl_m = _metrics(y_test, dl_result.final_score)
    print(f"\n   After decision layer — R²={dl_m['r2']:.4f}  "
          f"MAE={dl_m['mae']:.4f}  RMSE={dl_m['rmse']:.4f}")
    print(f"   Avg competition penalty : {dl_result.competition_penalty.mean():.3f}")
    print(f"   Avg patent factor       : {dl_result.patent_factor.mean():.3f}")
    print(f"   Avg pricing penalty     : {dl_result.pricing_penalty.mean():.3f}")

    # ── 8. Visualisations ─────────────────────────────────────────────────────
    print("\n[8/8] Generating evaluation plots...")
    _plot_feature_importance(model, list(X_train.columns))
    _plot_shap(model, X_train.sample(min(2000, len(X_train)), random_state=42))
    _plot_residuals(y_test, y_test_pred)
    _plot_learning_curve(model.get_xgb_params() | {"n_estimators": model.n_estimators}, X_train, y_train)

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(model,                  MODELS_DIR / "adoption_model.pkl")
    joblib.dump(list(X_train.columns),  MODELS_DIR / "feature_names.pkl")
    print(f"\n   ✓ Model saved  → {MODELS_DIR / 'adoption_model.pkl'}")
    print(f"   ✓ Features saved → {MODELS_DIR / 'feature_names.pkl'}")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"Done in {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
