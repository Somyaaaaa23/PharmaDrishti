"""
PharmaDrishti Overfitting and Data Leakage Diagnostic Pipeline.

Implements:
1) Proper train/test split metrics
2) 5-fold cross-validation
3) Learning curve
4) Target shuffle leakage test
5) Feature-target correlation checks
6) XGBoost + SHAP importance
7) Residual analysis
8) Final diagnostic report with risk assessment
9) Research-paper-ready figure exports
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from xgboost import XGBRegressor

from adoption_score_generator import compute_adoption_score
from data_loader import load_personas
from feature_fusion import build_fused_feature_matrix
from feature_selection import build_model_matrix
from graph_feature_extractor import (
    extract_graph_features,
    merge_graph_features_with_training_data,
)
from ingredient_mapper import map_nppa_to_products
from knowledge_graph_builder import build_knowledge_graph
from medicine_generator import generate_medicines
from model_config import DEFAULT_XGB_PARAMS, build_xgb_regressor
from outlier_detector import detect_outliers, remove_or_flag_outliers
from regulatory_data_loader import load_nppa_market_data, load_regulatory_data
from suitability_calculator import create_interaction_dataset


def _build_dataset(
    mode: str,
    base_dir: Path,
    persona_file: str,
    n_personas: int,
    n_medicines: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    personas = load_personas(str(base_dir / persona_file)).head(n_personas)
    medicines = generate_medicines(n=n_medicines)
    interactions = create_interaction_dataset(personas, medicines)
    # Compute adoption_score target before building feature matrix
    interactions["adoption_score"] = compute_adoption_score(interactions)
    X, y = build_model_matrix(interactions)

    if mode == "baseline":
        return X, y

    nppa = load_nppa_market_data(base_dir)
    regulatory = load_regulatory_data(base_dir)
    mapped = map_nppa_to_products(nppa, regulatory["products"])
    graph = build_knowledge_graph(
        mapped,
        regulatory["patents"],
        regulatory["exclusivity"],
        regulatory["reference"],
    )
    graph_features = extract_graph_features(graph)

    enhanced = merge_graph_features_with_training_data(
        interactions_df=interactions,
        medicines_df=medicines,
        graph_features_df=graph_features,
        nppa_mapping_df=mapped,
    )

    reference_df = regulatory["reference"].copy()
    reference_df["appl_no"] = pd.to_numeric(reference_df.get("appl_no"), errors="coerce")
    ref_appl_nos = set(reference_df["appl_no"].dropna().unique())
    mapped_ref = mapped[["medicine_name", "appl_no"]].copy()
    mapped_ref["is_reference_drug"] = mapped_ref["appl_no"].isin(ref_appl_nos).astype(float)
    ref_flags = mapped_ref.groupby("medicine_name", as_index=False)["is_reference_drug"].max()
    if "medicine_name" in enhanced.columns:
        enhanced = enhanced.merge(ref_flags, on="medicine_name", how="left")
        enhanced["is_reference_drug"] = enhanced["is_reference_drug"].fillna(0.0)

    if "price" in enhanced.columns:
        enhanced["medicine_price"] = enhanced["price"]
    outlier_columns = [
        column
        for column in [
            "medicine_price",
            "competitor_count",
            "patent_remaining_years",
            "exclusivity_remaining_years",
            "suitability_score",
        ]
        if column in enhanced.columns
    ]
    outlier_flags = detect_outliers(enhanced, outlier_columns, method="iqr")
    enhanced = remove_or_flag_outliers(enhanced, outlier_flags, strategy="flag")

    # Propagate adoption_score and use feature fusion layer
    if "adoption_score" not in enhanced.columns:
        enhanced["adoption_score"] = interactions.loc[enhanced.index, "adoption_score"].values
    x_enhanced, y_enhanced = build_fused_feature_matrix(enhanced)
    return x_enhanced, y_enhanced


def _load_model_params(model_path: Path | None) -> Dict:
    if model_path is None or not model_path.exists():
        return dict(DEFAULT_XGB_PARAMS)

    try:
        loaded_model = joblib.load(model_path)
        if hasattr(loaded_model, "get_xgb_params"):
            params = loaded_model.get_xgb_params()
            params.update(
                {
                    "n_estimators": getattr(loaded_model, "n_estimators", DEFAULT_MODEL_PARAMS["n_estimators"]),
                    "max_depth": getattr(loaded_model, "max_depth", DEFAULT_XGB_PARAMS["max_depth"]),
                    "learning_rate": getattr(loaded_model, "learning_rate", DEFAULT_XGB_PARAMS["learning_rate"]),
                    "subsample": getattr(loaded_model, "subsample", DEFAULT_XGB_PARAMS["subsample"]),
                    "colsample_bytree": getattr(
                        loaded_model,
                        "colsample_bytree",
                        DEFAULT_XGB_PARAMS["colsample_bytree"],
                    ),
                    "objective": "reg:squarederror",
                    "verbosity": 0,
                    "n_jobs": -1,
                    "random_state": 42,
                }
            )
            return params
    except Exception:
        pass

    return dict(DEFAULT_XGB_PARAMS)


def _evaluate_split(model: XGBRegressor, X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    }
    return {
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "model": model,
    }


def _run_cross_validation(model_params: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
    model = build_xgb_regressor(**model_params)
    fold_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    return {
        "fold_scores": [float(v) for v in fold_scores],
        "mean_r2": float(np.mean(fold_scores)),
        "std_r2": float(np.std(fold_scores)),
    }


def _plot_learning_curve(model_params: Dict, X: pd.DataFrame, y: pd.Series, output_dir: Path) -> str:
    model = build_xgb_regressor(**model_params)
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=42,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", label="Training R²", color="#1f77b4")
    ax.plot(train_sizes, valid_mean, "o-", label="Validation R²", color="#ff7f0e")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="#1f77b4")
    ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2, color="#ff7f0e")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R²")
    ax.legend()
    path = output_dir / "learning_curve.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _target_shuffle_test(model_params: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
    rng = np.random.default_rng(42)
    y_random = pd.Series(rng.permutation(y.values), index=y.index)

    model = XGBRegressor(**model_params)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_random,
        test_size=0.2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    random_r2 = float(r2_score(y_test, y_pred))

    return {
        "shuffle_r2": random_r2,
        "leakage_flag": random_r2 > 0.1,
    }


def _correlation_analysis(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> Dict:
    df = X.copy()
    df["target"] = y.values
    numeric = df.select_dtypes(include=[np.number]).copy()
    corr = numeric.corr()

    target_corr = corr["target"].drop("target", errors="ignore").abs().sort_values(ascending=False)
    leakage_features = target_corr[target_corr > 0.95].index.tolist()

    heatmap_features = target_corr.head(20).index.tolist()
    heatmap_df = numeric[heatmap_features + ["target"]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature-Target Correlation Heatmap")
    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "top_correlations": {k: float(v) for k, v in target_corr.head(10).to_dict().items()},
        "leakage_features": leakage_features,
        "heatmap_path": str(path),
    }


def _importance_plots(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=feature_importance.head(20),
        x="importance",
        y="feature",
        ax=ax1,
        color="#4C78A8",
    )
    ax1.set_title("XGBoost Feature Importance")
    fi_path = output_dir / "xgboost_feature_importance.png"
    fig1.savefig(fi_path, dpi=220, bbox_inches="tight")
    plt.close(fig1)

    shap_sample = X_train.sample(min(len(X_train), 3000), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)

    shap_path = output_dir / "shap_summary.png"
    plt.figure(figsize=(10, 6))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The NumPy global RNG was seeded by calling `np.random.seed`",
            category=FutureWarning,
        )
        shap.summary_plot(shap_values, shap_sample, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(shap_path, dpi=220, bbox_inches="tight")
    plt.close()

    total_importance = feature_importance["importance"].sum()
    top_share = (
        float(feature_importance.iloc[0]["importance"] / total_importance)
        if total_importance > 0
        else 0.0
    )

    return {
        "feature_importance_path": str(fi_path),
        "shap_summary_path": str(shap_path),
        "top_features": feature_importance.head(10).to_dict(orient="records"),
        "dominant_feature_flag": top_share > 0.7,
        "top_feature_share": top_share,
    }


def _residual_plots(y_test: pd.Series, y_pred: np.ndarray, output_dir: Path) -> Dict:
    residual = y_test.values - y_pred

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(residual, bins=40, kde=True, ax=ax1, color="#9467bd")
    ax1.set_title("Residual Distribution")
    ax1.set_xlabel("Residual (y_true - y_pred)")
    hist_path = output_dir / "residual_distribution.png"
    fig1.savefig(hist_path, dpi=220, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.35, ax=ax2)
    min_val = min(float(np.min(y_test.values)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_test.values)), float(np.max(y_pred)))
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.2)
    ax2.set_title("Predicted vs Actual")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    scatter_path = output_dir / "predicted_vs_actual.png"
    fig2.savefig(scatter_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    return {
        "residual_distribution_path": str(hist_path),
        "predicted_vs_actual_path": str(scatter_path),
    }


def _risk_assessment(
    split_metrics: Dict,
    cv_results: Dict,
    shuffle_results: Dict,
    corr_results: Dict,
    importance_results: Dict,
) -> Dict:
    overfit_flag = (
        (split_metrics["train_r2"] - split_metrics["test_r2"] > 0.03)
        or (split_metrics["test_mae"] > split_metrics["train_mae"] * 1.4)
        or (split_metrics["test_rmse"] > split_metrics["train_rmse"] * 1.4)
    )
    unstable_flag = cv_results["std_r2"] > 0.05
    leakage_flag = bool(shuffle_results["leakage_flag"] or len(corr_results["leakage_features"]) > 0)
    dominance_flag = bool(importance_results["dominant_feature_flag"])

    if not overfit_flag and not unstable_flag and not leakage_flag and not dominance_flag:
        verdict = "Model appears well generalized"
    else:
        verdict = "Possible overfitting / data leakage detected"

    return {
        "overfitting_flag": overfit_flag,
        "instability_flag": unstable_flag,
        "leakage_flag": leakage_flag,
        "single_feature_dominance_flag": dominance_flag,
        "verdict": verdict,
    }


def run_overfitting_diagnostics(
    mode: str = "baseline",
    base_dir: str | Path = ".",
    model_path: str | None = None,
    persona_file: str = "indian_healthcare_personas.json",
    n_personas: int = 1000,
    n_medicines: int = 1000,
    quick: bool = False,
) -> Dict:
    """Run diagnostics. Set quick=True to skip SHAP/CV/learning-curve for large datasets."""
    base = Path(base_dir)
    output_dir = base / "outputs" / "diagnostics" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = _build_dataset(
        mode=mode,
        base_dir=base,
        persona_file=persona_file,
        n_personas=n_personas,
        n_medicines=n_medicines,
    )
    model_params = _load_model_params(Path(model_path) if model_path else None)

    split_eval = _evaluate_split(XGBRegressor(**model_params), X, y)

    if quick:
        print("[quick mode] Skipping CV, learning curve, and SHAP for faster run.")
        cv_results = {"fold_scores": [], "mean_r2": float("nan"), "std_r2": float("nan")}
        learning_curve_path = "skipped"
        # lightweight feature importance only (no SHAP)
        fi_df = pd.DataFrame({
            "feature": split_eval["X_train"].columns,
            "importance": split_eval["model"].feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_path = output_dir / "xgboost_feature_importance.png"
        fig, ax = plt.subplots(figsize=(9, 6))
        import seaborn as _sns
        _sns.barplot(data=fi_df.head(20), x="importance", y="feature", ax=ax, color="#4C78A8")
        ax.set_title("XGBoost Feature Importance")
        fig.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        total_imp = fi_df["importance"].sum()
        top_share = float(fi_df.iloc[0]["importance"] / total_imp) if total_imp > 0 else 0.0
        importance_results = {
            "feature_importance_path": str(fi_path),
            "shap_summary_path": "skipped",
            "top_features": fi_df.head(10).to_dict(orient="records"),
            "dominant_feature_flag": top_share > 0.7,
            "top_feature_share": top_share,
        }
        shuffle_results = _target_shuffle_test(model_params, X, y)
        corr_results = _correlation_analysis(X, y, output_dir)
        residual_results = _residual_plots(split_eval["y_test"], split_eval["y_test_pred"], output_dir)
    else:
        cv_results = _run_cross_validation(model_params, X, y)
        learning_curve_path = _plot_learning_curve(model_params, X, y, output_dir)
        shuffle_results = _target_shuffle_test(model_params, X, y)
        corr_results = _correlation_analysis(X, y, output_dir)
        importance_results = _importance_plots(split_eval["model"], split_eval["X_train"], output_dir)
        residual_results = _residual_plots(split_eval["y_test"], split_eval["y_test_pred"], output_dir)

    summary = _risk_assessment(
        split_metrics=split_eval["metrics"],
        cv_results=cv_results,
        shuffle_results=shuffle_results,
        corr_results=corr_results,
        importance_results=importance_results,
    )

    report = {
        "mode": mode,
        "train_test_metrics": split_eval["metrics"],
        "cross_validation": cv_results,
        "target_shuffle_test": shuffle_results,
        "correlation_check": corr_results,
        "importance_summary": {
            "top_features": importance_results["top_features"],
            "top_feature_share": importance_results["top_feature_share"],
        },
        "risk_assessment": summary,
        "plots": {
            "learning_curve": learning_curve_path,
            "feature_importance": importance_results["feature_importance_path"],
            "shap_summary": importance_results["shap_summary_path"],
            "residual_distribution": residual_results["residual_distribution_path"],
            "predicted_vs_actual": residual_results["predicted_vs_actual_path"],
            "correlation_heatmap": corr_results["heatmap_path"],
        },
    }

    report_path = output_dir / "overfitting_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Overfitting Diagnostic Report ===")
    print(f"Mode: {mode}")
    print("\nTrain vs Test performance")
    for key, value in split_eval["metrics"].items():
        print(f"- {key}: {value:.6f}")

    print("\nCross-validation (5-fold)")
    print(f"- Fold scores: {cv_results['fold_scores']}")
    print(f"- Mean R2: {cv_results['mean_r2']:.6f}")
    print(f"- Std R2: {cv_results['std_r2']:.6f}")

    print("\nLeakage checks")
    print(f"- Target shuffle R2: {shuffle_results['shuffle_r2']:.6f}")
    print(f"- Correlation leakage features (>0.95): {corr_results['leakage_features']}")

    print("\nTop important features")
    for row in importance_results["top_features"][:5]:
        print(f"- {row['feature']}: {row['importance']:.6f}")

    print("\nOverfitting risk assessment")
    for flag_name, flag_value in summary.items():
        if flag_name != "verdict":
            print(f"- {flag_name}: {flag_value}")
    print(f"\nFinal verdict: {summary['verdict']}")
    print(f"\nSaved report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="PharmaDrishti overfitting and leakage diagnostics")
    parser.add_argument("--mode", choices=["baseline", "kg_enhanced"], default="baseline")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--persona-file", default="indian_healthcare_personas.json")
    parser.add_argument("--n-personas", type=int, default=1000)
    parser.add_argument("--n-medicines", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Skip SHAP/CV/learning-curve for faster large-scale runs")
    args = parser.parse_args()

    run_overfitting_diagnostics(
        mode=args.mode,
        base_dir=args.base_dir,
        model_path=args.model_path,
        persona_file=args.persona_file,
        n_personas=args.n_personas,
        n_medicines=args.n_medicines,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
