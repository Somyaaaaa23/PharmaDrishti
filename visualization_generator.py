"""
Visualization generation for KG-enhanced PharmaDrishti analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def _save(fig, output_dir: Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return str(path)


def _system_architecture_plot(output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.5, "Personas + Synthetic\nMedicines"),
        (0.25, 0.5, "Suitability\nEngine"),
        (0.45, 0.5, "Interaction\nDataset"),
        (0.65, 0.5, "Knowledge\nGraph Layer"),
        (0.85, 0.5, "XGBoost\nPrediction"),
    ]
    for x, y, label in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#d8ecff", "edgecolor": "#1f77b4"},
        )
    for i in range(len(boxes) - 1):
        ax.annotate(
            "",
            xy=(boxes[i + 1][0] - 0.07, 0.5),
            xytext=(boxes[i][0] + 0.07, 0.5),
            arrowprops={"arrowstyle": "->", "lw": 2},
        )
    return _save(fig, output_dir, "architecture_diagram.png")


def _knowledge_graph_plot(graph: nx.Graph, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 8))
    drug_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "Drug"]
    ingredient_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "Ingredient"]
    manufacturer_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "Manufacturer"]
    patent_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "Patent"]
    excl_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "Exclusivity"]

    sub_nodes = set(drug_nodes[:80] + ingredient_nodes[:30] + manufacturer_nodes[:20] + patent_nodes[:40] + excl_nodes[:30])
    sub_graph = graph.subgraph(sub_nodes).copy()
    pos = nx.spring_layout(sub_graph, seed=42, k=0.6)

    nx.draw_networkx_nodes(sub_graph, pos, nodelist=[n for n in sub_graph.nodes() if n in drug_nodes], node_size=70, node_color="#1f77b4", label="Drug", ax=ax)
    nx.draw_networkx_nodes(sub_graph, pos, nodelist=[n for n in sub_graph.nodes() if n in ingredient_nodes], node_size=60, node_color="#2ca02c", label="Ingredient", ax=ax)
    nx.draw_networkx_nodes(sub_graph, pos, nodelist=[n for n in sub_graph.nodes() if n in manufacturer_nodes], node_size=60, node_color="#ff7f0e", label="Manufacturer", ax=ax)
    nx.draw_networkx_nodes(sub_graph, pos, nodelist=[n for n in sub_graph.nodes() if n in patent_nodes], node_size=45, node_color="#d62728", label="Patent", ax=ax)
    nx.draw_networkx_nodes(sub_graph, pos, nodelist=[n for n in sub_graph.nodes() if n in excl_nodes], node_size=45, node_color="#9467bd", label="Exclusivity", ax=ax)
    nx.draw_networkx_edges(sub_graph, pos, alpha=0.25, width=0.8, ax=ax)
    ax.set_title("Knowledge Graph Snapshot")
    ax.legend(loc="upper right")
    ax.axis("off")
    return _save(fig, output_dir, "knowledge_graph_network.png")


def _feature_importance_plot(model, feature_names: list[str], output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(feature_names))
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="#4C78A8")
    ax.set_title("Feature Importance (XGBoost)")
    return _save(fig, output_dir, "feature_importance.png")


def _kg_source_importance_plot(model, feature_names: list[str], output_dir: Path) -> str:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(feature_names))

    source_rows = []
    for feature_name, importance in zip(feature_names, importances):
        lower = str(feature_name).lower()
        if "patent_" in lower:
            source = "patents_csv"
        elif "exclusivity_" in lower:
            source = "exclusivity_csv"
        elif "reference" in lower:
            source = "reference_csv"
        elif "approval_age" in lower or "manufacturer_" in lower or "competitor_count" in lower:
            source = "products_csv"
        elif "kg_match_score" in lower:
            source = "mapping_quality"
        else:
            source = "interaction_base"
        source_rows.append({"source": source, "importance": float(importance)})

    source_df = pd.DataFrame(source_rows)
    grouped = (
        source_df.groupby("source", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=grouped, x="importance", y="source", ax=ax, color="#59A14F")
    ax.set_title("Importance by Feature Source")
    ax.set_xlabel("Total Importance Share")
    ax.set_ylabel("Feature Source")
    return _save(fig, output_dir, "kg_source_importance.png")


def _adoption_distribution_plot(predictions: np.ndarray, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(predictions, bins=30, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("Adoption Probability Distribution")
    ax.set_xlabel("Predicted Adoption Probability")
    return _save(fig, output_dir, "adoption_distribution.png")


def _model_comparison_plot(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> str:
    rows = []
    for model_name, values in metrics.items():
        for metric_name, metric_value in values.items():
            rows.append({"model": model_name, "metric": metric_name, "value": metric_value})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Baseline vs KG-Enhanced Model Performance")
    return _save(fig, output_dir, "model_performance_comparison.png")


def _ablation_study_plot(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> str:
    baseline = metrics.get("baseline", {})
    kg_enhanced = metrics.get("kg_enhanced", {})

    rows = []
    for metric_name in ["r2", "mae", "rmse"]:
        if metric_name not in baseline or metric_name not in kg_enhanced:
            continue
        base_value = float(baseline[metric_name])
        kg_value = float(kg_enhanced[metric_name])

        if metric_name == "r2":
            improvement = kg_value - base_value
        else:
            improvement = base_value - kg_value

        rows.append(
            {
                "metric": metric_name.upper(),
                "baseline": base_value,
                "kg_enhanced": kg_value,
                "improvement": improvement,
            }
        )

    ablation_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.8, 1]})

    compare_df = ablation_df.melt(
        id_vars=["metric", "improvement"],
        value_vars=["baseline", "kg_enhanced"],
        var_name="model",
        value_name="value",
    )
    sns.barplot(data=compare_df, x="metric", y="value", hue="model", ax=axes[0])
    axes[0].set_title("Ablation: Baseline vs KG-Enhanced")
    axes[0].set_xlabel("Metric")
    axes[0].set_ylabel("Value")

    sns.barplot(data=ablation_df, x="metric", y="improvement", ax=axes[1], color="#2ca02c")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Net Gain from KG Features")
    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("Gain (R²↑ / Error↓)")

    for index, row in ablation_df.reset_index(drop=True).iterrows():
        axes[1].text(
            index,
            row["improvement"] + (0.01 if row["improvement"] >= 0 else -0.01),
            f"{row['improvement']:+.4f}",
            ha="center",
            va="bottom" if row["improvement"] >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    return _save(fig, output_dir, "ablation_study.png")


def _outlier_plot(training_df: pd.DataFrame, output_dir: Path) -> str:
    candidate_cols = [c for c in ["price", "competitor_count", "patent_remaining_years", "exclusivity_remaining_years", "suitability_score"] if c in training_df.columns]
    melt_df = training_df[candidate_cols].copy().melt(var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melt_df, x="feature", y="value", ax=ax)
    ax.set_title("Outlier Detection Overview")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, output_dir, "outlier_detection.png")


def _sustainability_plot(training_df: pd.DataFrame, output_dir: Path) -> str:
    df = training_df.copy()
    if "sustainability_score" not in df.columns:
        df["sustainability_score"] = (
            0.35 * (1 / (1 + df.get("competitor_count", 0)))
            + 0.35 * (1 / (1 + df.get("patent_remaining_years", 0)))
            + 0.30 * (1 / (1 + df.get("exclusivity_remaining_years", 0)))
        )
    x_col = "manufacturer_strength" if "manufacturer_strength" in df.columns else "manufacturing"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(len(df), 3000), random_state=42), x=x_col, y="sustainability_score", alpha=0.35, ax=ax)
    ax.set_title("Sustainability vs Regulatory/Manufacturer Strength")
    return _save(fig, output_dir, "sustainability_analysis.png")


def generate_visualizations(
    graph: nx.Graph,
    enhanced_model,
    enhanced_feature_names: list[str],
    enhanced_predictions: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    training_df: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
) -> Dict[str, str]:
    output_path = Path(output_dir)
    files = {
        "architecture": _system_architecture_plot(output_path),
        "knowledge_graph": _knowledge_graph_plot(graph, output_path),
        "feature_importance": _feature_importance_plot(enhanced_model, enhanced_feature_names, output_path),
        "kg_source_importance": _kg_source_importance_plot(enhanced_model, enhanced_feature_names, output_path),
        "adoption_distribution": _adoption_distribution_plot(enhanced_predictions, output_path),
        "model_comparison": _model_comparison_plot(metrics, output_path),
        "ablation_study": _ablation_study_plot(metrics, output_path),
        "outlier_detection": _outlier_plot(training_df, output_path),
        "sustainability": _sustainability_plot(training_df, output_path),
    }
    return files
