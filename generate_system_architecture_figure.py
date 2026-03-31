"""
Generate a paper-ready system architecture figure for PharmaDrishti.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, title, subtitle="", fc="#f7fbff", ec="#334e68", lw=1.4, title_size=10.5, subtitle_size=8.5):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color="#102a43")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=subtitle_size, color="#486581", wrap=True)
    return box


def add_arrow(ax, start, end, color="#486581", lw=1.5, style="-|>"):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=lw, color=color)
    ax.add_patch(arrow)


def draw_layer_band(ax, y, h, label):
    band = FancyBboxPatch(
        (0.12, y),
        0.84,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        linewidth=1.2,
        edgecolor="#9fb3c8",
        facecolor="#f4f6f8",
    )
    ax.add_patch(band)
    ax.text(0.045, y + h / 2, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#102a43")


def generate_figure(output_path: str | Path = "outputs/system_architecture_pharmadrishti.png") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    layer_specs = [
        (0.83, 0.12, "1. Data Sources\nLayer"),
        (0.67, 0.12, "2. Data Integration &\nPreprocessing Layer"),
        (0.51, 0.12, "3. Knowledge Graph\nIntelligence Layer"),
        (0.35, 0.12, "4. Feature Engineering &\nModeling Layer"),
        (0.19, 0.12, "5. Evaluation &\nExplainability Layer"),
        (0.03, 0.12, "6. Decision Support &\nResearch Output Layer"),
    ]
    for y, h, label in layer_specs:
        draw_layer_band(ax, y, h, label)

    # Layer 1
    add_box(ax, 0.16, 0.855, 0.16, 0.07, "Persona Dataset", "Synthetic Indian healthcare personas")
    add_box(ax, 0.35, 0.855, 0.16, 0.07, "Medicine Dataset", "5,000 simulated medicine profiles")
    add_box(ax, 0.54, 0.855, 0.16, 0.07, "NPPA Market Data", "Pricing and market product records")
    add_box(ax, 0.73, 0.855, 0.20, 0.07, "FDA Orange Book Data", "Products, patents, exclusivity, reference")

    # Layer 2
    add_box(ax, 0.18, 0.695, 0.17, 0.065, "Ingredient Mapping", "Normalize medicine names and match ingredients")
    add_box(ax, 0.38, 0.695, 0.17, 0.065, "Regulatory Standardization", "Link Orange Book entities via appl_no")
    add_box(ax, 0.58, 0.695, 0.15, 0.065, "Outlier Detection", "IQR / Z-score auditing and flagging")
    add_box(ax, 0.76, 0.695, 0.15, 0.065, "Interaction Generation", "Persona × medicine interaction dataset")

    # Layer 3
    add_box(ax, 0.18, 0.535, 0.26, 0.065, "Knowledge Graph Construction", "Nodes: Drug, Ingredient, Manufacturer, Patent, Exclusivity")
    add_box(ax, 0.48, 0.535, 0.20, 0.065, "Regulatory Relations", "contains, manufactured_by, protected_by")
    add_box(ax, 0.72, 0.535, 0.18, 0.065, "Competition Modeling", "competes_with edges for shared ingredients")

    # Layer 4
    add_box(ax, 0.16, 0.375, 0.19, 0.065, "Base Features", "Persona, medicine, market, affordability signals")
    add_box(ax, 0.39, 0.375, 0.19, 0.065, "Graph Feature Extraction", "competitor_count, protection horizon, approval_age")
    add_box(ax, 0.62, 0.375, 0.14, 0.065, "Safe Feature Selection", "Leakage-prone columns removed")
    add_box(ax, 0.79, 0.375, 0.12, 0.065, "XGBoost Model", "Baseline + KG-enhanced regression")

    # Layer 5
    add_box(ax, 0.16, 0.215, 0.18, 0.065, "Train/Test Evaluation", "R², MAE, RMSE on unseen data")
    add_box(ax, 0.38, 0.215, 0.17, 0.065, "Cross-Validation", "5-fold validation and learning curves")
    add_box(ax, 0.59, 0.215, 0.16, 0.065, "Leakage Diagnostics", "Target shuffle and correlation checks")
    add_box(ax, 0.78, 0.215, 0.13, 0.065, "SHAP Analysis", "Feature attribution and residual analysis")

    # Layer 6
    add_box(ax, 0.16, 0.055, 0.18, 0.065, "Adoption Prediction", "Product adoption probability")
    add_box(ax, 0.38, 0.055, 0.16, 0.065, "Sustainability Score", "Competition and regulatory resilience")
    add_box(ax, 0.58, 0.055, 0.16, 0.065, "Dashboard & Insights", "Scenario analysis and segment targeting")
    add_box(ax, 0.78, 0.055, 0.13, 0.065, "Paper Figures/Tables", "Tables, plots, diagnostics, exports")

    # Vertical arrows from top layer to processing
    top_centers = [0.24, 0.43, 0.62, 0.83]
    second_centers = [0.265, 0.465, 0.655, 0.835]
    for x1, x2 in zip(top_centers, second_centers):
        add_arrow(ax, (x1, 0.855), (x2, 0.76))

    # Horizontal arrows layer 2
    add_arrow(ax, (0.35, 0.727), (0.38, 0.727))
    add_arrow(ax, (0.55, 0.727), (0.58, 0.727))
    add_arrow(ax, (0.73, 0.727), (0.76, 0.727))

    # Vertical arrows 2 -> 3
    for x in [0.31, 0.48, 0.655, 0.835]:
        add_arrow(ax, (x, 0.695), (x, 0.60))

    # Horizontal arrows layer 3
    add_arrow(ax, (0.44, 0.567), (0.48, 0.567))
    add_arrow(ax, (0.68, 0.567), (0.72, 0.567))

    # Vertical arrows 3 -> 4
    add_arrow(ax, (0.31, 0.535), (0.255, 0.44))
    add_arrow(ax, (0.58, 0.535), (0.485, 0.44))
    add_arrow(ax, (0.81, 0.535), (0.485, 0.44), style="->")
    add_arrow(ax, (0.255, 0.375), (0.39, 0.375))
    add_arrow(ax, (0.58, 0.375), (0.62, 0.375))
    add_arrow(ax, (0.76, 0.375), (0.79, 0.375))

    # Vertical arrows 4 -> 5
    for x in [0.25, 0.465, 0.67, 0.845]:
        add_arrow(ax, (x, 0.375), (x, 0.28))

    # Horizontal arrows layer 5
    add_arrow(ax, (0.34, 0.247), (0.38, 0.247))
    add_arrow(ax, (0.55, 0.247), (0.59, 0.247))
    add_arrow(ax, (0.75, 0.247), (0.78, 0.247))

    # Vertical arrows 5 -> 6
    for x in [0.25, 0.46, 0.67, 0.845]:
        add_arrow(ax, (x, 0.215), (x, 0.12))

    fig.suptitle("PharmaDrishti System Architecture", fontsize=20, fontweight="bold", y=0.985)
    ax.text(
        0.5,
        0.008,
        "Integration of synthetic market simulation, FDA Orange Book regulatory intelligence, knowledge graph modeling, and XGBoost-based adoption prediction.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#486581",
    )

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = generate_figure()
    print(path)