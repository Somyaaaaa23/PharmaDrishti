"""Generate docs/Architecture_Diagram.pdf (+png) and docs/PharmaDrishti_Presentation.pdf."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)
M = json.loads((ROOT / "digital_twin/models/metrics.json").read_text())
BLUE, INK, MUTED, YEL = "#004aad", "#102a43", "#486581", "#ffde59"


def box(ax, x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec="#334e68", lw=1.4))
    ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center", fontsize=8.3, color=MUTED, linespacing=1.4)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=MUTED))


def architecture():
    fig, ax = plt.subplots(figsize=(14, 7.8)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "PharmaDrishti GlucoTwin — Type 2 Diabetes Digital Twin Architecture", ha="center", fontsize=15, fontweight="bold", color=INK)
    # sources
    box(ax, 0.02, 0.62, 0.2, 0.22, "Static / Historical EHR", "Synthea-style synthetic records\nage, sex, state, BMI, BP\nHbA1c, fasting glucose, LDL\nTCF7L2 genetic marker\nDx history, metformin / insulin", "#e6f0ff")
    box(ax, 0.02, 0.28, 0.2, 0.25, "Dynamic Wearable / IoT", "Simulated 5-min streams (7 days)\nCGM glucose (Libre/Dexcom-like)\nHeart rate, HRV (RMSSD)\nSteps, sleep stages\nMeal / carb logs", "#fff7d6")
    # processing
    box(ax, 0.28, 0.62, 0.2, 0.22, "EHR Ingestion", "Validation, encoding\nper-patient static vector\n(12 clinical features)", "#f0f4f8")
    box(ax, 0.28, 0.28, 0.2, 0.25, "Time-series Feature Eng.", "Rolling windows 30m–24h\nglucose trend, variability, TIR\ncarbs 1h/3h, steps 1h\nHR / HRV 6h, deep sleep", "#f0f4f8")
    box(ax, 0.53, 0.43, 0.22, 0.25, "Fusion + Prediction Model", "Static + dynamic feature join\nGradient-boosted trees\n(HistGradientBoosting)\nPatient-level train/test split\nOutput: P(spike >180 in 2 h)", "#e3f9e5")
    # outputs
    box(ax, 0.79, 0.62, 0.19, 0.22, "Virtual Patient State", "Risk score + alert level\n(LOW / MODERATE / HIGH)\n7-day TIR, mean glucose", "#ffe3e3")
    box(ax, 0.79, 0.28, 0.19, 0.25, "Doctor Dashboard", "Streamlit + Plotly\nEHR summary, CGM replay\nearly-warning alerts\nwhat-if: walk / carb cut\nsimulated on the twin", "#ffe3e3")
    arrow(ax, (0.22, 0.73), (0.28, 0.73)); arrow(ax, (0.22, 0.405), (0.28, 0.405))
    arrow(ax, (0.48, 0.73), (0.56, 0.68)); arrow(ax, (0.48, 0.405), (0.56, 0.43))
    arrow(ax, (0.74, 0.6), (0.79, 0.7)); arrow(ax, (0.74, 0.5), (0.79, 0.42)); arrow(ax, (0.885, 0.53), (0.885, 0.62))
    ax.add_patch(FancyArrowPatch((0.885, 0.27), (0.12, 0.27), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#d64545", ls="--"))
    ax.text(0.5, 0.08, "Feedback loop: new CGM/wearable readings continuously update the twin (every 5 min)", ha="center", fontsize=9, color="#d64545")
    ax.text(0.5, 0.03, f"Held-out patients: ROC-AUC {M['roc_auc_fused']} · recall {M['recall']} · PR-AUC {M['pr_auc_fused']}  |  Privacy: 100% synthetic data (DPDP / HIPAA safe)", ha="center", fontsize=9.5, color=INK)
    fig.savefig(DOCS / "Architecture_Diagram.pdf", bbox_inches="tight"); fig.savefig(DOCS / "architecture.png", dpi=180, bbox_inches="tight"); plt.close(fig)


SLIDES = [
    ("PharmaDrishti GlucoTwin", None),
    ("Problem Statement", [
        "- India has ~101 million people with diabetes and ~136 million with prediabetes (ICMR-INDIAB, 2023)",
        "- Care is reactive: patients are seen every 3 months, HbA1c is a lagging indicator",
        "- Post-meal glucose spikes drive long-term complications (heart, kidney, eye, nerve damage)",
        "- CGMs and smartwatches produce data, but doctors have no tool that fuses it with the EHR",
        "  and warns them before an adverse event happens"]),
    ("Healthcare Use Case", [
        "- Condition: Type 2 Diabetes (one of India's most common chronic conditions)",
        "- Adverse event: hyperglycaemic spike (CGM > 180 mg/dL) predicted 2 hours in advance",
        "- Doctor: sees each virtual patient's EHR + live CGM / HR / HRV / steps / sleep in one view",
        "- Early warning -> nudge a 15-min walk, defer carbs, review the pre-meal dose",
        "- What-if simulation on the twin: test an intervention before advising the patient",
        "- Population view: rank patients by risk for remote monitoring in under-served districts"]),
    ("Data Streams (Sandbox Rules)", [
        "1. Static EHR (synthetic, Synthea-style): demographics, state, BMI, BP, HbA1c, fasting glucose,",
        "   LDL, TCF7L2 risk alleles, family history, T2D / hypertension Dx, metformin / insulin",
        "2. Dynamic wearables (simulated, 5-min): CGM glucose, heart rate, HRV (RMSSD), steps,",
        "   sleep stages, meal carb logs. Model includes meal absorption, insulin resistance,",
        "   activity effect, dawn phenomenon, poor-sleep stress",
        f"- {M['patients']} virtual patients x {M['days_per_patient']} days = {M['train_samples'] + M['test_samples']:,} labelled 5-min windows",
        "- No real patient data -> compliant with DPDP Act / HIPAA"]),
    ("Architecture", "IMG"),
    ("AI/ML Model", [
        "- Fusion: 12 static EHR features joined with 14 rolling wearable features at each time step",
        "- Model: gradient-boosted decision trees (scikit-learn HistGradientBoostingClassifier),",
        "  class-balanced, 300 iterations",
        "- Label: glucose currently <= 180 and > 180 at any point in the next 120 minutes",
        "- Validation: patient-level split - 20% of virtual patients never seen in training",
        "- Ablation shows the value of fusing both data streams (next slide)"]),
    ("Outcomes & Results", [
        f"- Fused model ROC-AUC: {M['roc_auc_fused']}   |   PR-AUC: {M['pr_auc_fused']}",
        f"- Recall {M['recall']}  ·  Precision {M['precision']}  ·  F1 {M['f1']}   (threshold 0.5, spike rate {M['spike_rate']:.1%})",
        f"- Ablation: wearable-only AUC {M['roc_auc_wearable_only']}, EHR-only AUC {M['roc_auc_ehr_only']} -> fusion is best",
        "- Most spikes are flagged up to 2 h ahead -> time to act",
        "- Working doctor dashboard: CGM replay, alerts, what-if intervention simulator",
        "- Trains in ~15 s on a laptop; inference in milliseconds"]),
    ("Tech Stack", [
        "- Python 3.10+, NumPy, Pandas",
        "- scikit-learn (HistGradientBoostingClassifier), joblib",
        "- Streamlit + Plotly doctor dashboard",
        "- Matplotlib (architecture diagram & slides)",
        "- Existing PharmaDrishti modules: XGBoost, SHAP, NetworkX knowledge graph of",
        "  NPPA + FDA Orange Book data for therapy affordability & access analysis"]),
    ("Limitations & Roadmap", [
        "- Synthetic physiology - next: validate on OhioT1DM / open CGM datasets and MIMIC-IV EHRs",
        "- Add hypoglycaemia (<70) and cardiovascular-anomaly (HRV + sleep) predictions",
        "- Personalised fine-tuning per patient as more twin data arrives",
        "- FHIR / ABDM (Ayushman Bharat Digital Mission) integration for real EHR ingestion",
        "- Link to PharmaDrishti pricing engine: recommend affordable therapy per patient segment"]),
]


def presentation():
    with PdfPages(DOCS / "PharmaDrishti_Presentation.pdf") as pdf:
        for i, (title, body) in enumerate(SLIDES):
            fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=BLUE)); ax.add_patch(plt.Rectangle((0, 0.875), 1, 0.006, color=YEL))
            ax.text(0.04, 0.94, title, color="white", fontsize=28, fontweight="bold", va="center")
            ax.text(0.96, 0.03, f"PharmaDrishti GlucoTwin  |  {i + 1}/{len(SLIDES)}", color=MUTED, fontsize=10, ha="right")
            if body is None:
                ax.text(0.5, 0.62, "A Type 2 Diabetes Digital Twin", ha="center", fontsize=30, color=INK, fontweight="bold")
                ax.text(0.5, 0.5, "Fusing synthetic EHR + wearable CGM data to predict glucose spikes 2 hours ahead", ha="center", fontsize=17, color=MUTED)
                ax.text(0.5, 0.3, "Digital Twin Challenge 2026 — Happiest Health", ha="center", fontsize=14, color=BLUE)
            elif body == "IMG":
                iax = fig.add_axes([0.04, 0.05, 0.92, 0.8]); iax.imshow(mpimg.imread(DOCS / "architecture.png")); iax.axis("off")
            else:
                y = 0.79
                for line in body:
                    ax.text(0.05, y, line, fontsize=16, color=INK, va="top"); y -= 0.095
            pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    architecture(); presentation(); print("docs written")
