# PharmaDrishti 💊
### A Digital Twin of India's Pharmaceutical Market — AI-Powered Drug Adoption Simulation

<div align="center">

**Digital Twin Challenge 2026 — Prototype & Code Submission**

**Transform pharmaceutical market research from months to minutes**

📽️ [Demo Video](#-demo-video) · 🏗️ [Architecture Diagram (PDF)](docs/Architecture_Diagram.pdf) · 📑 [Presentation (PDF)](docs/PharmaDrishti_Presentation.pdf) · ⚖️ [MIT License](LICENSE)

</div>

---

## 📌 Submission Checklist

| Item | Where |
| --- | --- |
| Team details | [Team Details](#-team-details) |
| College / Incubator information | [College / Incubator](#-college--incubator-information) |
| Project title | [Project Title](#-project-title) |
| Problem statement | [Problem Statement](#-problem-statement) |
| Healthcare use case | [Healthcare Use Case](#-healthcare-use-case) |
| Technical stack | [Technical Stack](#-technical-stack) |
| AI/ML model & framework details | [AI/ML Model Details](#-aiml-model--framework-details) |
| 15–20 min demo video (Unlisted YouTube) | [Demo Video](#-demo-video) |
| Open-source license | [License](#-open-source-license) · [LICENSE](LICENSE) |
| Architecture diagram (PDF) | [docs/Architecture_Diagram.pdf](docs/Architecture_Diagram.pdf) |
| Presentation (PDF) | [docs/PharmaDrishti_Presentation.pdf](docs/PharmaDrishti_Presentation.pdf) |
| Public accessibility | All files are in this public repository; the demo video is Unlisted (viewable by link, no sign-in needed) |

---

## 👥 Team Details

**Team Name:** PharmaDrishti

| Name | Role | Email | GitHub |
| --- | --- | --- | --- |
| Somya Upadhyay | Team Lead — ML & Data Engineering | `<email>` | [@Somyaaaaa23](https://github.com/Somyaaaaa23) |
| `<Member 2>` | `<Role>` | `<email>` | `<github>` |
| `<Member 3>` | `<Role>` | `<email>` | `<github>` |

---

## 🏫 College / Incubator Information

- **College / Institution:** `<College name>`
- **Department / Program:** `<Department>`
- **City, State:** `<City, State>`
- **Incubator (if any):** `<Incubator name / Not applicable>`

---

## 🏷️ Project Title

**PharmaDrishti — A Digital Twin of India's Pharmaceutical Market for Predicting Drug Adoption, Affordability and Access**

---

## ❗ Problem Statement

Launching a medicine in India is a high-risk decision made with slow, expensive and incomplete information:

- Traditional market research for a single drug launch takes **3–6 months** and costs **₹50L–₹1Cr+**.
- India's patient population is extremely heterogeneous — income, city tier, state, disease burden and insurance coverage (PM-JAY, CGHS, ESIC, private) all change whether a patient can and will adopt a drug.
- Wrong pricing or wrong target segments lead to **poor adoption, unaffordable therapy for patients, and failed launches**.
- There is no fast, explainable way to run **"what-if" launch scenarios** (price, coverage, manufacturing reach, safety profile) before committing capital.

---

## 🏥 Healthcare Use Case

PharmaDrishti builds a **digital twin of the Indian patient + pharmaceutical market**: 10,000 synthetic patient personas interacting with 5,000 simulated medicines (50M persona–medicine interactions), grounded in real NPPA price data and FDA Orange Book regulatory data.

Who uses it and for what:

1. **Pharma companies / launch teams** — predict adoption suitability of a new drug for each demographic segment in under 5 seconds; pick the best launch states and segments.
2. **Pricing & access teams** — simulate price points and see the adoption vs. affordability trade-off; find segments priced out of therapy.
3. **Policy makers / payers** — estimate how insurance-scheme coverage changes uptake and equity of access across city tiers.
4. **Regulatory & strategy teams** — use the knowledge graph to see patent/exclusivity horizons and competitor counts for an ingredient.

Example: a diabetes drug priced at ₹30K/batch shows ~82% adoption in Tier-1 high-income segments but a sharp drop in Tier-3, pointing to tiered pricing or local manufacturing partnerships.

---

## 🧰 Technical Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.10+ |
| ML / Modelling | XGBoost, scikit-learn |
| Explainability | SHAP |
| Knowledge Graph | NetworkX (Drug, Ingredient, Manufacturer, Patent, Exclusivity nodes) |
| Data processing | Pandas, NumPy |
| Data sources | NPPA price dataset (`nppaipdms.csv`), FDA Orange Book (products, patents, exclusivity), synthetic Indian healthcare personas |
| Frontend / Dashboard | Streamlit |
| Visualisation | Plotly, Matplotlib, Seaborn |
| GenAI insights | Google Gemini (optional, for natural-language recommendations) |

---

## 🤖 AI/ML Model & Framework Details

- **Task:** supervised regression — predict a 0–100 adoption suitability score for a (persona, medicine) pair.
- **Training data:** synthetic interaction engine (`generate_personas.py`, `medicine_generator.py`, `suitability_calculator.py`) — target built from a multi-factor score: disease match 30%, affordability 25%, market pricing 20%, manufacturing access 15%, risk tolerance 10%.
- **Models:**
  - **Baseline:** XGBoost Regressor on persona + medicine features (`train_model.py` → `adoption_model.pkl`).
  - **KG-enhanced:** same model plus graph-derived features (competitor count, patent/exclusivity protection horizon, manufacturer degree) from the regulatory knowledge graph (`knowledge_graph_builder.py`, `graph_feature_extractor.py`, `kg_enhanced_training.py`).
- **Feature engineering:** leakage-safe feature selection (`feature_selection.py`), feature fusion (`feature_fusion.py`), IQR outlier flagging (`outlier_detector.py`).
- **Validation:** train/test gap, 5-fold cross-validation, learning curves, target-shuffle leakage test, residual analysis (`overfitting_diagnostics.py`); model benchmarking (`benchmark_models.py`).
- **Explainability:** SHAP feature attribution + rule-based decision layer (`decision_layer.py`) and optional Gemini-generated recommendations.
- **Performance (synthetic validation):** R² > 0.85, prediction latency < 5 s per scenario across all segments.

---

## 📽️ Demo Video

▶️ **15–20 minute demo (Unlisted YouTube):** `<PASTE YOUTUBE LINK HERE>`

---

## 🏗️ Architecture Diagram

📄 **PDF:** [docs/Architecture_Diagram.pdf](docs/Architecture_Diagram.pdf)

![PharmaDrishti Architecture](docs/architecture.png)

---

## 📑 Presentation

📄 **PDF:** [docs/PharmaDrishti_Presentation.pdf](docs/PharmaDrishti_Presentation.pdf) — problem, healthcare use case, solution approach, architecture, tech stack, outcomes & results, limitations and roadmap.

---

## ⚖️ Open-source License

This project is released under the **MIT License** — see [LICENSE](LICENSE). Third-party datasets (NPPA, FDA Orange Book) remain subject to their original public terms.

---

## 🎯 Executive Summary

PharmaDrishti is a simulation-driven pharmaceutical market intelligence system designed to estimate product adoption probability across heterogeneous demographic and economic segments in India.

The system integrates:

- Large-scale synthetic persona modeling
- Multi-factor adoption scoring
- Regulatory intelligence from FDA Orange Book datasets
- Market intelligence from NPPA dataset
- Knowledge graph feature augmentation
- Gradient-boosted regression (XGBoost)
- SHAP-based feature attribution
- Scenario-based pricing simulation

Predictions are generated in under 5 seconds using a pre-trained model calibrated on large-scale simulated interaction data.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run dashboard
streamlit run dashboard.py
```

**Dashboard opens at:** `http://localhost:8501`

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎯 Multi-Factor Analysis
- **Disease Match** (30%)
- **Affordability** (25%)
- **Market Pricing** (20%)
- **Manufacturing** (15%)
- **Risk Tolerance** (10%)

</td>
<td width="50%">

### 🤖 AI-Powered Insights
- Explains WHY scores are what they are
- Identifies specific barriers
- Actionable recommendations
- Expected impact predictions

</td>
</tr>
<tr>
<td width="50%">

### 📊 Interactive Dashboard
- Real-time predictions (< 5 sec)
- Visual analytics & charts
- Scenario comparison
- CSV export for presentations

</td>
<td width="50%">

### 🔬 Robust ML Model
- Baseline + KG-enhanced models
- Leakage-safe feature selection
- Overfitting/leakage diagnostics
- XGBoost algorithm
- Explainable AI (SHAP)

</td>
</tr>
</table>

---

## 💡 How It Works

```mermaid
graph LR
    A[NPPA + Personas + Medicines] --> B[Ingredient Mapping]
    B --> C[FDA Orange Book Integration]
    C --> D[Knowledge Graph]
    D --> E[Graph Feature Extraction]
    E --> F[Suitability + XGBoost]
    F --> G[Adoption Prediction + Insights]
```

### 1. Configure Your Medicine
- Target disease
- Batch price (₹5K-₹500K)
- Manufacturing score
- Brand strength
- Side effect risk
- Insurance compatibility under schemes

### 2. AI Analysis
- Analyzes across persona segments
- Builds regulatory-market knowledge graph
- Calculates suitability and graph-augmented predictions

### 3. Get Results
- Overall suitability score
- Best target segments
- Revenue estimates
- AI-powered recommendations

---
## 📊 Data Simulation Framework

The training dataset is generated using a large-scale synthetic interaction engine:

- 10,000 demographic personas
- 5,000 simulated pharmaceutical products
- 50,000,000 persona–medicine interactions

Each interaction encodes:

- Disease relevance
- Price-to-income ratio
- Insurance coverage compatibility
- Manufacturing accessibility
- Brand strength
- Risk tolerance alignment

The resulting dataset is used to train a supervised regression model to predict adoption suitability score.

### Regulatory + Knowledge Graph Layer

PharmaDrishti integrates:

- `nppaipdms.csv` (market data)
- `standardized_data/products_standardized.csv`
- `standardized_data/patents_standardized.csv`
- `standardized_data/exclusivity_standardized.csv`
- `standardized_data/reference_standardized.csv`

Knowledge graph nodes include Drug, Ingredient, Manufacturer, Patent, and Exclusivity entities. Graph-derived features (e.g., competitor count and protection horizon) are appended to training data for the KG-enhanced model.

---

## 🧪 Training & Diagnostics Workflow

### 1) Baseline model training

```bash
python train_model.py
```

### 2) KG-enhanced training (expanded personas + full medicines)

```bash
python - <<'PY'
from kg_enhanced_training import run_pipeline
run_pipeline(
    base_dir='.',
    persona_target_count=1000,
    persona_file='indian_healthcare_personas_expanded.json',
    n_personas=None,
    n_medicines=5000,
    outlier_method='iqr',
    outlier_strategy='flag',
)
PY
```

### 3) Overfitting / leakage diagnostics

```bash
python overfitting_diagnostics.py --mode baseline --base-dir . --model-path models/adoption_model.pkl --persona-file indian_healthcare_personas_expanded.json --n-personas 1000 --n-medicines 1000
python overfitting_diagnostics.py --mode kg_enhanced --base-dir . --model-path models/adoption_model_kg_enhanced.pkl --persona-file indian_healthcare_personas_expanded.json --n-personas 1000 --n-medicines 1000
```

Diagnostics include train/test gap, 5-fold CV, learning curves, target-shuffle leakage test, feature correlations, SHAP importance, and residual analysis.

---

## 🖼️ Research Outputs

Generated in `outputs/`:

- `graph_summary_metrics.png` (Table 1-ready summary)
- `table2_ablation_study.png`
- `table3_dataset_overview.png`
- `plot4_degree_distribution.png`
- `plot6_model_performance_comparison.png`
- `research_paper_sections.txt`
- `diagnostics/baseline/overfitting_diagnostic_report.json`
- `diagnostics/kg_enhanced/overfitting_diagnostic_report.json`

---


### ROI Example
**Mid-Size Pharma Company** (5 products/year)
- Traditional Cost: ₹5Cr
- PharmaDrishti: ₹25L
- **Annual Savings: ₹4.75Cr**
- **ROI: 1,900%**

---

## 🎓 Use Cases

### 1. Batch Pricing Optimization
Test multiple price points to maximize revenue vs adoption

**Example:**
- ₹30K: 82% adoption, ₹24.6Cr revenue
- ₹80K: 51% adoption, ₹40.8Cr revenue
- **Recommendation:** Tiered pricing by city tier

### 2. Manufacturing Assessment
Evaluate manufacturing capability for target markets

**Example:**
- Tier 1: 68% adoption (high manufacturing access)
- Tier 3: 23% adoption (manufacturing barrier)
- **Recommendation:** Partner with local manufacturers

### 3. Market Segmentation
Identify optimal target demographics for phased launch

**Example:**
- Tier 1, High Income: 85% adoption
- Tier 2, Middle Income: 62% adoption
- **Recommendation:** Start Tier 1, expand after 6 months

---

**Model Performance (Synthetic Validation):**
- R² Score: > 0.85
- Training Time: 15–30 minutes
- Memory Requirement: 8–16GB RAM
- Model Size: ~50–100 MB

---

## 📊 Dashboard Guide

### Input Parameters

| Parameter                   | Range     | Description                  |
| --------------------------- | --------- | ---------------------------- |
| **Target Disease**          | Dropdown  | Disease medicine treats      |
| **Batch Price**             | ₹5K-₹500K | Medicine batch price         |
| **Launch Locality**         | Dropdown  | State or Pan-India           |
| **Brand Strength**          | 0.0-1.0   | Generic to Premium           |
| **Side Effect Risk**        | 0.0-1.0   | Safe to High Risk            |
| **Manufacturing Score**     | 0.0-1.0   | Limited to High Capability   |
| **Insurance under Schemes** | 0.0-1.0   | Not Covered to Fully Covered |

### Score Interpretation

| Score Range | Interpretation                      |
| ----------- | ----------------------------------- |
| 70–100      | High predicted adoption suitability |
| 40–70       | Moderate suitability                |
| 0–40        | Low suitability                     |

---

### Core Modules

1. Persona Generator – Constructs demographic and economic distributions
2. Interaction Engine – Simulates adoption score per persona-product pair
3. Regulatory Loader – Ingests NPPA + Orange Book datasets
4. Knowledge Graph Builder – Creates regulatory-market graph
5. Graph Feature Extractor – Derives competition/protection features
6. Model Trainer – Trains baseline and KG-enhanced XGBoost regressors
7. Diagnostics Engine – Overfitting and leakage checks
8. Prediction Engine – Real-time inference

```
┌─────────────────────────────────────┐
│      Streamlit Dashboard            │
│                                     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Persona Manager | Prediction Engine│
│  Insights Generator (Gemini AI)     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  10K Personas | XGBoost Model       │
│  50M Interactions                   │
└─────────────────────────────────────┘
```

### Technology Stack
- **Frontend:** Streamlit 
- **ML:** XGBoost, SHAP
- **AI:** Google Gemini
- **Data:** Pandas, NumPy
- **Viz:** Plotly

---

## 📚 Documentation

| Document                                | Description                  |
| --------------------------------------- | ---------------------------- |
| **README.md**                           | This file - overview         |
| **DASHBOARD_SETUP.md**                  | Detailed setup guide         |
| **FEATURES.md**                         | Complete feature docs        |
| **QUICK_REFERENCE.md**                  | Quick reference card         |
| **METRICS_AND_FORMULAS.md**             | Scoring metrics and formulas |
| **outputs/research_paper_sections.txt** | Paper-ready section draft    |

---

## 🎯 System Metrics

| Metric                 | Value                      |
| ---------------------- | -------------------------- |
| Personas               | 10,000                     |
| Simulated Interactions | 50,000,000                 |
| Diseases Modeled       | 10                         |
| Model Type             | XGBoost Regressor          |
| Validation R²          | > 0.85 (synthetic dataset) |
| Training Time          | 15–30 minutes              |
| Prediction Latency     | < 5 seconds                |
| Model Size             | ~50–100 MB                 |
| Geographic Modeling    | 10 states, 25+ cities      |

---

## 🔧 Troubleshooting

### Dashboard Won't Start
```bash
pip install -r requirements.txt
streamlit run dashboard.py --server.port 8502
```

### AI Insights Missing
- Configure Gemini API key
- Check internet connection
- Verify key at [Google AI Studio](https://makersuite.google.com/app/apikey)

### Model Training Fails
- Ensure 16GB+ RAM
- Check disk space (5GB+)
- Reduce dataset size temporarily

---

## 🗺️ Roadmap

### ✅ Phase 1: MVP (Completed)
- [x] 10,000 diverse personas
- [x] 50M interaction dataset
- [x] XGBoost prediction model
- [x] Interactive dashboard
- [x] AI-powered insights

### 📅 Phase 2: Q1 2026
- [ ] 20,000+ personas
- [ ] 20+ disease categories
- [ ] NPPA pricing integration
- [ ] Competitive analysis
- [ ] Mobile responsive design

### 📅 Phase 3: Q2 2026
- [ ] Pilot with pharma companies
- [ ] Real launch validation
- [ ] Case studies
- [ ] API access

### 📅 Phase 4: Q3-Q4 2026
- [ ] SaaS platform
- [ ] Enterprise features
- [ ] Mobile app
- [ ] Multi-language support


---

## ⚠️ Limitations

- Current implementation relies on synthetic data simulation
- Does not model physician prescription behavior
- Regulatory coverage depends on source data completeness and mapping quality
- No real-world post-launch validation integrated yet
- Economic and insurance parameters are modeled assumptions

---


