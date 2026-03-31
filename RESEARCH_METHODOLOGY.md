# PharmaDrishti: Methodology

## AI-Powered Pharmaceutical Market Intelligence Using Synthetic Persona Modeling and Gradient-Boosted Regression

---

## Abstract

PharmaDrishti is a simulation-driven pharmaceutical market intelligence system designed to estimate
product adoption probability across heterogeneous demographic and economic segments in India. The
system integrates large-scale synthetic persona modeling, multi-factor adoption scoring, regulatory
intelligence from FDA Orange Book and NPPA datasets, knowledge graph feature augmentation, and
gradient-boosted regression (XGBoost). Predictions are generated in under 5 seconds using a
pre-trained model calibrated on 50,000,000 simulated persona–medicine interaction pairs. This
methodology document describes the complete pipeline from data generation to model deployment.

---

## 1. Introduction

Pharmaceutical market research in India traditionally requires 6–12 months and costs ₹50L–₹2Cr per
product. PharmaDrishti reduces this to 2–3 seconds at a fraction of the cost by replacing field
surveys with a statistically grounded synthetic simulation engine. The system models India's
demographic diversity across 10 states, 25+ cities, 5 income strata, and 10 chronic disease
categories, producing adoption probability scores that guide pricing, launch locality, and
segmentation decisions.

The core research questions addressed are:
1. Can synthetic persona modeling accurately simulate pharmaceutical adoption behavior?
2. Does integrating real market pricing data (NPPA) improve prediction quality?
3. Can a gradient-boosted model trained on simulated data generalize to real-world segments?

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│                   Streamlit Web Dashboard                        │
│              (Real-time Predictions, Visual Analytics)           │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       APPLICATION LAYER                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  Data Loader    │  │  Suitability     │  │  AI Insights   │  │
│  │  (Personas +    │  │  Calculator      │  │  (Gemini API)  │  │
│  │   Market Data)  │  │  (5-Component)   │  │                │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  Medicine       │  │  Knowledge Graph │  │  Feature       │  │
│  │  Generator      │  │  Builder         │  │  Extractor     │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  10K Personas   │  │  XGBoost Model   │  │  NPPA Market   │  │
│  │  (JSON)         │  │  (adoption_model │  │  Data          │  │
│  │                 │  │   .pkl)          │  │  (NPPAIPDMS)   │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```


---

## 3. Data Collection and Synthetic Data Generation

### 3.1 Overview

The absence of large-scale, labeled pharmaceutical adoption datasets in the Indian context necessitated
a synthetic data generation approach. Two primary datasets were constructed: (1) a persona dataset
representing 10,000 Indian healthcare consumers, and (2) a medicine dataset of 5,000 synthetic
pharmaceutical products. Their Cartesian product yields 50,000,000 interaction pairs used for
supervised model training.

```
┌──────────────────────────────────────────────────────────────────┐
│                  SYNTHETIC DATA GENERATION FLOW                  │
│                                                                  │
│   ┌──────────────┐        ┌──────────────────┐                  │
│   │  Demographic │        │  Disease          │                  │
│   │  Distributions│       │  Prevalence Data  │                  │
│   │  (Census-    │        │  (ICMR / WHO)     │                  │
│   │   informed)  │        │                  │                  │
│   └──────┬───────┘        └────────┬─────────┘                  │
│          │                         │                             │
│          ▼                         ▼                             │
│   ┌──────────────────────────────────────────┐                  │
│   │         Persona Generator (n=10,000)     │                  │
│   │   Age · Gender · State · City Tier       │                  │
│   │   Income · Occupation · Insurance        │                  │
│   │   Chronic Diseases · Price Sensitivity   │                  │
│   └──────────────────────┬───────────────────┘                  │
│                          │                                       │
│   ┌──────────────────────┼───────────────────┐                  │
│   │  Medicine Generator  │   (n=5,000)        │                  │
│   │  Disease · Price     │   Risk · Mfg Score │                  │
│   └──────────────────────┘                   │                  │
│                          │                   │                  │
│                          ▼                   ▼                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │     Interaction Engine: 10,000 × 5,000 = 50M pairs       │  │
│   │     Suitability Score per pair (5-component formula)     │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Persona Generation

#### 3.2.1 Demographic Modeling

The persona generator (`generate_personas.py`) produces 10,000 synthetic Indian healthcare consumers
using statistically grounded distributions. A fixed random seed (42) ensures reproducibility.

**Age Distribution:**
Age is sampled from a Beta distribution to produce a realistic working-age population:

```
Age ~ Beta(α=2, β=2) × 42 + 18    →    Age ∈ [18, 60]
```

This yields a bell-shaped distribution centered around 39 years, reflecting India's working-age
demographic profile.

**Geographic Distribution:**
Ten Indian states are modeled with state-specific city-tier probability vectors:

| State          | Tier 1 Prob | Tier 2 Prob | Tier 3 Prob |
|----------------|-------------|-------------|-------------|
| Delhi          | 0.90        | 0.10        | 0.00        |
| Maharashtra    | 0.40        | 0.35        | 0.25        |
| Karnataka      | 0.35        | 0.40        | 0.25        |
| Uttar Pradesh  | 0.15        | 0.30        | 0.55        |
| Rajasthan      | 0.20        | 0.35        | 0.45        |

City tier is sampled as:

```
Tier ~ Categorical([p_tier1, p_tier2, p_tier3])
```

#### 3.2.2 Income Modeling

Income level is conditioned on city tier and age. Tier 1 cities have a higher probability of
upper-income strata; Tier 3 cities skew toward lower income. An age factor adjusts probabilities:

```
age_factor = min(1.0, (age - 18) / 30)

Tier 1 income_probs = [0.10, 0.20, 0.35, 0.25, 0.10]  (Low → High)
Tier 3 income_probs = [0.35, 0.35, 0.20, 0.08, 0.02]
```

Annual income within each stratum is drawn from a log-normal distribution:

```
Annual_Income ~ LogNormal(μ = log((min + max) / 2), σ = 0.3)
```

Income ranges by stratum:

| Income Level  | Range (INR/year)        |
|---------------|-------------------------|
| Low           | ₹1,20,000 – ₹2,00,000  |
| Lower-Middle  | ₹2,00,000 – ₹5,00,000  |
| Middle        | ₹5,00,000 – ₹10,00,000 |
| Upper-Middle  | ₹10,00,000 – ₹25,00,000|
| High          | ₹25,00,000 – ₹76,00,000|


#### 3.2.3 Disease Prevalence Modeling

Chronic disease assignment is age-dependent. Each disease has a base prevalence rate; the actual
probability is scaled by an age factor:

```
age_factor = (age - 18) / 42    →    0 at age 18, 1 at age 60

P(disease | age) = base_prevalence × (0.5 + 1.5 × age_factor)
```

Disease base prevalence rates modeled:

| Disease                  | Base Prevalence |
|--------------------------|-----------------|
| Hypertension             | 25%             |
| Obesity                  | 20%             |
| Anemia                   | 18%             |
| Arthritis                | 15%             |
| Thyroid Disorder         | 10%             |
| Type 2 Diabetes          | 12%             |
| Asthma                   | 8%              |
| COPD                     | 5%              |
| Heart Disease            | 6%              |
| Chronic Kidney Disease   | 4%              |

A persona may have multiple comorbidities. The number of chronic conditions directly influences
monthly healthcare spend:

```
base_spend_pct = 0.02 + (0.03 × no_of_conditions / 5)
Monthly_Healthcare_Spend = Monthly_Income × base_spend_pct
```

#### 3.2.4 Price Sensitivity Modeling

Price sensitivity is inversely correlated with income level and includes Gaussian noise:

```
income_percentile = income_level_index / 4    →    [0, 0.25, 0.5, 0.75, 1.0]

Price_Sensitivity = 1.0 - income_percentile + N(0, 0.1)
Price_Sensitivity = clip(Price_Sensitivity, 0.0, 1.0)
```

#### 3.2.5 Digital Health Adoption

Digital adoption is modeled as a composite score of age, city tier, and income:

```
digital_score = (1 - age/60) × 0.4
              + (1 - tier_index/2) × 0.3
              + income_percentile × 0.3
              + N(0, 0.1)

Adoption Level:
  digital_score > 0.7  →  "High"
  digital_score > 0.4  →  "Medium"
  digital_score > 0.2  →  "Low"
  else                 →  "None"
```

---

### 3.3 Medicine Generation

The medicine generator (`medicine_generator.py`) produces 5,000 synthetic pharmaceutical products
with realistic statistical distributions.

```
┌──────────────────────────────────────────────────────────────────┐
│                   MEDICINE GENERATION PIPELINE                   │
│                                                                  │
│  Input: n = 5,000                                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  For each medicine i:                                      │  │
│  │                                                            │  │
│  │  medicine_id  = "MED_{i:05d}"                             │  │
│  │                                                            │  │
│  │  target_disease ~ Uniform({10 disease categories})        │  │
│  │                                                            │  │
│  │  batch_price   ~ LogNormal(μ=10, σ=1.5)                   │  │
│  │                  clipped to [₹5,000, ₹5,00,000]           │  │
│  │                                                            │  │
│  │  side_effect_risk ~ Beta(α=2, β=5)  [skewed low]          │  │
│  │                                                            │  │
│  │  manufacturing_score ~ Uniform(0, 1)                      │  │
│  │                                                            │  │
│  │  insurance_compatibility ~ Uniform(0, 1)                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Batch Price Distribution:**
The log-normal distribution (μ=10, σ=1.5) captures the right-skewed nature of pharmaceutical
pricing, where most medicines are moderately priced but a long tail of high-cost specialty drugs
exists. The clipping to [₹5,000, ₹5,00,000] reflects realistic Indian batch pricing bounds.

**Side Effect Risk Distribution:**
Beta(α=2, β=5) is chosen because it is skewed toward lower values (mean ≈ 0.286), reflecting the
regulatory reality that most approved medicines have low-to-moderate side effect profiles.

---

## 4. Feature Engineering

### 4.1 Feature Set

The interaction dataset encodes 25+ features per persona–medicine pair, organized into four groups:

**Persona Features (15):**

| Feature                    | Type        | Description                              |
|----------------------------|-------------|------------------------------------------|
| age                        | Numeric     | Persona age (18–60)                      |
| gender_encoded             | Binary      | Female=0, Male=1                         |
| income                     | Numeric     | Annual income (INR)                      |
| income_level_encoded       | Ordinal     | Low=1 to High=5                          |
| city_tier_encoded          | Ordinal     | Tier 1=1, Tier 2=2, Tier 3=3            |
| chronic_condition_count    | Numeric     | Number of chronic diseases               |
| price_sensitivity          | Float [0,1] | Price sensitivity score                  |
| monthly_healthcare_spend   | Numeric     | Monthly healthcare expenditure (INR)     |
| persona_disease_encoded    | Ordinal     | Primary disease label-encoded            |
| insurance_sum_assured      | Numeric     | Insurance coverage amount (INR)          |
| insured_flag               | Binary      | 0=Uninsured, 1=Insured                   |
| last_doctor_visit_months   | Numeric     | Months since last doctor visit           |
| digital_adoption_encoded   | Ordinal     | None=0, Low=1, Medium=2, High=3         |
| log_income                 | Float       | log(1 + annual_income)                   |
| income_to_insurance_ratio  | Float       | insurance_sum / annual_income            |

**Medicine Features (5):**

| Feature                    | Type        | Description                              |
|----------------------------|-------------|------------------------------------------|
| price                      | Numeric     | Batch price (INR)                        |
| side_effect_risk           | Float [0,1] | Side effect risk score                   |
| manufacturing              | Float [0,1] | Manufacturing capability score           |
| insurance_compatibility    | Float [0,1] | Insurance scheme compatibility           |
| medicine_disease_encoded   | Ordinal     | Target disease label-encoded             |

**Interaction / Derived Features (8):**

| Feature                         | Formula                                          |
|---------------------------------|--------------------------------------------------|
| log_price                       | log(1 + batch_price)                             |
| price_to_income_ratio           | batch_price / annual_income                      |
| price_to_monthly_income_ratio   | batch_price / (annual_income / 12)               |
| price_to_healthcare_spend_ratio | batch_price / monthly_healthcare_spend           |
| age_condition_interaction       | age × no_of_chronic_conditions                   |
| price_sensitivity_x_income      | price_sensitivity × log(1 + annual_income)       |
| insurance_price_interaction     | insurance_compatibility × batch_price            |

**Computed Score Features (4):**

| Feature             | Description                                      |
|---------------------|--------------------------------------------------|
| disease_match       | Binary: 1 if medicine targets persona's disease  |
| affordability       | Affordability score (0–1)                        |
| manufacturing_match | Manufacturing score × location factor            |
| risk_score          | side_effect_risk × price_sensitivity             |


---

## 5. Suitability Scoring Formula

### 5.1 Overview

The suitability score is the target variable for model training. It is computed analytically for
every persona–medicine pair using a 5-component weighted formula followed by a sigmoid
transformation. This deterministic formula encodes domain knowledge about pharmaceutical adoption
drivers in the Indian market.

```
┌──────────────────────────────────────────────────────────────────┐
│                  SUITABILITY SCORING PIPELINE                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Component 1: Disease Match          Weight = 0.30       │   │
│  │  Component 2: Affordability          Weight = 0.25       │   │
│  │  Component 3: Market Pricing         Weight = 0.20       │   │
│  │  Component 4: Manufacturing Access   Weight = 0.15       │   │
│  │  Component 5: Risk Tolerance         Weight = 0.10       │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Weighted_Sum = Σ(weight_i × component_i)                │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Scaled_Sum = (Weighted_Sum - 0.5) × 12                  │   │
│  │  Suitability = σ(Scaled_Sum) = 1 / (1 + e^(-Scaled_Sum))│   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Master Formula

```
Suitability_Score = σ((W_sum - 0.5) × 12)

where:

W_sum = 0.30 × Disease_Match
      + 0.25 × Affordability
      + 0.20 × Market_Pricing
      + 0.15 × Manufacturing
      + 0.10 × Risk_Tolerance

σ(x) = 1 / (1 + e^(-x))
```

### 5.3 Component Formulas

#### Component 1: Disease Match (30%)

Binary indicator of therapeutic relevance:

```
Disease_Match = 1.0   if target_disease ∈ persona_chronic_diseases
              = 0.0   otherwise
```

This is the highest-weighted component, reflecting the clinical primacy of disease relevance in
adoption decisions.

#### Component 2: Affordability (25%)

Measures the medicine's price relative to the persona's purchasing power:

```
Monthly_Income = Annual_Income_INR / 12

Affordability = clip(1.0 - (Batch_Price / (Monthly_Income × 100)), 0.0, 1.0)
```

The denominator `Monthly_Income × 100` represents a threshold of 100 months of income, beyond
which affordability reaches zero. This captures the reality that batch prices are typically
amortized across multiple months of use.

#### Component 3: Market Pricing Competitiveness (20%)

Evaluates price positioning relative to NPPA market data:

```
Price_Ratio = Batch_Price / Avg_Market_Price_for_Disease

Market_Pricing_Score:
  Price_Ratio < 0.7          →  1.0
  0.7 ≤ Price_Ratio < 1.3   →  1.0 - (Price_Ratio - 0.7) × (0.4 / 0.6)
  1.3 ≤ Price_Ratio < 2.0   →  0.6 - (Price_Ratio - 1.3) × (0.4 / 0.7)
  Price_Ratio ≥ 2.0          →  0.1
```

If no market data is available for the disease category, a neutral score of 0.5 is assigned.

#### Component 4: Manufacturing Access (15%)

Combines the medicine's manufacturing capability with the persona's geographic accessibility:

```
Tier_Accessibility = { Tier 1: 0.9, Tier 2: 0.7, Tier 3: 0.5 }

Manufacturing = Medicine_Manufacturing_Score × Tier_Accessibility[City_Tier]
```

#### Component 5: Risk Tolerance (10%)

Models the interaction between side effect risk and the persona's risk aversion (proxied by price
sensitivity):

```
Risk_Score = Side_Effect_Risk × Price_Sensitivity

Risk_Tolerance = 1 - Risk_Score
```

### 5.4 Sigmoid Transformation

The weighted sum is centered at 0.5 and scaled by 12 before applying the sigmoid function. This
scaling ensures that the sigmoid operates in a sensitive region, producing well-separated
probabilities rather than clustering near 0.5:

```
Scaled_Sum = (W_sum - 0.5) × 12

Suitability = 1 / (1 + e^(-Scaled_Sum))
```

**Worked Example:**

```
Input:
  Persona: Type 2 Diabetes, ₹6,00,000/yr income, Tier 1, price_sensitivity=0.4
  Medicine: Type 2 Diabetes, ₹50,000 batch, mfg_score=0.8, side_effect_risk=0.3
  Market:   Avg price for diabetes medicines = ₹45,000

Step 1 – Disease Match:
  "type 2 diabetes" ∈ "type 2 diabetes"  →  1.0

Step 2 – Affordability:
  Monthly_Income = 6,00,000 / 12 = 50,000
  Affordability = 1.0 - (50,000 / (50,000 × 100)) = 1.0 - 0.01 = 0.99

Step 3 – Market Pricing:
  Price_Ratio = 50,000 / 45,000 = 1.11
  Score = 1.0 - (1.11 - 0.7) × (0.4/0.6) = 1.0 - 0.273 = 0.727

Step 4 – Manufacturing:
  Score = 0.8 × 0.9 (Tier 1) = 0.72

Step 5 – Risk Tolerance:
  Risk_Score = 0.3 × 0.4 = 0.12
  Risk_Tolerance = 1 - 0.12 = 0.88

Weighted Sum:
  W_sum = (0.30×1.0) + (0.25×0.99) + (0.20×0.727) + (0.15×0.72) + (0.10×0.88)
        = 0.300 + 0.2475 + 0.1454 + 0.108 + 0.088
        = 0.8889

Sigmoid:
  Scaled = (0.8889 - 0.5) × 12 = 4.667
  Suitability = 1 / (1 + e^(-4.667)) ≈ 0.991  →  99.1%
```


---

## 6. Machine Learning Pipeline

### 6.1 Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     ML TRAINING PIPELINE                         │
│                                                                  │
│  [1] Load 10,000 Personas                                        │
│       └─► JSON → DataFrame                                       │
│                                                                  │
│  [2] Generate 5,000 Medicines                                    │
│       └─► LogNormal prices, Beta risk, Uniform mfg              │
│                                                                  │
│  [3] Create Interaction Dataset                                  │
│       └─► 10,000 × 5,000 = 50,000,000 pairs                     │
│       └─► Compute suitability score per pair                     │
│       └─► Extract 25+ features per pair                         │
│                                                                  │
│  [4] Feature Matrix Construction                                 │
│       └─► build_model_matrix() → X (features), y (scores)       │
│                                                                  │
│  [5] Train / Validation / Test Split                             │
│       └─► 80% train | 10% validation | 10% test                 │
│                                                                  │
│  [6] XGBoost Regressor Training                                  │
│       └─► eval_set on validation, early stopping                │
│                                                                  │
│  [7] Evaluation                                                  │
│       └─► R², RMSE, MAE on test set                             │
│                                                                  │
│  [8] Artifact Persistence                                        │
│       └─► adoption_model.pkl, feature_names.pkl                 │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Dataset Split

The 50M interaction dataset is partitioned as follows:

```
Total Dataset: 50,000,000 pairs
│
├── Train Set (80%):  40,000,000 pairs
│     ├── Fit Set (90% of train):  36,000,000 pairs  ← model training
│     └── Validation Set (10%):    4,000,000 pairs   ← early stopping
│
└── Test Set (20%):  10,000,000 pairs  ← final evaluation
```

All splits use `random_state=42` for reproducibility.

### 6.3 XGBoost Hyperparameters

| Hyperparameter    | Value | Rationale                                          |
|-------------------|-------|----------------------------------------------------|
| n_estimators      | 200   | Sufficient for 50M samples; prevents underfitting  |
| max_depth         | 8     | Captures complex interaction patterns              |
| learning_rate     | 0.05  | Conservative rate for stable convergence           |
| subsample         | 0.8   | Reduces overfitting via row subsampling            |
| colsample_bytree  | 0.8   | Reduces overfitting via feature subsampling        |
| random_state      | 42    | Reproducibility                                    |
| objective         | reg:squarederror | Regression on continuous [0,1] scores |

### 6.4 Evaluation Metrics

Model performance is assessed using three complementary metrics:

**Coefficient of Determination (R²):**
```
R² = 1 - (SS_res / SS_tot)

where:
  SS_res = Σ(y_i - ŷ_i)²    (residual sum of squares)
  SS_tot = Σ(y_i - ȳ)²      (total sum of squares)

Target: R² > 0.85
```

**Root Mean Squared Error (RMSE):**
```
RMSE = √(1/n × Σ(y_i - ŷ_i)²)
```

**Mean Absolute Error (MAE):**
```
MAE = 1/n × Σ|y_i - ŷ_i|
```

Reported performance on the synthetic validation dataset:

| Metric | Value     |
|--------|-----------|
| R²     | > 0.85    |
| RMSE   | < 0.05    |
| MAE    | < 0.03    |

### 6.5 Knowledge Graph Enhanced Model

In addition to the baseline model, PharmaDrishti implements a KG-enhanced variant that augments
the feature set with graph-derived regulatory intelligence:

```
┌──────────────────────────────────────────────────────────────────┐
│              KNOWLEDGE GRAPH FEATURE AUGMENTATION               │
│                                                                  │
│  Nodes:                                                          │
│    Drug ──── Ingredient ──── Manufacturer                       │
│      │                           │                              │
│    Patent ──────────────── Exclusivity                          │
│                                                                  │
│  Data Sources:                                                   │
│    - NPPAIPDMS (953 medicines, 2013–2023)                       │
│    - FDA Orange Book (products, patents, exclusivity)           │
│    - Standardized CSVs (products, patents, exclusivity,         │
│                          reference listed drugs)                │
│                                                                  │
│  Graph-Derived Features:                                        │
│    - competitor_count: # of similar drugs in market            │
│    - protection_horizon: months of patent/exclusivity left      │
│    - market_concentration: Herfindahl index for disease         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Market Pricing Integration

### 7.1 NPPAIPDMS Dataset

The National Pharmaceutical Pricing Authority's IPDMS dataset provides real-world market pricing
intelligence. The dataset contains 953 medicine records spanning 2013–2023 with the following
schema (post-standardization):

| Column          | Type    | Description                          |
|-----------------|---------|--------------------------------------|
| id              | Integer | Record identifier                    |
| year            | Integer | Year of price notification           |
| medicine_name   | String  | Medicine name                        |
| formulation     | String  | Dosage form and strength             |
| notification    | String  | NPPA notification reference          |
| price_numeric   | Float   | Standardized price (INR)             |

### 7.2 Price Competitiveness Scoring

For each medicine being evaluated, the system:
1. Extracts disease keywords from the target disease string
2. Queries the NPPAIPDMS dataset for medicines with matching keywords
3. Computes the average market price from matched records
4. Applies the piecewise linear scoring function (Section 5.3, Component 3)

This integration penalizes overpriced medicines and rewards competitive pricing, directly
influencing the suitability score and, by extension, the model's training signal.

---

## 8. Explainability: SHAP Analysis

PharmaDrishti integrates SHAP (SHapley Additive exPlanations) to provide feature-level attribution
for each prediction:

```
SHAP Value for feature j:
  φ_j = Σ_{S ⊆ F\{j}} [|S|!(|F|-|S|-1)! / |F|!] × [f(S∪{j}) - f(S)]

where F = full feature set, S = feature subset
```

SHAP values quantify each feature's marginal contribution to the prediction, enabling:
- Identification of the primary adoption barrier for a specific medicine
- Segment-level feature importance analysis
- Actionable recommendations (e.g., "reduce price by 20% to improve affordability score")

---

## 9. Dashboard and Deployment

### 9.1 Prediction Latency

At inference time, the dashboard:
1. Accepts medicine parameters from the user (disease, price, manufacturing score, etc.)
2. Loads the pre-trained XGBoost model (`adoption_model.pkl`)
3. Constructs feature vectors for all 10,000 personas
4. Runs batch inference
5. Aggregates results into segment-level analytics

End-to-end prediction latency: < 5 seconds for 10,000 personas.

### 9.2 AI Insights Generation

Post-prediction, the system passes the top-level results to Google Gemini Pro API, which generates
natural language explanations of:
- Why the overall score is high/low
- Which segments are most/least receptive
- Specific actionable recommendations with expected impact

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

1. Synthetic data dependency: The model is trained entirely on simulated interactions. Real-world
   adoption behavior may deviate from the modeled distributions.

2. No physician behavior modeling: Prescription decisions by doctors are not captured; the model
   assumes direct consumer adoption.

3. Static disease prevalence: Disease probabilities are fixed constants; temporal trends (e.g.,
   rising diabetes prevalence) are not modeled.

4. Regulatory coverage gaps: Knowledge graph completeness depends on source data quality and
   ingredient-name mapping accuracy.

5. No post-launch validation: The system has not yet been validated against real pharmaceutical
   launch outcomes.

### 10.2 Future Work

| Phase    | Timeline | Planned Enhancement                              |
|----------|----------|--------------------------------------------------|
| Phase 2  | Q1 2026  | 20,000+ personas, 20+ disease categories         |
| Phase 2  | Q1 2026  | Full NPPA pricing integration                    |
| Phase 3  | Q2 2026  | Pilot validation with pharma company launch data |
| Phase 3  | Q2 2026  | Physician prescription behavior layer            |
| Phase 4  | Q3 2026  | SaaS platform, REST API, mobile app              |
| Phase 4  | Q4 2026  | Multi-language support, regional dialect NLP     |

---

## 11. References

1. National Pharmaceutical Pricing Authority (NPPA). *IPDMS Price Data 2013–2023*. Government of
   India, Ministry of Chemicals and Fertilizers.

2. FDA Orange Book: *Approved Drug Products with Therapeutic Equivalence Evaluations*. U.S. Food
   and Drug Administration.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the
   22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

4. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
   *Advances in Neural Information Processing Systems*, 30.

5. Indian Council of Medical Research (ICMR). *India: Health of the Nation's States*. Public
   Health Foundation of India, 2017.

6. Census of India 2011. *Primary Census Abstract*. Office of the Registrar General & Census
   Commissioner, India.

---

## Appendix A: Technology Stack

| Component         | Technology       | Version  | Purpose                        |
|-------------------|------------------|----------|--------------------------------|
| Frontend          | Streamlit        | 1.28+    | Interactive web dashboard      |
| ML Framework      | XGBoost          | 1.7+     | Gradient boosted regression    |
| Explainability    | SHAP             | Latest   | Feature attribution            |
| AI Insights       | Google Gemini    | Pro      | Natural language explanations  |
| Data Processing   | Pandas / NumPy   | Latest   | Data manipulation              |
| Visualization     | Plotly           | Latest   | Interactive charts             |
| Model Persistence | Joblib           | Latest   | Model serialization            |
| Language          | Python           | 3.8+     | Core implementation            |

---

## Appendix B: System Requirements

| Resource          | Minimum   | Recommended |
|-------------------|-----------|-------------|
| RAM               | 8 GB      | 16 GB       |
| Disk Space        | 5 GB      | 10 GB       |
| CPU Cores         | 4         | 8+          |
| Training Time     | 15–30 min | —           |
| Model Size        | 50–100 MB | —           |
| Prediction Time   | < 5 sec   | —           |

---

*Document Version: 1.0 | Last Updated: March 2026*
