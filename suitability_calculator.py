"""
Suitability Calculator Module

Calculates adoption probability for persona-medicine pairs using a weighted formula.
This is the core logic for generating training data for the ML model.
"""

import math
import pandas as pd
import numpy as np


TIER_1 = 'Tier 1'
TIER_2 = 'Tier 2'
TIER_3 = 'Tier 3'
TIER_ACCESSIBILITY = {
    TIER_1: 0.9,
    TIER_2: 0.7,
    TIER_3: 0.5,
}

GENDER_ENCODING = {
    'Female': 0,
    'Male': 1,
}

INCOME_LEVEL_ENCODING = {
    'Low': 1,
    'Lower-Middle': 2,
    'Middle': 3,
    'Upper-Middle': 4,
    'High': 5,
}

DIGITAL_ADOPTION_ENCODING = {
    'None': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
}


def _extract_primary_disease(disease_text):
    text = str(disease_text or "").strip()
    if not text:
        return "Unknown"
    return text.split(",")[0].strip()


def calculate_suitability(persona, medicine, market_data=None):
    """
    Calculate adoption probability for a persona-medicine pair.
    
    Uses a weighted formula considering:
    - Disease match (30%): Binary match between medicine disease and persona conditions
    - Affordability (25%): Price relative to persona income
    - Market pricing (20%): Price competitiveness vs market alternatives
    - Availability (15%): Medicine availability × persona location accessibility
    - Risk tolerance (10%): Side effect risk × persona risk aversion
    
    Parameters:
    -----------
    persona : dict or pd.Series
        Persona data containing:
        - chronic_diseases: str (comma-separated or "None")
        - annual_income_inr: float
        - city_tier: str ("Tier 1", "Tier 2", "Tier 3")
        - price_sensitivity_score: float (0-1)
    
    medicine : dict or pd.Series
        Medicine profile containing:
        - target_disease: str
        - batch_price_inr: float
        - manufacturing_score: float (0-1)
        - side_effect_risk: float (0-1)
    
    market_data : pd.DataFrame, optional
        Market pricing data for competitive analysis
    
    Returns:
    --------
    float
        Adoption probability between 0 and 1
    """
    # 1. Disease Match (30% weight) - Reduced from 35%
    persona_diseases = str(persona.get('chronic_diseases', '')).lower()
    target_disease = str(medicine.get('target_disease', '')).lower()
    
    disease_match = 1.0 if target_disease in persona_diseases else 0.0
    
    # 2. Affordability (25% weight)
    monthly_income = persona.get('annual_income_inr', 0) / 12
    batch_price = medicine.get('batch_price_inr', 0)
    
    if monthly_income > 0:
        affordability = max(0.0, 1.0 - (batch_price / (monthly_income * 100)))
        affordability = min(1.0, affordability)
    else:
        affordability = 0.0
    
    # 3. Market Pricing Competitiveness (20% weight) - NEW
    market_pricing_score = 0.5  # Default neutral score
    
    if market_data is not None and not market_data.empty:
        try:
            # Find similar medicines in market
            disease_keywords = target_disease.split()
            similar_medicines = market_data[
                market_data['medicine_name'].str.contains('|'.join(disease_keywords), case=False, na=False)
            ]
            
            if len(similar_medicines) > 0:
                market_prices = similar_medicines['price_numeric'].dropna()
                if len(market_prices) > 0:
                    avg_market_price = market_prices.mean()
                    
                    # Calculate competitiveness score
                    if avg_market_price > 0:
                        price_ratio = batch_price / avg_market_price
                        
                        # Score based on price position:
                        # - Much cheaper (< 0.7): 1.0 (very competitive)
                        # - Competitive (0.7-1.3): 0.8-0.6 (good)
                        # - Expensive (1.3-2.0): 0.6-0.2 (poor)
                        # - Very expensive (> 2.0): 0.1 (very poor)
                        
                        if price_ratio < 0.7:
                            market_pricing_score = 1.0
                        elif price_ratio < 1.3:
                            # Linear interpolation between 1.0 and 0.6
                            market_pricing_score = 1.0 - (price_ratio - 0.7) * (0.4 / 0.6)
                        elif price_ratio < 2.0:
                            # Linear interpolation between 0.6 and 0.2
                            market_pricing_score = 0.6 - (price_ratio - 1.3) * (0.4 / 0.7)
                        else:
                            market_pricing_score = 0.1
        except Exception:
            # If market analysis fails, use neutral score
            market_pricing_score = 0.5
    
    # 4. Manufacturing (15% weight)
    city_tier = persona.get('city_tier', TIER_3)
    location_factor = TIER_ACCESSIBILITY.get(city_tier, 0.5)
    manufacturing = medicine.get('manufacturing_score', 0.5) * location_factor
    
    # 5. Risk Tolerance (10% weight) - Reduced from 25%
    side_effect_risk = medicine.get('side_effect_risk', 0.5)
    price_sensitivity = persona.get('price_sensitivity_score', 0.5)
    risk_score = side_effect_risk * price_sensitivity
    
    # Calculate weighted sum with new market pricing component
    weighted_sum = (
        0.30 * disease_match +
        0.25 * affordability +
        0.20 * market_pricing_score +
        0.15 * manufacturing +
        0.10 * (1 - risk_score)
    )
    
    # Apply sigmoid transformation to get probability (0-1)
    scaled_sum = (weighted_sum - 0.5) * 12
    probability = 1 / (1 + math.exp(-scaled_sum))
    
    return probability



def create_interaction_dataset(personas_df, medicines_df):
    """
    Create interaction dataset with all persona-medicine pairs.
    
    For each persona-medicine combination:
    - Calculate suitability score using calculate_suitability()
    - Extract features for ML training
    - Store in interaction record
    
    Parameters:
    -----------
    personas_df : pd.DataFrame
        DataFrame containing persona data with columns:
        - name, age, annual_income_inr, city_tier, price_sensitivity_score,
          chronic_diseases, no_of_chronic_conditions, etc.
    
    medicines_df : pd.DataFrame
        DataFrame containing medicine profiles with columns:
        - medicine_id, target_disease, batch_price_inr,
          side_effect_risk, manufacturing_score, insurance_compatibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all interaction pairs containing:
        - persona_id: Persona identifier
        - medicine_id: Medicine identifier
        - suitability_score: Calculated adoption probability (0-1)
        - Feature columns for ML training (age, income, city_tier, price, etc.)
    
    Example:
    --------
    >>> personas = load_personas()
    >>> medicines = generate_medicines(5000)
    >>> interactions = create_interaction_dataset(personas, medicines)
    >>> print(f"Generated {len(interactions)} interaction pairs")
    >>> print(f"Expected: {len(personas)} × {len(medicines)} = {len(personas) * len(medicines)}")
    """
    # ── Encoding maps ────────────────────────────────────────────────────────
    tier_encoding = {TIER_1: 1, TIER_2: 2, TIER_3: 3}
    disease_values = sorted(
        {
            _extract_primary_disease(v)
            for v in personas_df.get('chronic_diseases', pd.Series(dtype=str)).dropna().tolist()
        }.union(set(medicines_df.get('target_disease', pd.Series(dtype=str)).dropna().tolist()))
    )
    disease_encoding = {name: idx + 1 for idx, name in enumerate(disease_values)}

    # ── Vectorised persona feature table ─────────────────────────────────────
    p = personas_df.copy().reset_index(drop=True)
    p['_persona_id'] = p.get('name', pd.Series(['P'] * len(p), dtype=str)).astype(str) + '_' + p.index.astype(str)
    p['_age']                    = pd.to_numeric(p.get('age', 0), errors='coerce').fillna(0)
    p['_gender_enc']             = p.get('gender', pd.Series(['Female']*len(p))).map(GENDER_ENCODING).fillna(0)
    p['_income']                 = pd.to_numeric(p.get('annual_income_inr', 0), errors='coerce').fillna(0)
    p['_income_level_enc']       = p.get('income_level', pd.Series(['Low']*len(p))).map(INCOME_LEVEL_ENCODING).fillna(1)
    p['_city_tier_enc']          = p.get('city_tier', pd.Series([TIER_3]*len(p))).map(tier_encoding).fillna(3)
    p['_location_factor']        = p.get('city_tier', pd.Series([TIER_3]*len(p))).map(TIER_ACCESSIBILITY).fillna(0.5)
    p['_chronic_count']          = pd.to_numeric(p.get('no_of_chronic_conditions', 0), errors='coerce').fillna(0)
    p['_price_sens']             = pd.to_numeric(p.get('price_sensitivity_score', 0.5), errors='coerce').fillna(0.5)
    p['_monthly_hc_spend']       = pd.to_numeric(p.get('monthly_healthcare_spend_inr', 0), errors='coerce').fillna(0)
    p['_insurance']              = pd.to_numeric(p.get('insurance_sum_assured_inr', 0), errors='coerce').fillna(0)
    p['_insured_flag']           = (p.get('insurance_status', pd.Series(['Uninsured']*len(p))) != 'Uninsured').astype(float)
    p['_last_visit']             = pd.to_numeric(p.get('last_doctor_visit_months_ago', 0), errors='coerce').fillna(0)
    p['_digital_enc']            = p.get('digital_health_adoption', pd.Series(['None']*len(p))).map(DIGITAL_ADOPTION_ENCODING).fillna(0)
    p['_primary_disease']        = p.get('chronic_diseases', pd.Series(['Unknown']*len(p))).apply(_extract_primary_disease)
    p['_persona_disease_enc']    = p['_primary_disease'].map(disease_encoding).fillna(0)
    p['_chronic_diseases_lower'] = p.get('chronic_diseases', pd.Series(['']*len(p))).astype(str).str.lower()
    p['_monthly_income']         = (p['_income'] / 12).clip(lower=1)
    p['_log_income']             = np.log1p(p['_income'])
    p['_age_cond']               = p['_age'] * p['_chronic_count']
    p['_price_sens_x_income']    = p['_price_sens'] * p['_log_income']
    p['_income_to_ins_ratio']    = p['_insurance'] / p['_income'].clip(lower=1)
    p['_key'] = 1

    # ── Vectorised medicine feature table ────────────────────────────────────
    m = medicines_df.copy().reset_index(drop=True)
    m['_price']               = pd.to_numeric(m.get('batch_price_inr', 0), errors='coerce').fillna(0)
    m['_side_effect_risk']    = pd.to_numeric(m.get('side_effect_risk', 0.5), errors='coerce').fillna(0.5)
    m['_manufacturing']       = pd.to_numeric(m.get('manufacturing_score', 0.5), errors='coerce').fillna(0.5)
    m['_ins_compat']          = pd.to_numeric(m.get('insurance_compatibility', 0.5), errors='coerce').fillna(0.5)
    m['_target_disease']      = m.get('target_disease', pd.Series(['Unknown']*len(m))).astype(str)
    m['_med_disease_enc']     = m['_target_disease'].map(disease_encoding).fillna(0)
    m['_log_price']           = np.log1p(m['_price'])
    m['_target_disease_lower'] = m['_target_disease'].str.lower()
    m['_key'] = 1

    # ── Cross join ────────────────────────────────────────────────────────────
    df = pd.merge(p, m, on='_key', how='inner', validate='many_to_many').drop(columns='_key')

    # ── Suitability formula components (vectorised) ───────────────────────────
    # disease match
    df['disease_match'] = df.apply(
        lambda r: 1.0 if r['_target_disease_lower'] in r['_chronic_diseases_lower'] else 0.0, axis=1
    )
    # affordability
    df['affordability'] = (1.0 - df['_price'] / (df['_monthly_income'] * 100)).clip(0.0, 1.0)
    # manufacturing match
    df['manufacturing_match'] = df['_manufacturing'] * df['_location_factor']
    # risk score
    df['risk_score'] = df['_side_effect_risk'] * df['_price_sens']

    # ── Suitability score (sigmoid of weighted sum) ───────────────────────────
    # market_pricing_score proxy: 0.5 (no market data available at this stage)
    weighted = (
        0.30 * df['disease_match'] +
        0.25 * df['affordability'] +
        0.20 * 0.5 +
        0.15 * df['manufacturing_match'] +
        0.10 * (1 - df['risk_score'])
    )
    df['suitability_score'] = 1 / (1 + np.exp(-((weighted - 0.5) * 12)))

    # ── Cross features ────────────────────────────────────────────────────────
    df['price_to_income_ratio']           = df['_price'] / df['_income'].clip(lower=1)
    df['price_to_monthly_income_ratio']   = df['_price'] / df['_monthly_income']
    df['price_to_healthcare_spend_ratio'] = df['_price'] / df['_monthly_hc_spend'].clip(lower=1)
    df['insurance_price_interaction']     = df['_ins_compat'] * df['_price']

    # ── Rename to final schema ────────────────────────────────────────────────
    result = pd.DataFrame({
        'persona_id':                    df['_persona_id'],
        'medicine_id':                   df.get('medicine_id', df.index.astype(str)),
        'suitability_score':             df['suitability_score'],
        'age':                           df['_age'],
        'gender_encoded':                df['_gender_enc'],
        'income':                        df['_income'],
        'income_level_encoded':          df['_income_level_enc'],
        'city_tier_encoded':             df['_city_tier_enc'],
        'chronic_condition_count':       df['_chronic_count'],
        'price_sensitivity':             df['_price_sens'],
        'monthly_healthcare_spend':      df['_monthly_hc_spend'],
        'persona_disease_encoded':       df['_persona_disease_enc'],
        'insurance_sum_assured':         df['_insurance'],
        'insured_flag':                  df['_insured_flag'],
        'last_doctor_visit_months':      df['_last_visit'],
        'digital_adoption_encoded':      df['_digital_enc'],
        'price':                         df['_price'],
        'side_effect_risk':              df['_side_effect_risk'],
        'manufacturing':                 df['_manufacturing'],
        'insurance_compatibility':       df['_ins_compat'],
        'medicine_disease_encoded':      df['_med_disease_enc'],
        'log_price':                     df['_log_price'],
        'log_income':                    df['_log_income'],
        'price_to_income_ratio':         df['price_to_income_ratio'],
        'price_to_monthly_income_ratio': df['price_to_monthly_income_ratio'],
        'price_to_healthcare_spend_ratio': df['price_to_healthcare_spend_ratio'],
        'income_to_insurance_ratio':     df['_income_to_ins_ratio'],
        'age_condition_interaction':     df['_age_cond'],
        'price_sensitivity_x_income':    df['_price_sens_x_income'],
        'insurance_price_interaction':   df['insurance_price_interaction'],
        'disease_match':                 df['disease_match'],
        'affordability':                 df['affordability'],
        'manufacturing_match':           df['manufacturing_match'],
        'risk_score':                    df['risk_score'],
    })

    return result
