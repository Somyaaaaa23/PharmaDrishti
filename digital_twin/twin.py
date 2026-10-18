"""
PharmaDrishti GlucoTwin - Type 2 Diabetes Digital Twin (proof of concept).

Fuses two synthetic data streams per virtual patient:
  1. Static EHR: demographics, diagnoses, labs (HbA1c, fasting glucose, lipids),
     genetic risk marker (TCF7L2), medication (metformin / insulin).
  2. Dynamic wearable/IoT: 5-min CGM glucose, heart rate, HRV, steps, sleep stage,
     plus logged meals (carbs).

Predicts: hyperglycaemic spike (CGM > 180 mg/dL) within the next 2 hours.

Usage:
    python digital_twin/twin.py            # generate data, train, evaluate, save model
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"

STEP_MIN = 5                      # CGM sampling interval (minutes)
STEPS_PER_DAY = 24 * 60 // STEP_MIN
HORIZON = 120 // STEP_MIN         # 2-hour prediction horizon
SPIKE_MGDL = 180

INDIAN_STATES = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Uttar Pradesh",
                 "West Bengal", "Gujarat", "Kerala", "Telangana", "Punjab"]


# ---------------------------------------------------------------- EHR stream
def generate_ehr(n_patients: int = 200, seed: int = 7) -> pd.DataFrame:
    """Synthetic Synthea-style EHR records for Indian adults with / at risk of T2D."""
    rng = np.random.default_rng(seed)
    age = rng.integers(28, 76, n_patients)
    sex = rng.choice(["M", "F"], n_patients)
    bmi = np.clip(rng.normal(27, 4.5, n_patients), 18, 42)
    tcf7l2 = rng.choice([0, 1, 2], n_patients, p=[0.55, 0.37, 0.08])     # risk allele count
    family_hx = rng.random(n_patients) < 0.45
    hba1c = np.clip(5.4 + 0.04 * (bmi - 23) + 0.35 * tcf7l2 + 0.6 * family_hx
                    + 0.015 * (age - 40) + rng.normal(0, 0.9, n_patients), 5.0, 12.5)
    diabetic = hba1c >= 6.5
    on_metformin = diabetic & (rng.random(n_patients) < 0.8)
    on_insulin = (hba1c >= 8.5) & (rng.random(n_patients) < 0.5)
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n_patients)],
        "age": age,
        "sex": sex,
        "state": rng.choice(INDIAN_STATES, n_patients),
        "bmi": bmi.round(1),
        "hba1c": hba1c.round(1),
        "fasting_glucose": (28.7 * hba1c - 46.7 + rng.normal(0, 8, n_patients)).round(0),
        "ldl": np.clip(rng.normal(120, 30, n_patients), 60, 220).round(0),
        "systolic_bp": np.clip(rng.normal(128 + 0.4 * (age - 40), 14), 100, 190).round(0),
        "tcf7l2_risk_alleles": tcf7l2,
        "family_history_t2d": family_hx.astype(int),
        "dx_t2d": diabetic.astype(int),
        "dx_hypertension": (rng.random(n_patients) < 0.3 + 0.004 * (age - 40)).astype(int),
        "on_metformin": on_metformin.astype(int),
        "on_insulin": on_insulin.astype(int),
    })
    return df


# ----------------------------------------------------------- wearable stream
def simulate_wearables(patient: pd.Series, days: int = 7, seed: int = 0) -> pd.DataFrame:
    """Physiology-inspired simulation of CGM + smartwatch data for one patient."""
    rng = np.random.default_rng(seed)
    n = days * STEPS_PER_DAY
    t = pd.date_range("2026-09-01", periods=n, freq=f"{STEP_MIN}min")
    minute = t.hour * 60 + t.minute

    insulin_resistance = 0.6 + 0.12 * (patient.hba1c - 5.5) + 0.02 * (patient.bmi - 23)
    insulin_resistance *= 0.8 if patient.on_metformin else 1.0
    insulin_resistance *= 0.7 if patient.on_insulin else 1.0
    basal = patient.fasting_glucose

    # Meals: breakfast / lunch / dinner (+ random snack), Indian carb-heavy diet
    carbs = np.zeros(n)
    for d in range(days):
        for centre, mu in [(8.5, 55), (13.5, 75), (20.5, 80), (17, 25)]:
            if centre == 17 and rng.random() > 0.5:
                continue
            idx = d * STEPS_PER_DAY + int((centre + rng.normal(0, 0.6)) * 60 / STEP_MIN)
            if 0 <= idx < n:
                carbs[idx] += max(0, rng.normal(mu, mu * 0.35))

    # Activity: steps per 5 min, with a walk on some days
    awake = (minute >= 6.5 * 60) & (minute <= 23 * 60)
    steps = np.where(awake, rng.gamma(2, 18, n), 0)
    for d in range(days):
        if rng.random() < 0.55:
            s = d * STEPS_PER_DAY + int(rng.uniform(6.5, 19) * 60 / STEP_MIN)
            steps[s:s + 6] += rng.uniform(400, 650)

    # Sleep stage: 0 awake, 1 light, 2 deep, 3 REM
    sleep = np.where(awake, 0, rng.choice([1, 2, 3], n, p=[0.55, 0.2, 0.25]))
    poor_sleep_night = np.repeat(rng.random(days) < 0.3, STEPS_PER_DAY)

    # Glucose dynamics: meal absorption kernel convolved with carbs, damped by activity
    k = np.arange(0, 36)                                   # 3 h absorption
    kernel = (k / 9.0) * np.exp(1 - k / 9.0)
    meal_effect = np.convolve(carbs, kernel, mode="full")[:n] * 0.9 * insulin_resistance
    activity = pd.Series(steps).rolling(12, min_periods=1).mean().to_numpy()
    dawn = 12 * np.exp(-((minute - 6 * 60) / 60.0) ** 2)  # dawn phenomenon
    stress = np.where(poor_sleep_night, 10, 0)
    noise = np.cumsum(rng.normal(0, 1.2, n)) * 0.15
    glucose = basal + meal_effect - 0.06 * activity * insulin_resistance + dawn + stress + noise
    glucose = np.clip(glucose, 55, 400)

    hr = 62 + 0.04 * steps + 3 * poor_sleep_night + rng.normal(0, 3, n) - 6 * (sleep > 0)
    hrv = np.clip(55 - 0.35 * (patient.age - 30) - 6 * poor_sleep_night
                  - 0.05 * (glucose - 120) + rng.normal(0, 5, n), 8, 120)

    return pd.DataFrame({
        "timestamp": t, "patient_id": patient.patient_id,
        "glucose": glucose.round(1), "carbs_g": carbs.round(0),
        "steps": steps.round(0), "heart_rate": hr.round(0), "hrv_rmssd": hrv.round(1),
        "sleep_stage": sleep,
    })


# ------------------------------------------------------------------- fusion
EHR_FEATURES = ["age", "bmi", "hba1c", "fasting_glucose", "ldl", "systolic_bp",
                "tcf7l2_risk_alleles", "family_history_t2d", "dx_t2d", "dx_hypertension",
                "on_metformin", "on_insulin"]


def build_features(ehr: pd.DataFrame, ts: pd.DataFrame, drop_spiking: bool = True) -> pd.DataFrame:
    """Join static EHR with rolling wearable features; label = spike in next 2 h."""
    out = []
    for pid, g in ts.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        f = pd.DataFrame({"patient_id": pid, "timestamp": g.timestamp})
        gl = g.glucose
        f["glucose_now"] = gl
        f["glucose_delta_30m"] = gl - gl.shift(6)
        f["glucose_delta_60m"] = gl - gl.shift(12)
        f["glucose_mean_3h"] = gl.rolling(36).mean()
        f["glucose_std_3h"] = gl.rolling(36).std()
        f["glucose_max_24h"] = gl.rolling(STEPS_PER_DAY, min_periods=36).max()
        f["time_in_range_24h"] = gl.between(70, 180).rolling(STEPS_PER_DAY, min_periods=36).mean()
        f["carbs_last_1h"] = g.carbs_g.rolling(12, min_periods=1).sum()
        f["carbs_last_3h"] = g.carbs_g.rolling(36, min_periods=1).sum()
        f["steps_last_1h"] = g.steps.rolling(12, min_periods=1).sum()
        f["hr_mean_1h"] = g.heart_rate.rolling(12, min_periods=1).mean()
        f["hrv_mean_6h"] = g.hrv_rmssd.rolling(72, min_periods=1).mean()
        f["deep_sleep_last_night"] = (g.sleep_stage == 2).rolling(STEPS_PER_DAY, min_periods=1).sum()
        f["hour"] = g.timestamp.dt.hour
        future_max = gl[::-1].rolling(HORIZON, min_periods=1).max()[::-1].shift(-1)
        f["label_spike_2h"] = ((future_max > SPIKE_MGDL) & (gl <= SPIKE_MGDL)).astype(int)
        f["valid"] = gl <= SPIKE_MGDL                     # predict only when not already spiking
        out.append(f.iloc[36:-HORIZON])
    feats = pd.concat(out, ignore_index=True)
    if drop_spiking:
        feats = feats[feats.valid]
    feats = feats.drop(columns="valid")
    return feats.merge(ehr[["patient_id"] + EHR_FEATURES], on="patient_id")


FEATURES = None  # filled after build


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("patient_id", "timestamp", "label_spike_2h")]


# ----------------------------------------------------------------- pipeline
def run(n_patients: int = 200, days: int = 7) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    ehr = generate_ehr(n_patients)
    ts = pd.concat([simulate_wearables(p, days, seed=i) for i, p in ehr.iterrows()], ignore_index=True)
    ehr.to_csv(DATA_DIR / "synthetic_ehr.csv", index=False)
    ts.to_csv(DATA_DIR / "synthetic_wearables.csv.gz", index=False, compression="gzip")

    feats = build_features(ehr, ts)
    cols = feature_columns(feats)
    # patient-level split: twins of unseen patients
    pids = ehr.patient_id.sample(frac=1, random_state=0).tolist()
    test_ids = set(pids[: int(0.2 * len(pids))])
    tr, te = feats[~feats.patient_id.isin(test_ids)], feats[feats.patient_id.isin(test_ids)]

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=0)
    model.fit(tr[cols], tr.label_spike_2h)
    p = model.predict_proba(te[cols])[:, 1]
    yhat = (p >= 0.5).astype(int)

    # ablation: wearable-only and EHR-only
    def auc_with(subset):
        m = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)
        m.fit(tr[subset], tr.label_spike_2h)
        return roc_auc_score(te.label_spike_2h, m.predict_proba(te[subset])[:, 1])

    wear_cols = [c for c in cols if c not in EHR_FEATURES]
    metrics = {
        "patients": n_patients, "days_per_patient": days,
        "train_samples": len(tr), "test_samples": len(te),
        "spike_rate": round(float(feats.label_spike_2h.mean()), 3),
        "roc_auc_fused": round(roc_auc_score(te.label_spike_2h, p), 3),
        "pr_auc_fused": round(average_precision_score(te.label_spike_2h, p), 3),
        "precision": round(precision_score(te.label_spike_2h, yhat), 3),
        "recall": round(recall_score(te.label_spike_2h, yhat), 3),
        "f1": round(f1_score(te.label_spike_2h, yhat), 3),
        "roc_auc_wearable_only": round(auc_with(wear_cols), 3),
        "roc_auc_ehr_only": round(auc_with(EHR_FEATURES + ["hour"]), 3),
    }
    joblib.dump({"model": model, "features": cols}, MODEL_DIR / "glucotwin.pkl")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
