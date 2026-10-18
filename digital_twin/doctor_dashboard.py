"""
GlucoTwin - doctor-facing dashboard for the virtual patient.

Run:  streamlit run digital_twin/doctor_dashboard.py
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twin  # noqa: E402

st.set_page_config(page_title="GlucoTwin - Virtual Patient", layout="wide")


@st.cache_resource
def load():
    if not (twin.MODEL_DIR / "glucotwin.pkl").exists():
        twin.run()
    bundle = joblib.load(twin.MODEL_DIR / "glucotwin.pkl")
    ehr = pd.read_csv(twin.DATA_DIR / "synthetic_ehr.csv")
    ts = pd.read_csv(twin.DATA_DIR / "synthetic_wearables.csv.gz", parse_dates=["timestamp"])
    metrics = json.loads((twin.MODEL_DIR / "metrics.json").read_text())
    return bundle, ehr, ts, metrics


bundle, ehr, ts, metrics = load()
model, cols = bundle["model"], bundle["features"]

st.title("GlucoTwin — Type 2 Diabetes Digital Twin")
st.caption("Synthetic EHR + simulated CGM / smartwatch data → 2-hour hyperglycaemia (>180 mg/dL) early warning")

# ------------------------------------------------------------ patient picker
with st.sidebar:
    st.header("Patient")
    pid = st.selectbox("Virtual patient", ehr.patient_id, index=int((ehr.hba1c - 7.8).abs().idxmin()))
    p = ehr.set_index("patient_id").loc[pid]
    st.subheader("EHR summary")
    st.write(f"**{int(p.age)} y, {p.sex}, {p.state}**")
    st.metric("HbA1c", f"{p.hba1c}%")
    st.metric("Fasting glucose", f"{int(p.fasting_glucose)} mg/dL")
    st.write(f"BMI {p.bmi} · BP {int(p.systolic_bp)} · LDL {int(p.ldl)}")
    st.write(f"TCF7L2 risk alleles: {int(p.tcf7l2_risk_alleles)} · Family hx: {'Yes' if p.family_history_t2d else 'No'}")
    st.write("Dx: " + ", ".join([d for d, f in [("T2D", p.dx_t2d), ("Hypertension", p.dx_hypertension)] if f] or ["None"]))
    st.write("Rx: " + ", ".join([d for d, f in [("Metformin", p.on_metformin), ("Insulin", p.on_insulin)] if f] or ["None"]))
    st.divider()
    st.caption(f"Model ROC-AUC {metrics['roc_auc_fused']} · Recall {metrics['recall']} (held-out patients)")

g = ts[ts.patient_id == pid].reset_index(drop=True)
feats = twin.build_features(ehr[ehr.patient_id == pid], g, drop_spiking=False)

# ------------------------------------------------------------ time slider
st.subheader("Replay the last 7 days")
idx = st.slider("Current time", 0, len(feats) - 1, len(feats) - 1, format="%d")
row = feats.iloc[[idx]]
now = row.timestamp.iloc[0]
risk = float(model.predict_proba(row[cols])[:, 1][0])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Now", now.strftime("%a %d %b %H:%M"))
c2.metric("Current glucose", f"{row.glucose_now.iloc[0]:.0f} mg/dL", f"{row.glucose_delta_30m.iloc[0]:+.0f} in 30 min")
c3.metric("Spike risk (next 2 h)", f"{risk:.0%}")
level = "HIGH" if risk >= 0.7 else "MODERATE" if risk >= 0.4 else "LOW"
c4.metric("Alert level", level)
if level == "HIGH":
    st.error("⚠️ High probability of glucose > 180 mg/dL within 2 hours. Consider a 15-min walk, "
             "deferring carbs, or reviewing the pre-meal dose.")
elif level == "MODERATE":
    st.warning("Rising risk — monitor and nudge patient towards light activity.")
else:
    st.success("Glucose expected to stay in range.")

# ------------------------------------------------------------ charts
window = g[(g.timestamp > now - pd.Timedelta(hours=24)) & (g.timestamp <= now + pd.Timedelta(hours=2))]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04,
                    subplot_titles=("CGM glucose (mg/dL) & meals", "Heart rate / HRV", "Steps"))
past = window[window.timestamp <= now]
future = window[window.timestamp > now]
fig.add_trace(go.Scatter(x=past.timestamp, y=past.glucose, name="Glucose", line=dict(color="#004aad")), 1, 1)
fig.add_trace(go.Scatter(x=future.timestamp, y=future.glucose, name="Actual (future, for validation)",
                         line=dict(color="#9fb3c8", dash="dot")), 1, 1)
meals = past[past.carbs_g > 0]
fig.add_trace(go.Bar(x=meals.timestamp, y=meals.carbs_g, name="Carbs (g)", marker_color="#f0b429", width=600000), 1, 1)
fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.07, line_width=0, row=1, col=1)
fig.add_vline(x=now, line_dash="dash", line_color="red")
fig.add_trace(go.Scatter(x=past.timestamp, y=past.heart_rate, name="HR", line=dict(color="#d64545")), 2, 1)
fig.add_trace(go.Scatter(x=past.timestamp, y=past.hrv_rmssd, name="HRV", line=dict(color="#8719e0")), 2, 1)
fig.add_trace(go.Bar(x=past.timestamp, y=past.steps, name="Steps", marker_color="#3ebd93"), 3, 1)
fig.update_layout(height=650, margin=dict(t=40, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ what-if
st.subheader("What-if: simulate an intervention on the twin")
w1, w2 = st.columns(2)
extra_steps = w1.slider("Walk in the next hour (steps)", 0, 3000, 0, 250)
carb_cut = w2.slider("Reduce upcoming / recent carbs by (%)", 0, 100, 0, 10)
sim = row.copy()
sim["steps_last_1h"] += extra_steps
sim["carbs_last_1h"] *= 1 - carb_cut / 100
sim["carbs_last_3h"] *= 1 - carb_cut / 100
sim_risk = float(model.predict_proba(sim[cols])[:, 1][0])
st.metric("Simulated spike risk", f"{sim_risk:.0%}", f"{(sim_risk - risk) * 100:+.0f} pts", delta_color="inverse")

# ------------------------------------------------------------ weekly summary
st.subheader("7-day summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Time in range (70-180)", f"{g.glucose.between(70, 180).mean():.0%}")
s2.metric("Time above 180", f"{(g.glucose > 180).mean():.0%}")
s3.metric("Mean glucose", f"{g.glucose.mean():.0f} mg/dL")
s4.metric("Avg daily steps", f"{g.steps.sum() / 7:,.0f}")
