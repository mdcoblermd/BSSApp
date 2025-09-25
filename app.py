# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:58:03 2025
BSS

@author: mdcob
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re

# ---------- Page setup ----------
st.set_page_config(page_title="BSS", layout="centered")
st.markdown("""
<style>
.block-container { max-width: 1100px; padding: 2rem 4rem; }
html, body, [class*="css"] { font-size: 1.10rem; }
input, select, textarea { font-size: 1.0rem !important; }
label { margin-bottom: 0.2rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_artifacts():
    with open("calibrated_model_burn.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_burn.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------- Title & intro ----------
st.title("BSS Score")
st.markdown("""
<h4 style='margin-top:-8px;color:gray;'>A mortality prediction tool for burns patients with >20% TBSA</h4>
<p style='font-size:16px;color:#555;'>
Developed by <b>MD Cobler-Lichter MD MSDS</b>, JM Delamater MD MPH, AM Reyes MD MPH, TR Arcieri MD,
JI Kaufman MD, SS Satahoo MD, LR Pizano MD MBA, N Namias MD MBA, KG Proctor PhD, CI Schulman MD PhD MSPH
</p>
<p style='font-size:14px;color:#555;'>
Models are calibrated so the output reflects the predicted probability of in-hospital mortality based on the training data.
</p>
""", unsafe_allow_html=True)


# ---------- Helpers (live inputs; namespaced keys; no reset on rerun) ----------
def int_input_live(label, key, min_val=None, max_val=None, placeholder=""):
    raw_key = f"numraw_{key}"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = ""
    raw = st.text_input(label, key=raw_key, placeholder=placeholder).strip()
    if raw == "":
        return np.nan
    if re.fullmatch(r"\d+", raw):
        v = int(raw)
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            return np.nan
        return v
    return np.nan

def float_input_live(label, key, min_val=None, max_val=None, placeholder=""):
    raw_key = f"numraw_{key}"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = ""
    raw = st.text_input(label, key=raw_key, placeholder=placeholder).strip()
    if raw == "":
        return np.nan
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        v = float(raw)
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            return np.nan
        return v
    return np.nan

def bin_tbsa(x):
    """Map user-entered TBSA (20–100) to model TBSA buckets."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if 20 <= x <= 29:
        return 25
    elif 30 <= x <= 39:
        return 35
    elif 40 <= x <= 89:
        return 65
    elif 90 <= x <= 100:
        return 90
    # Shouldn't happen because we constrain input, but keep safe fallback:
    return np.nan

def yes_no_radio(label, key):
    return st.radio(label, ['No', 'Yes'], index=0, horizontal=True, key=key)


# ---------- Labels / bounds ----------
label_map = {
    'TBSAforBaux':      "TBSA",
    'AGEYEARS':         "Age",
    'TOTALGCS':         "Arrival GCS",
    'SBP':              "Arrival SBP",
    'PULSERATE':        "Arrival Heart Rate",
    'InhalationInjury': "Inhalation Injury?"
}

bounds = {
    'TBSAforBaux': (20, 100),
    'TOTALGCS': (3, 15),
    'SBP': (0, 360),
    'PULSERATE': (0, 320),
    'AGEYEARS': (0, 150),
}


# ---------- Patient Info & Vitals + Predict BUTTON (INSIDE a form) ----------
with st.form("bss_form", clear_on_submit=False):
    user_inputs = {}
    sbp_val = np.nan
    pulse_val = np.nan

    # Just one column now (no st.columns)
    st.subheader("Patient Info & Vitals")

    tbsa_raw = None
    for var in ['AGEYEARS','TOTALGCS','SBP', 'PULSERATE', 'TBSAforBaux']:
        lo, hi = bounds[var]
        if var == 'TBSAforBaux':
            val = (int_input_live(label_map[var], var, min_val=lo, max_val=hi))
            val = bin_tbsa(val)
        else:
            val = int_input_live(label_map[var], var, min_val=lo, max_val=hi)
        user_inputs[var] = val
        if var == 'SBP': sbp_val = val
        if var == 'PULSERATE': pulse_val = val

    inhalation = st.radio(label_map['InhalationInjury'], ['Yes', 'No'],
                          index=0, horizontal=True, key='InhalationInjury')
    user_inputs['InhalationInjury'] = 1 if inhalation == 'Yes' else 0

    # 🔔 Warning if TBSA < 20
    if isnan(tbsa_raw):
        st.warning("⚠️ This model is designed for patients with ≥20% TBSA burns. Predictions not be valid for TBSA <20%.")
    
    # ShockIndex
    if (isinstance(sbp_val, (int, float)) and isinstance(pulse_val, (int, float))
        and not np.isnan(sbp_val) and not np.isnan(pulse_val) and sbp_val != 0):
        user_inputs['ShockIndex'] = pulse_val / sbp_val
    elif sbp_val == 0 or pulse_val == 0:
        user_inputs['ShockIndex'] = 2.0
    else:
        user_inputs['ShockIndex'] = np.nan

    # Build X in model's expected order (NaNs allowed)
    X = None
    try:
        X = pd.DataFrame([user_inputs], columns=scaler.feature_names_in_)
    except Exception as e:
        st.error(f"Column alignment error: {e}")

    submitted = st.form_submit_button("Predict Mortality")


# ---------- Output (persist last prediction) ----------
st.markdown("### BSS Score (Predicted Mortality):")
mortality_output = st.empty()

if 'last_pred' not in st.session_state:
    st.session_state['last_pred'] = None

if submitted and X is not None:
    try:
        X_scaled = scaler.transform(X)
        pred = float(model.predict_proba(X_scaled)[:, 1][0])
        st.session_state['last_pred'] = pred
    except Exception as e:
        st.session_state['last_pred'] = None
        mortality_output.error(f"Error during prediction: {e}")

if st.session_state['last_pred'] is not None:
    mortality_output.markdown(
        f"<p style='font-size:36px;font-weight:bold;color:#d62728;'>{st.session_state['last_pred']:.1%}</p>",
        unsafe_allow_html=True
    )












