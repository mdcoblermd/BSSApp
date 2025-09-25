# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:58:03 2025
MLISS

@author: mdcob
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(layout="centered")

st.markdown("""
    <style>
    /* Widen the page a bit more */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    /* Make fonts more legible */
    html, body, [class*="css"]  {
        font-size: 1.25rem;
    }

    /* Optional: make input boxes bigger */
    input, select, textarea {
        font-size: 1.1rem !important;
    }

    /* Reduce vertical whitespace between form elements */
    label {
        margin-bottom: 0.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)




# === Load Model and Scaler ===
with open('calibrated_model_burn.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler_burn.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Burn Severity Score (BSS)")


st.markdown(
    """
    <h4 style='margin-top: -10px; color: gray;'>
        A mortality prediction tool for burns patients with >20% TBSA
    </h4>
    <p style='font-size:22px; color: #555; font-style: bold;'>
        Developed by MD Cobler-Lichter MD MSDS, JM Delamater MD MPH, AM Reyes MD MPH, TR Arcieri MD,
        JI Kaufman MD, SS Satahoo MD, LR Pizano MD MBA, N Namias MD MBA, KG Proctor PhD, CI Schulman MD PhD MSPH
    </p>
    <p style='font-size:18x; color: #555; font-style: italic;'>
        Our models are calibrated such that the outputted scores reflects an accurate prediction
        of the true probability of in-hospital mortality based on our training data
    </p>,
    """,
    unsafe_allow_html=True
)


# --- Helper functions ---
def int_input(label, key):
    raw = st.text_input(label, value="", key=key, help="Enter a whole number or leave blank")
    try:
        return int(raw)
    except ValueError:
        return np.nan

def yes_no_radio(label, key):
    return st.radio(label, ['No', 'Yes'], index=0, horizontal=True, key=key)

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

# --- Label map (model_var -> user-facing label) ---
label_map = {
    'TBSAforBaux':      "TBSA",
    'AGEYEARS':         "Age",
    'TOTALGCS':         "Arrival GCS",
    'SBP':              "Arrival SBP",
    'PULSERATE':        "Arrival Heart Rate",
    'InhalationInjury': "Inhalation Injury?"
}

user_inputs = {}
sbp_val = None
pulse_val = None

st.subheader("Patient Info & Vitals")

for model_var, ui_label in label_map.items():
    if model_var == 'InhalationInjury':
        yn = yes_no_radio(ui_label, key=model_var)
        user_inputs[model_var] = 1 if yn == 'Yes' else 0

    elif model_var == 'TBSAforBaux':
        # Constrain input to 20–100, then bin to model TBSA
        tbsa_raw = st.number_input(ui_label, min_value=20, max_value=100, step=1, key=model_var)
        user_inputs[model_var] = bin_tbsa(tbsa_raw)
        # If you want to show what the model will use, uncomment:
        # st.caption(f"Model TBSA used: {int(user_inputs[model_var])}%")

    else:
        val = int_input(ui_label, model_var)
        user_inputs[model_var] = val
        if model_var == 'SBP':
            sbp_val = val
        elif model_var == 'PULSERATE':
            pulse_val = val

# --- Backend-only ShockIndex (not shown in UI) ---
if sbp_val == 0:
    user_inputs['ShockIndex'] = 2.0
elif (sbp_val is not None and pulse_val is not None
      and not (np.isnan(sbp_val) or np.isnan(pulse_val))):
    user_inputs['ShockIndex'] = pulse_val / sbp_val
else:
    user_inputs['ShockIndex'] = np.nan

# Build model input frame (columns are model-side names)
input_df = pd.DataFrame([user_inputs])


# Build model input frame (columns are model variable names)
input_df = pd.DataFrame([user_inputs])

# # Missing input warning
# if any(pd.isna(v) for v in user_inputs.values()):
#     st.markdown(
#         "<span style='color:red; font-weight:bold;'>"
#         "⚠️ One or more of the input variables are missing. "
#         "A score will still be calculated but it may be inaccurate."
#         "</span>",
#         unsafe_allow_html=True
#     )

st.markdown("### BSS Score (Predicted Mortality in severe burns):")
mortality_output = st.empty()

# Predict (ensure your scaler/model expect these exact feature names, including ShockIndex)
if st.button("Predict Mortality"):
    try:
        # Align to training feature order if needed
        input_df = input_df[scaler.feature_names_in_]
        X_scaled = scaler.transform(input_df)
        proba = model.predict_proba(X_scaled)[:, 1][0]

        mortality_output.markdown(
            f"<p style='font-size:36px; font-weight:bold; color:#d62728;'>{proba:.1%}</p>",
            unsafe_allow_html=True
        )
    except Exception as e:
        mortality_output.error(f"Error: {e}")

# # Reset
# if st.button("Reset Form"):
#     for k in list(st.session_state.keys()):
#         del st.session_state[k]
#     st.rerun()
