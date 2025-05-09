# pages/4_Dashboard_Summary.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="📊 Dashboard Summary", layout="centered")
st.title("📊 LFA Experiment Summary")

# --- Debug Preview (optional dev mode check) ---
try:
    ic50_by_group = st.session_state.get("ic50_by_group")
    model_metrics = st.session_state.get("model_metrics")
    optimized_row = st.session_state.get("optimized_row")
    input_df = st.session_state.get("rsm_input_df")
except Exception as e:
    st.error(f"Error loading session state: {e}")
    ic50_by_group, model_metrics, optimized_row, input_df = None, None, None, None

st.markdown("---")

# --- Summary Narrative ---
st.subheader("🧪 Experimental Summary")

if ic50_by_group and model_metrics and optimized_row is not None and input_df is not None:
    ic50_series = pd.Series(ic50_by_group)
    best_group = ic50_series.idxmin()
    best_ic50 = ic50_series.min()
    n_groups = len(ic50_series)

    eq_parts = [f"{v:.2f}×{k}" for k, v in model_metrics["coefficients"].items()]
    intercept = model_metrics["intercept"]
    equation = f"Y = {intercept:.2f} + " + " + ".join(eq_parts)

    st.markdown(f"""
    In this experiment, you compared **{n_groups} groups**. The group with the lowest IC50 was **{best_group}** with a value of **{best_ic50:.2f}**.

    A quadratic response surface model was fit to the design matrix. The resulting model equation was:

    $${equation}$$

    The model achieved an R² of **{model_metrics['R2']:.4f}**, indicating a strong fit to the observed data.
    """)

    st.markdown("---")

    st.subheader("🔍 Optimal Experimental Condition")
    st.dataframe(pd.DataFrame(optimized_row).T.style.format("{:.4f}"))

else:
    st.warning("Insufficient data to display summary. Please complete pages 2 and 3 first.")