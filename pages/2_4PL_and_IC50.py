# pages/2_4PL_and_IC50.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.four_param_fit import fit_4pl_and_ic50

st.header("📊 4PL Curve Fitting & IC50 Analysis")

if "display_df" not in st.session_state:
    st.warning("Please upload and preprocess data on Page 1 first.")
    st.stop()

df = st.session_state["display_df"]
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")
x_col = st.selectbox("Select concentration (X-axis) column", df.columns)
y_col = st.selectbox("Select signal (Y-axis) column", df.columns)
group_col = st.selectbox("Optional grouping column", [None] + list(df.columns))

st.markdown("### Combined Plot by Group")
results, plots, overlay_fig, ic50_dict = fit_4pl_and_ic50(df, x_col, y_col, group_col, return_plotly=True)

st.plotly_chart(overlay_fig, use_container_width=True)

st.markdown("### Individual Fit Results")

for key in results:
    st.markdown(f"### 4PL Fit for Group: {key}")
    r2 = results[key].get("R2")
    if r2 is not None:
        st.markdown(f"**R² = {r2:.3f}**")
    st.write(results[key])
    st.plotly_chart(plots[key], use_container_width=True)


# Save IC50s in session state for downstream reuse
if group_col:
    st.session_state["ic50_by_group"] = ic50_dict