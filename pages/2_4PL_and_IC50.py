import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.four_param_fit import fit_4pl_and_ic50

st.header("📊 4PL Curve Fitting & IC50 Analysis")

if "display_df" not in st.session_state:
    st.warning("Please upload and preprocess data on Page 1 first.")
    st.stop()

df = st.session_state["display_df"]

# Compute derived features if TLH and CLH exist
if all(col in df.columns for col in ["TLH", "CLH"]):
    df["TLH - CLH"] = df["TLH"] - df["CLH"]
    df["CLH / TLH"] = df["CLH"] / df["TLH"].replace(0, np.nan)
    df["TLH / CLH"] = df["TLH"] / df["CLH"].replace(0, np.nan)
    df["Normalized TLH - CLH"] = (df["TLH"] - df["CLH"]) / (df["TLH"] + df["CLH"]).replace(0, np.nan)

st.dataframe(df.head(), use_container_width=True)

with st.expander("ℹ️ Derived columns added"):
    st.write("The following additional columns were computed and added to your data:")
    st.markdown("- TLH - CLH\n- CLH / TLH\n- TLH / CLH\n- Normalized TLH - CLH")

st.markdown("---")
x_col = st.selectbox("Select concentration (X-axis) column", df.columns)
y_col = st.selectbox("Select signal (Y-axis) column", df.columns)
group_col = st.selectbox("Optional grouping column", [None] + list(df.columns))

st.markdown("### Combined Plot by Group")
results, plots, overlay_fig, ic50_dict = fit_4pl_and_ic50(df, x_col, y_col, group_col, return_plotly=True)

st.plotly_chart(overlay_fig, use_container_width=True)

st.markdown("### Individual Fit Results")
cv_summary = {}
cv_raw = []

for key in results:
    st.markdown(f"### 4PL Fit for Group: {key}")
    r2 = results[key].get("R2")
    mean_cv = results[key].get("Mean CV (%)")
    if r2 is not None:
        st.markdown(f"**R² = {r2:.3f}**")
    if mean_cv is not None:
        st.markdown(f"**Mean CV% = {mean_cv:.2f}**")
        cv_summary[key] = mean_cv

    st.write(results[key])
    st.plotly_chart(plots[key], use_container_width=True)

    # Collect per-concentration CVs for barplot
    for row in results[key].get("CV by Concentration (%)", []):
        cv_raw.append({"Group": key, "Concentration": row["Concentration"], "CV (%)": row["CV (%)"]})

# CV% per concentration bar chart (colored by concentration)
if cv_raw:
    st.markdown("### CV% by Group and Concentration")
    cv_df = pd.DataFrame(cv_raw)
    fig = go.Figure()
    for conc in sorted(cv_df["Concentration"].unique()):
        subset = cv_df[cv_df["Concentration"] == conc]
        fig.add_trace(go.Bar(x=subset["Group"], y=subset["CV (%)"], name=f"{conc}"))
    fig.update_layout(barmode="group", xaxis_title="Group", yaxis_title="CV (%)", title="CV% by Group and Progesterone Concentration")
    st.plotly_chart(fig, use_container_width=True)

# Summary plot: Mean CV% by group
if cv_summary:
    st.markdown("### Mean CV% Summary by Group")
    summary_fig = go.Figure()
    summary_fig.add_trace(go.Bar(x=list(cv_summary.keys()), y=list(cv_summary.values()), name="Mean CV%"))
    summary_fig.update_layout(xaxis_title="Group", yaxis_title="Mean CV%", title="Average CV% Across Progesterone Levels")
    st.plotly_chart(summary_fig, use_container_width=True)

# Save IC50s in session state for downstream reuse
if group_col:
    st.session_state["ic50_by_group"] = ic50_dict