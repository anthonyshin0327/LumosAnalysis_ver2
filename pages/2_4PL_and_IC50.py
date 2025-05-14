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

# Run 4PL fitting and compute summary data
results, plots, overlay_fig, ic50_dict = fit_4pl_and_ic50(df, x_col, y_col, group_col, return_plotly=True)



with st.expander("🔍 Individual Fit Results"):
    # Display the combined overlay plot at the top
    st.markdown("### Combined Overlay Plot by Group")
    st.plotly_chart(overlay_fig, use_container_width=True)
    cv_summary = {}
    ic50_summary = {}
    cv_raw = []

    for key in results:
        st.markdown(f"### 4PL Fit for Group: {key}")
        r2 = results[key].get("R2")
        mean_cv = results[key].get("Mean CV (%)")
        ic50_val = results[key].get("c (IC50)")
        if r2 is not None:
            st.markdown(f"**R² = {r2:.3f}**")
        if mean_cv is not None:
            st.markdown(f"**Mean CV% = {mean_cv:.2f}**")
            cv_summary[key] = mean_cv
        if ic50_val is not None:
            ic50_summary[key] = ic50_val

        st.write(results[key])
        st.plotly_chart(plots[key], use_container_width=True)

        for row in results[key].get("CV by Concentration (%)", []):
            cv_raw.append({"Group": key, "Concentration": row["Concentration"], "CV (%)": row["CV (%)"]})
        cv_raw.append({"Group": key, "Concentration": row["Concentration"], "CV (%)": row["CV (%)"]})

# Combined Summary Plots

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
if cv_summary:
    st.markdown("### Mean CV% Summary by Group")
    summary_fig = go.Figure()
    summary_fig.add_trace(go.Bar(x=list(cv_summary.keys()), y=list(cv_summary.values()), name="Mean CV%"))
    summary_fig.update_layout(xaxis_title="Group", yaxis_title="Mean CV%", title="Average CV% Across Progesterone Levels")
    st.plotly_chart(summary_fig, use_container_width=True)

# Summary plot: IC50 by group
if ic50_summary:
    st.markdown("### IC50 Summary by Group")

    # Prepare data sorted by IC50 ascending
    sorted_ic50 = sorted(ic50_summary.items(), key=lambda x: x[1])
    groups_sorted = [x[0] for x in sorted_ic50]
    ic50_values = [x[1] for x in sorted_ic50]

    # Determine best (lowest) IC50 group for highlighting and annotation
    best_group = groups_sorted[0]
    best_ic50 = ic50_values[0]

    # Compute analytical range: ~25% to 75% response from 4PL S-curve, approximate ±1 log from IC50
    from numpy import log10, power
    range_min = best_ic50 / 10
    range_max = best_ic50 * 10

    # Create horizontal barplot with best group in red
    ic50_fig = go.Figure()
    for idx, (g, v) in enumerate(sorted_ic50[::-1]):  # Reverse so smallest is at top
        color = "crimson" if g == best_group else None
        ic50_fig.add_trace(go.Bar(
            x=[v],
            y=[f"{idx+1}. {g}"],  # Label groups numerically
            orientation="h",
            name=g,
            marker_color=color,
            text=f"IC50 = {v:.2f}",
            textposition="auto"
        ))

    subtitle = f"The group {best_group} has the best sensitivity at IC50 = {best_ic50:.2f} with analytical range of {range_min:.2f}–{range_max:.2f}"
    ic50_fig.update_layout(
        title={"text": "IC50 by Group", "x": 0.5, "xanchor": "center"},
        xaxis_title="IC50",
        yaxis_title="Group (ascending IC50)",
        title_font_size=20,
        margin=dict(l=100, r=40, t=60, b=40),
        annotations=[
            dict(
                text=subtitle,
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(size=12)
            )
        ],
        showlegend=False,
        bargap=0.05  # Reduce gap between bars
    )
    st.plotly_chart(ic50_fig, use_container_width=True)

# Save IC50s in session state for downstream reuse
if group_col:
    st.session_state["ic50_by_group"] = ic50_dict
