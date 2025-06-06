import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# Assuming the backend code is saved in core/five_param_fit.py
from core.five_param_fit import fit_5pl_and_ic50 

st.set_page_config(layout="wide") 

st.header("📊 5PL Curve Fitting & IC50 Analysis") 

# Check if data is loaded from Page 1, using "processed_df" as the key
if "processed_df" not in st.session_state or st.session_state["processed_df"] is None:
    st.warning("Please complete data processing on Page 1 (Upload and Explore) first, or upload a CSV/Excel file below.")
    
    uploaded_file_pg2 = st.file_uploader("Upload your data (CSV or Excel) for 5PL Analysis", type=["csv", "xlsx"], key="page2_direct_uploader")
    if uploaded_file_pg2:
        try:
            if uploaded_file_pg2.name.endswith('.csv'):
                df_temp = pd.read_csv(uploaded_file_pg2)
            else:
                df_temp = pd.read_excel(uploaded_file_pg2)
            st.session_state["processed_df"] = df_temp # Save to "processed_df"
            st.success("File uploaded successfully on this page!")
            # Ensure df is loaded for the current run
            df = st.session_state["processed_df"]
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.stop() 
else:
    # Load the full processed DataFrame from Page 1
    df = st.session_state["processed_df"]

# --- Data Preprocessing & Derived Columns (operates on the df loaded above) ---
st.markdown("### 1. Data Preview & Preparation")
with st.expander("Show/Hide Full Data and Derived Columns", expanded=False):
    st.write("Displaying data loaded for analysis (from Page 1 or direct upload):")
    st.dataframe(df.head(), use_container_width=True)

    # Make a copy for modifications on this page to avoid altering session state df directly here
    df_page2_local = df.copy()

    # Compute derived features if TLH and CLH exist
    if "TLH" in df_page2_local.columns and "CLH" in df_page2_local.columns:
        try:
            # Ensure numeric conversion for calculation robustness
            df_page2_local["TLH"] = pd.to_numeric(df_page2_local["TLH"], errors='coerce')
            df_page2_local["CLH"] = pd.to_numeric(df_page2_local["CLH"], errors='coerce')

            # Drop rows where TLH or CLH became NaN after conversion, if they are essential for these derived cols
            df_page2_local.dropna(subset=["TLH", "CLH"], inplace=True) # Modifies df_page2_local

            if not df_page2_local.empty:
                df_page2_local["TLH - CLH"] = df_page2_local["TLH"] - df_page2_local["CLH"]
                df_page2_local["CLH / TLH"] = (df_page2_local["CLH"] / df_page2_local["TLH"].replace(0, np.nan)).fillna(0)
                df_page2_local["TLH / CLH"] = (df_page2_local["TLH"] / df_page2_local["CLH"].replace(0, np.nan)).fillna(0)
                df_page2_local["Normalized TLH - CLH"] = ((df_page2_local["TLH"] - df_page2_local["CLH"]) / 
                                              (df_page2_local["TLH"] + df_page2_local["CLH"]).replace(0, np.nan)).fillna(0)
                
                st.write("Derived columns (TLH/CLH based), if applicable, were computed on the local copy for this page:")
                # Display only relevant columns if they were computed
                derived_cols_to_show = [col for col in ["TLH", "CLH", "TLH - CLH", "CLH / TLH", "TLH / CLH", "Normalized TLH - CLH"] if col in df_page2_local.columns]
                if derived_cols_to_show:
                     st.dataframe(df_page2_local[derived_cols_to_show].head(), use_container_width=True)
            else:
                st.warning("After attempting numeric conversion for TLH/CLH, no data remained. Derived features not computed.")

        except Exception as e:
            st.error(f"Error computing derived TLH/CLH columns on this page: {e}")
    else:
        st.info("Original columns 'TLH' and/or 'CLH' not found in the loaded data. Skipping TLH/CLH-based derived feature calculation on this page.")
    
    # IMPORTANT: For the 5PL fitting, use the df that has the derived columns added (df_page2_local)
    # Or, if you prefer the 5PL function to not see these page-2-specific derived cols, pass the original `df` from session state.
    # For consistency, let's assume the 5PL function should operate on the most complete df available on this page.
    df_for_fitting = df_page2_local # Use this df for column selection and fitting.
    
st.markdown("---")

# --- Column Selection for 5PL Fit (using df_for_fitting) ---
st.markdown("### 2. Select Columns for 5PL Fitting")
# Filter for numeric columns for X and Y axes from the DataFrame prepared for fitting
numeric_cols = df_for_fitting.select_dtypes(include=np.number).columns.tolist()
# Grouping column can be numeric or categorical
all_cols = df_for_fitting.columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in the data available for fitting. Please check your data processing on Page 1 or the uploaded file.")
    st.stop()

# Try to pre-select based on common names if they exist, otherwise default to first available
def get_col_index(cols_list, target_name, default_index=0):
    try:
        return cols_list.index(target_name)
    except ValueError:
        return default_index if cols_list else 0 # default_index if list not empty, else 0

x_col_default_idx = get_col_index(numeric_cols, "progesterone", 0) # Example: try to find "progesterone"
y_col_default_idx = get_col_index(numeric_cols, "TLH_normalized", 1 if len(numeric_cols) > 1 else 0) # Example

x_col = st.selectbox("Select **concentration (X-axis)** column (must be numeric)", numeric_cols, index=x_col_default_idx)
y_col = st.selectbox("Select **signal (Y-axis)** column (must be numeric)", numeric_cols, index=y_col_default_idx)

group_col_options = [None] + all_cols
group_col_default_idx = get_col_index(group_col_options, "group", 0) # Example: try to find "group"
group_col = st.selectbox("Optional: Select **grouping** column (for separate fits per group)", group_col_options, index=group_col_default_idx)


if not x_col or not y_col:
    st.warning("Please select both X (concentration) and Y (signal) columns.")
    st.stop()

# --- Perform 5PL Fitting ---
st.markdown("---")
st.markdown("### 3. 5PL Fit Results")

if st.button("Run 5PL Fitting and Analysis", type="primary"):
    with st.spinner("Performing 5PL fitting... This may take a moment."):
        # Pass df_for_fitting which includes any page-2 derived columns
        results, plots, overlay_fig, ic50_dict = fit_5pl_and_ic50(
            df_for_fitting, x_col, y_col, group_col, return_plotly=True
        )
    
    st.session_state['5pl_results'] = results
    st.session_state['5pl_plots'] = plots
    st.session_state['5pl_overlay_fig'] = overlay_fig
    st.session_state['5pl_ic50_dict'] = ic50_dict
    st.success("5PL Fitting Complete!")

# --- Display Results (No changes from here downwards in this section's logic) ---
if '5pl_results' in st.session_state:
    results = st.session_state['5pl_results']
    plots = st.session_state['5pl_plots']
    overlay_fig = st.session_state['5pl_overlay_fig']
    ic50_dict = st.session_state['5pl_ic50_dict']

    st.markdown("#### Combined 5PL Overlay Plot by Group")
    st.plotly_chart(overlay_fig, use_container_width=True)
    st.markdown("---")
    
    with st.expander("🔍 Individual Fit Results & Parameters", expanded=True):
        cv_summary = {}
        ic50_summary = {} 
        cv_raw_data = [] 

        for key in results:
            st.markdown(f"#### Fit for Group: `{key}`")
            if "error" in results[key]:
                st.error(f"Could not fit model for group {key}: {results[key]['error']}")
                if plots.get(key):
                    st.plotly_chart(plots[key], use_container_width=True)
                continue

            params_df = pd.DataFrame([results[key]])
            param_order = ["a (Min Asymptote)", "b (Hill's Slope)", "c (IC50)", 
                           "d (Max Asymptote)", "g (Asymmetry Factor)", "R2", "Mean CV (%)"]
            display_cols = [p for p in param_order if p in params_df.columns]
            
            # Dynamically create formatters for available columns to avoid errors if a param is missing
            float_formatters = {}
            for p_name, fmt_str in [("a (Min Asymptote)", "{:.3e}"), ("c (IC50)", "{:.3e}"), ("d (Max Asymptote)", "{:.3e}")]:
                if p_name in display_cols: float_formatters[p_name] = fmt_str
            for p_name, fmt_str in [("b (Hill's Slope)", "{:.3f}"), ("g (Asymmetry Factor)", "{:.3f}"), ("R2", "{:.3f}"), ("Mean CV (%)", "{:.3f}")]:
                 if p_name in display_cols: float_formatters[p_name] = fmt_str
            
            st.dataframe(params_df[display_cols].style.format(float_formatters), use_container_width=True)

            r2_val = results[key].get("R2")
            mean_cv_val = results[key].get("Mean CV (%)")
            ic50_val = results[key].get("c (IC50)")

            if r2_val is not None: st.metric(label=f"R² for {key}", value=f"{r2_val:.4f}")
            if mean_cv_val is not None:
                st.metric(label=f"Mean CV% for {key}", value=f"{mean_cv_val:.2f}%")
                cv_summary[key] = mean_cv_val
            if ic50_val is not None: ic50_summary[key] = ic50_val

            if plots.get(key): st.plotly_chart(plots[key], use_container_width=True)
            
            cv_by_conc_list = results[key].get("CV by Concentration (%)", [])
            for record in cv_by_conc_list:
                cv_raw_data.append({"Group": key, "Concentration": record["Concentration"], "CV (%)": record["CV (%)"]})
            st.markdown("---")


    st.markdown("### 4. Summary Visualizations")

    if cv_raw_data:
        st.markdown("#### CV% by Group and Concentration")
        cv_df = pd.DataFrame(cv_raw_data)
        try:
            fig_cv_conc = go.Figure()
            unique_concentrations = sorted(cv_df["Concentration"].unique())
            for conc_val in unique_concentrations:
                subset = cv_df[cv_df["Concentration"] == conc_val]
                fig_cv_conc.add_trace(go.Bar(x=subset["Group"], y=subset["CV (%)"], name=f"Conc: {conc_val:.2e}"))
            fig_cv_conc.update_layout(barmode="group", xaxis_title="Group", yaxis_title="CV (%)", title_text="CV% by Group and Concentration Level", legend_title_text="Concentration")
            st.plotly_chart(fig_cv_conc, use_container_width=True)
        except Exception as e: st.error(f"Could not plot CV by concentration: {e}")

    if cv_summary:
        st.markdown("#### Mean CV% Summary by Group")
        summary_fig = go.Figure()
        summary_fig.add_trace(go.Bar(x=list(cv_summary.keys()), y=list(cv_summary.values()), name="Mean CV%", marker_color='teal'))
        summary_fig.update_layout(xaxis_title="Group", yaxis_title="Mean CV%", title_text="Average CV% Across All Concentration Levels per Group")
        st.plotly_chart(summary_fig, use_container_width=True)

    if ic50_summary:
        st.markdown("#### IC50 Summary by Group (from 5PL Fit)")
        valid_ic50_summary = {k: v for k, v in ic50_summary.items() if pd.notna(v)}
        if valid_ic50_summary:
            sorted_ic50 = sorted(valid_ic50_summary.items(), key=lambda x: x[1])
            groups_sorted = [x[0] for x in sorted_ic50]
            ic50_values_sorted = [x[1] for x in sorted_ic50]
            best_group = groups_sorted[0]
            best_ic50 = ic50_values_sorted[0]
            range_min = best_ic50 / 10 
            range_max = best_ic50 * 10
            ic50_bar_fig = go.Figure()
            for idx, (g, v) in enumerate(sorted_ic50): 
                color = "crimson" if g == best_group else "cornflowerblue"
                ic50_bar_fig.add_trace(go.Bar(x=[v], y=[f"{g}"], orientation="h", name=g, marker_color=color, text=f"IC50 = {v:.3e}", textposition="auto"))
            ic50_bar_fig.update_yaxes(categoryorder="total ascending")
            subtitle = (f"Group '{best_group}' shows the lowest IC50: {best_ic50:.3e}. Approx. analytical range: {range_min:.2e} – {range_max:.2e}.")
            ic50_bar_fig.update_layout(title={"text": f"IC50 Values by Group (Sorted by Sensitivity)<br><sup>{subtitle}</sup>", "x": 0.5, "xanchor": "center"}, xaxis_title="IC50 Value (Concentration Units, log scale suggested for axis)", yaxis_title="Group", xaxis_type="log", title_font_size=18, margin=dict(l=150, r=40, t=100, b=40), showlegend=False, bargap=0.1)
            st.plotly_chart(ic50_bar_fig, use_container_width=True)
            if group_col: st.session_state["ic50_by_group"] = ic50_dict 
            else:
                single_ic50 = ic50_dict.get("all_data") # Backend uses "all_data" if no group_col
                if single_ic50 is not None: st.session_state["ic50_by_group"] = {"all_data": single_ic50}
        else: st.warning("No valid IC50 values were found to plot the summary.")
else:
    st.info("Click the 'Run 5PL Fitting and Analysis' button to see results.")

st.markdown("---")
st.caption("5PL Model: Y = d + (a - d) / ((1 + (X / c)^b)^g)")
