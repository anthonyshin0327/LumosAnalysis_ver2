import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# Assuming the backend code is saved in core/five_param_fit.py
from core.five_param_fit import fit_5pl_and_ic50 # Updated import

st.set_page_config(layout="wide") # Use wide layout for better plot display

st.header("📊 5PL Curve Fitting & IC50 Analysis") # Updated to 5PL

# Check if data is loaded from a previous step (e.g., Page 1)
if "display_df" not in st.session_state or st.session_state["display_df"] is None:
    st.warning("Please upload and preprocess data on Page 1 first, or upload a CSV/Excel file below.")
    
    # Allow file upload on this page as a fallback or for direct use
    uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state["display_df"] = df
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.stop() # Stop if no data is available

df = st.session_state["display_df"]

# --- Data Preprocessing & Derived Columns ---
st.markdown("### 1. Data Preview & Preparation")
with st.expander("Show/Hide Raw Data and Derived Columns", expanded=False):
    st.dataframe(df.head(), use_container_width=True)

    # Compute derived features if TLH and CLH exist
    # Ensure these columns are of numeric type before operations
    if "TLH" in df.columns and "CLH" in df.columns:
        try:
            df["TLH"] = pd.to_numeric(df["TLH"], errors='coerce')
            df["CLH"] = pd.to_numeric(df["CLH"], errors='coerce')

            # Drop rows where TLH or CLH could not be converted to numeric if they are essential
            df.dropna(subset=["TLH", "CLH"], inplace=True)

            if not df.empty: # Proceed if df is not empty after potential NA drop
                df["TLH - CLH"] = df["TLH"] - df["CLH"]
                # Handle potential division by zero or NaN/Inf results
                df["CLH / TLH"] = (df["CLH"] / df["TLH"].replace(0, np.nan)).fillna(0)
                df["TLH / CLH"] = (df["TLH"] / df["CLH"].replace(0, np.nan)).fillna(0)
                df["Normalized TLH - CLH"] = ((df["TLH"] - df["CLH"]) / 
                                              (df["TLH"] + df["CLH"]).replace(0, np.nan)).fillna(0)
                
                st.write("Derived columns (TLH/CLH based) were computed:")
                st.markdown("- `TLH - CLH`\n- `CLH / TLH`\n- `TLH / CLH`\n- `Normalized TLH - CLH`")
                st.dataframe(df[["TLH", "CLH", "TLH - CLH", "CLH / TLH", "TLH / CLH", "Normalized TLH - CLH"]].head(), use_container_width=True)
            else:
                st.warning("TLH or CLH columns became all NaNs after numeric conversion or were empty. Derived features not computed.")

        except Exception as e:
            st.error(f"Error computing derived columns: {e}. Please ensure TLH and CLH are numeric.")
    else:
        st.info("Columns 'TLH' and/or 'CLH' not found. Skipping derived feature calculation.")

st.markdown("---")

# --- Column Selection for 5PL Fit ---
st.markdown("### 2. Select Columns for 5PL Fitting")
# Filter for numeric columns for X and Y axes, as 5PL requires numeric input
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols_for_grouping = df.select_dtypes(exclude=np.number).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in the data. Cannot perform 5PL fitting. Please check your data.")
    st.stop()

x_col = st.selectbox("Select **concentration (X-axis)** column (must be numeric)", numeric_cols, index=0 if numeric_cols else None)
y_col = st.selectbox("Select **signal (Y-axis)** column (must be numeric)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0 if numeric_cols else None)

# Grouping column can be numeric or categorical
all_cols = df.columns.tolist()
group_col_options = [None] + all_cols
group_col = st.selectbox("Optional: Select **grouping** column (for separate fits per group)", group_col_options)

if not x_col or not y_col:
    st.warning("Please select both X (concentration) and Y (signal) columns.")
    st.stop()

# --- Perform 5PL Fitting ---
st.markdown("---")
st.markdown("### 3. 5PL Fit Results")

if st.button("Run 5PL Fitting and Analysis", type="primary"):
    with st.spinner("Performing 5PL fitting... This may take a moment."):
        # Run 5PL fitting and compute summary data
        # Pass the dataframe, selected columns, and grouping column
        results, plots, overlay_fig, ic50_dict = fit_5pl_and_ic50(
            df, x_col, y_col, group_col, return_plotly=True
        )
    
    st.session_state['5pl_results'] = results
    st.session_state['5pl_plots'] = plots
    st.session_state['5pl_overlay_fig'] = overlay_fig
    st.session_state['5pl_ic50_dict'] = ic50_dict
    st.success("5PL Fitting Complete!")

# --- Display Results ---
if '5pl_results' in st.session_state:
    results = st.session_state['5pl_results']
    plots = st.session_state['5pl_plots']
    overlay_fig = st.session_state['5pl_overlay_fig']
    ic50_dict = st.session_state['5pl_ic50_dict']

    # Display the combined overlay plot at the top
    st.markdown("#### Combined 5PL Overlay Plot by Group")
    st.plotly_chart(overlay_fig, use_container_width=True)
    
    st.markdown("---")
    
    with st.expander("🔍 Individual Fit Results & Parameters", expanded=True):
        cv_summary = {}
        ic50_summary = {} # For the IC50 summary plot
        cv_raw_data = [] # For CV by concentration plot

        for key in results:
            st.markdown(f"#### Fit for Group: `{key}`")
            if "error" in results[key]:
                st.error(f"Could not fit model for group {key}: {results[key]['error']}")
                # Ensure plot exists for this key even if it's an error plot from backend
                if plots.get(key):
                    st.plotly_chart(plots[key], use_container_width=True)
                continue

            # Displaying parameters
            params_df = pd.DataFrame([results[key]])
            # Select and reorder columns for better display
            param_order = ["a (Min Asymptote)", "b (Hill's Slope)", "c (IC50)", 
                           "d (Max Asymptote)", "g (Asymmetry Factor)", "R2", "Mean CV (%)"]
            # Filter out missing keys from param_order just in case
            display_cols = [p for p in param_order if p in params_df.columns]
            st.dataframe(params_df[display_cols].style.format("{:.3e}", subset=["a (Min Asymptote)", "c (IC50)", "d (Max Asymptote)"])
                                                       .format("{:.3f}", subset=["b (Hill's Slope)", "g (Asymmetry Factor)", "R2", "Mean CV (%)"]),
                         use_container_width=True)

            r2_val = results[key].get("R2")
            mean_cv_val = results[key].get("Mean CV (%)")
            ic50_val = results[key].get("c (IC50)")

            if r2_val is not None:
                st.metric(label=f"R² for {key}", value=f"{r2_val:.4f}")
            if mean_cv_val is not None:
                st.metric(label=f"Mean CV% for {key}", value=f"{mean_cv_val:.2f}%")
                cv_summary[key] = mean_cv_val # For overall CV summary plot
            if ic50_val is not None:
                ic50_summary[key] = ic50_val # For overall IC50 summary plot

            # Display individual plot if it exists
            if plots.get(key):
                st.plotly_chart(plots[key], use_container_width=True)
            
            # Collect CV by concentration data
            cv_by_conc_list = results[key].get("CV by Concentration (%)", [])
            for record in cv_by_conc_list:
                cv_raw_data.append({
                    "Group": key, 
                    "Concentration": record["Concentration"], 
                    "CV (%)": record["CV (%)"]
                })
            st.markdown("---") # Separator between groups


    # --- Combined Summary Plots ---
    st.markdown("### 4. Summary Visualizations")

    # CV% per concentration bar chart
    if cv_raw_data:
        st.markdown("#### CV% by Group and Concentration")
        cv_df = pd.DataFrame(cv_raw_data)
        # Pivot for easier plotting if many groups/concentrations
        try:
            # Create a bar chart
            fig_cv_conc = go.Figure()
            unique_concentrations = sorted(cv_df["Concentration"].unique())
            
            for conc_val in unique_concentrations:
                subset = cv_df[cv_df["Concentration"] == conc_val]
                fig_cv_conc.add_trace(go.Bar(
                    x=subset["Group"], 
                    y=subset["CV (%)"], 
                    name=f"Conc: {conc_val:.2e}" # Format concentration for legend
                ))
            
            fig_cv_conc.update_layout(
                barmode="group", 
                xaxis_title="Group", 
                yaxis_title="CV (%)", 
                title_text="CV% by Group and Concentration Level",
                legend_title_text="Concentration"
            )
            st.plotly_chart(fig_cv_conc, use_container_width=True)
        except Exception as e:
            st.error(f"Could not plot CV by concentration: {e}")


    # Mean CV% by Group bar chart
    if cv_summary:
        st.markdown("#### Mean CV% Summary by Group")
        summary_fig = go.Figure()
        summary_fig.add_trace(go.Bar(
            x=list(cv_summary.keys()), 
            y=list(cv_summary.values()), 
            name="Mean CV%",
            marker_color='teal'
        ))
        summary_fig.update_layout(
            xaxis_title="Group", 
            yaxis_title="Mean CV%", 
            title_text="Average CV% Across All Concentration Levels per Group"
        )
        st.plotly_chart(summary_fig, use_container_width=True)

    # Summary plot: IC50 by group (Horizontal Bar Chart)
    if ic50_summary:
        st.markdown("#### IC50 Summary by Group (from 5PL Fit)")
        
        # Filter out NaN IC50 values before sorting
        valid_ic50_summary = {k: v for k, v in ic50_summary.items() if pd.notna(v)}

        if valid_ic50_summary:
            # Prepare data sorted by IC50 ascending
            sorted_ic50 = sorted(valid_ic50_summary.items(), key=lambda x: x[1])
            groups_sorted = [x[0] for x in sorted_ic50]
            ic50_values_sorted = [x[1] for x in sorted_ic50]

            best_group = groups_sorted[0]
            best_ic50 = ic50_values_sorted[0]

            # Approximate analytical range (example: ±1 log from IC50)
            # This is a rule of thumb; actual analytical range depends on assay precision and curve shape.
            # For 5PL, this might be more complex if asymmetry is high.
            range_min = best_ic50 / 10 
            range_max = best_ic50 * 10

            ic50_bar_fig = go.Figure()
            # Iterate in reverse for the plot so that the smallest IC50 (best) is at the top
            for idx, (g, v) in enumerate(sorted_ic50): 
                color = "crimson" if g == best_group else "cornflowerblue"
                ic50_bar_fig.add_trace(go.Bar(
                    x=[v],
                    y=[f"{g}"], # Use group name directly
                    orientation="h",
                    name=g,
                    marker_color=color,
                    text=f"IC50 = {v:.3e}", # Use scientific notation for IC50
                    textposition="auto" 
                ))
            
            # Sort y-axis by IC50 value by setting categoryorder
            ic50_bar_fig.update_yaxes(categoryorder="total ascending")


            subtitle = (f"Group '{best_group}' shows the lowest IC50: {best_ic50:.3e}. "
                        f"Approx. analytical range based on this IC50: {range_min:.2e} – {range_max:.2e}.")

            ic50_bar_fig.update_layout(
                title={"text": f"IC50 Values by Group (Sorted by Sensitivity)<br><sup>{subtitle}</sup>", "x": 0.5, "xanchor": "center"},
                xaxis_title="IC50 Value (Concentration Units, log scale suggested for axis)",
                yaxis_title="Group",
                xaxis_type="log", # Display IC50s on a log scale for better visualization if they span orders of magnitude
                title_font_size=18,
                margin=dict(l=150, r=40, t=100, b=40), # Adjust left margin for group names
                showlegend=False,
                bargap=0.1
            )
            st.plotly_chart(ic50_bar_fig, use_container_width=True)
            
            # Save IC50s in session state for downstream reuse
            if group_col: # Only save if grouping was actually used
                st.session_state["ic50_by_group"] = ic50_dict 
            else: # If no grouping, save the single IC50 value if it exists
                # The backend returns "all_data" as key if no group_col
                single_ic50 = ic50_dict.get("all_data") 
                if single_ic50 is not None:
                     st.session_state["ic50_by_group"] = {"all_data": single_ic50}


        else:
            st.warning("No valid IC50 values were found to plot the summary.")

else:
    st.info("Click the 'Run 5PL Fitting and Analysis' button to see results.")

st.markdown("---")
st.caption("5PL Model: Y = d + (a - d) / ((1 + (X / c)^b)^g)")
