# pages/3_RSM_and_Regression.py
import streamlit as st
import pandas as pd
from core.rsm_model import fit_rsm_model, run_anova
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures

st.header("🔪 RSM: Response Surface Modeling")

# Step 1: Select Y source
y_mode = st.radio("Select Y axis mode", ["IC50", "Slope at concentration"], horizontal=True)
group_col = st.selectbox("Select group column", options=st.session_state.get("display_df", pd.DataFrame()).columns)

y_series = None
if y_mode == "IC50":
    if "ic50_by_group" in st.session_state:
        st.success("Using IC50 values from Page 2.")
        y_series = pd.Series(st.session_state["ic50_by_group"])
        y_series.index = y_series.index.astype(str)
    else:
        st.warning("IC50 values not found. Please run 4PL analysis first.")
        st.stop()
else:
    st.warning("Only IC50 mode is implemented in this version.")
    st.stop()

st.markdown("---")
st.subheader("🧬 Define Experimental Conditions")
group_labels = list(y_series.dropna().index)

# User input for variable names
custom_vars_raw = st.text_input("Enter names of input variables (comma-separated)", value="Fab,BSA")
confirmed = st.button("✅ Confirm Variables")
if confirmed:
    st.session_state["rsm_input_confirmed"] = True
    st.session_state["rsm_input_vars"] = [v.strip() for v in custom_vars_raw.split(",") if v.strip()]
    st.session_state["rsm_input_df"] = pd.DataFrame({"group": group_labels})
    for var in st.session_state["rsm_input_vars"]:
        st.session_state["rsm_input_df"][var] = [None] * len(group_labels)

if st.session_state.get("rsm_input_confirmed"):
    st.markdown("### Input Table")
    input_df = st.data_editor(st.session_state["rsm_input_df"], use_container_width=True, key="input_table")
    if st.button("✅ Save Inputs"):
        st.session_state["rsm_input_df"] = input_df.copy()

if "rsm_input_df" not in st.session_state:
    st.warning("Please input and confirm variables above first.")
    st.stop()

st.markdown("---")
input_df = st.session_state["rsm_input_df"].copy()
usable_inputs = [col for col in input_df.columns if col != "group"]
x_cols = st.multiselect("Select input variables for modeling", usable_inputs, default=usable_inputs)

y_series.index = y_series.index.astype(str)
input_df["group"] = input_df["group"].astype(str)
merged = input_df.set_index("group").join(y_series.rename("Y"), how="inner")
merged.dropna(inplace=True)

st.subheader("📈 Model Results")
if len(merged) < 3:
    st.error("Not enough data to fit a regression model. Fill in more input values.")
    st.dataframe(merged)
else:
    model, y_pred, metrics = fit_rsm_model(merged, x_cols, "Y")

    coef_str = [f"{v:.2f} \\cdot {k}" for k, v in metrics["coefficients"].items()]
    intercept = metrics["intercept"]
    latex_eq = f"Y = {intercept:.2f} + " + " + ".join(coef_str)
    st.latex(latex_eq)

    coefs = pd.Series(metrics["coefficients"])
    fig_coef = px.bar(
        coefs.abs().sort_values(ascending=False),
        orientation='v',
        labels={'value': 'Absolute Coefficient'},
        title="🔎 Coefficient Importance"
    )
    st.plotly_chart(fig_coef, use_container_width=True, key="coef_plot")

    if len(x_cols) > 2:
        st.markdown("### 🔭 2D Slice Viewer")
        fixed_vars = {}
        var_x = st.selectbox("Select X-axis variable", x_cols, index=0, key="slice_x")
        remaining = [v for v in x_cols if v != var_x]
        var_y = st.selectbox("Select Y-axis variable", remaining, index=0, key="slice_y")

        for var in x_cols:
            if var != var_x and var != var_y:
                try:
                    default_val = float(pd.to_numeric(merged[var], errors='coerce').mean())
                except:
                    default_val = 0.0
                fixed_val = st.number_input(f"Set value for {var} (fixed):", value=default_val, key=f"fixed_{var}")
                fixed_vars[var] = fixed_val

        x_range = np.linspace(merged[var_x].astype(float).min(), merged[var_x].astype(float).max(), 50)
        y_range = np.linspace(merged[var_y].astype(float).min(), merged[var_y].astype(float).max(), 50)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        grid = pd.DataFrame({var_x: x_grid.ravel(), var_y: y_grid.ravel()})
        for k, v in fixed_vars.items():
            grid[k] = v

        poly = PolynomialFeatures(degree=2, include_bias=False)
        z_grid = model.predict(poly.fit_transform(grid)).reshape(x_grid.shape)

        fig_slice = go.Figure(data=[
            go.Surface(z=z_grid, x=x_grid, y=y_grid, colorscale='Viridis', showscale=False, opacity=0.7)
        ])
        fig_slice.update_layout(scene=dict(
            xaxis_title=var_x,
            yaxis_title=var_y,
            zaxis_title="Predicted Y"
        ))
        st.plotly_chart(fig_slice, use_container_width=True, key="slice_view")

    st.json(metrics)

    if len(x_cols) == 1:
        fig = px.scatter(merged, x=x_cols[0], y="Y", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    elif len(x_cols) == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        x_range = np.linspace(merged[x_cols[0]].astype(float).min(), merged[x_cols[0]].astype(float).max(), 50)
        y_range = np.linspace(merged[x_cols[1]].astype(float).min(), merged[x_cols[1]].astype(float).max(), 50)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        grid_df = pd.DataFrame({x_cols[0]: x_grid.ravel(), x_cols[1]: y_grid.ravel()})
        z_pred = model.predict(poly.fit_transform(grid_df)).reshape(x_grid.shape)

        fig = go.Figure()
        fig.add_trace(go.Surface(
            z=z_pred,
            x=x_grid,
            y=y_grid,
            showscale=False,
            colorscale='Viridis',
            opacity=0.4,
            name='Surface Fit'
        ))
        fig.add_trace(go.Scatter3d(
            x=merged[x_cols[0]],
            y=merged[x_cols[1]],
            z=merged["Y"],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Data Points'
        ))
        st.plotly_chart(fig, use_container_width=True, key="surface_plot")
    else:
        st.info("More than 2 inputs: visualization skipped. See coefficients above.")

    st.subheader("📊 ANOVA")
    anova_table = run_anova(None, input_df[x_cols], merged["Y"])
    st.dataframe(anova_table)
    st.download_button("Download Design + Output", merged.to_csv(index=False), file_name="rsm_input_output.csv")
