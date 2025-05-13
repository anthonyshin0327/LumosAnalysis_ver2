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




# Add this to the end of your existing code in `pages/3_RSM_and_Regression.py`

# Optimization Section
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

st.markdown("---")
st.subheader("🚀 Optimization Module")

if "model" in locals() and model is not None:
    st.info("This module searches for input settings that minimize the IC50 value using the fitted RSM model.")

    resolution = 100  # fixed high resolution for finer optimization

    # Generate search grid
    bounds = {}
    for var in x_cols:
        col_data = merged[var].astype(float)
        bounds[var] = (col_data.min(), col_data.max())

    grid = pd.DataFrame({})
    for var in x_cols:
        grid[var] = np.linspace(bounds[var][0], bounds[var][1], resolution)

    mesh = np.meshgrid(*[grid[col] for col in x_cols])
    flat_grid = pd.DataFrame({var: mesh[i].ravel() for i, var in enumerate(x_cols)})

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(flat_grid)
    preds = model.predict(X_poly)

    flat_grid["Predicted_Y"] = preds
    best_row = flat_grid.loc[preds.argmin()]

    st.markdown("### ✅ Optimal Settings to **Minimize IC50**")
    st.write(best_row.to_frame().rename(columns={0: "Value"}))

    st.markdown("### 📈 Visual Confirmation")
    if len(x_cols) == 2:
        import plotly.express as px
        fig_opt = px.scatter_3d(flat_grid, x=x_cols[0], y=x_cols[1], z="Predicted_Y",
                                color="Predicted_Y", opacity=0.6, title="RSM Prediction Surface")
        fig_opt.add_scatter3d(x=[best_row[x_cols[0]]], y=[best_row[x_cols[1]]], z=[best_row["Predicted_Y"]],
                              mode="markers", marker=dict(size=6, color="red"), name="Optimal")
        st.plotly_chart(fig_opt, use_container_width=True)
    elif len(x_cols) == 1:
        import plotly.graph_objects as go
        fig_1d = go.Figure()
        fig_1d.add_trace(go.Scatter(x=flat_grid[x_cols[0]], y=flat_grid["Predicted_Y"], mode="lines", name="Prediction"))
        fig_1d.add_trace(go.Scatter(x=[best_row[x_cols[0]]], y=[best_row["Predicted_Y"]],
                                    mode="markers", name="Optimal", marker=dict(color="red", size=8)))
        fig_1d.update_layout(title="Optimization Curve")
        st.plotly_chart(fig_1d, use_container_width=True)
    else:
        st.warning("Optimization visualization is only supported for 1 or 2 input variables.")
else:
    st.warning("Please fit a model first to enable optimization.")

# Extrapolated Optimization Section
st.markdown("---")
st.subheader("🧭 Optimization with 30% Extrapolation Window")

if "model" in locals() and model is not None:
    st.info("Searching for optimal input values with 30% extrapolation beyond original bounds.")

    extrap_bounds = {}
    for var in x_cols:
        col_data = merged[var].astype(float)
        min_val, max_val = col_data.min(), col_data.max()
        range_val = max_val - min_val
        extrap_bounds[var] = (
            min_val - 0.3 * range_val,
            max_val + 0.3 * range_val
        )

    extrap_grid = pd.DataFrame({})
    for var in x_cols:
        extrap_grid[var] = np.linspace(extrap_bounds[var][0], extrap_bounds[var][1], resolution)

    extrap_mesh = np.meshgrid(*[extrap_grid[col] for col in x_cols])
    extrap_flat = pd.DataFrame({var: extrap_mesh[i].ravel() for i, var in enumerate(x_cols)})

    X_poly_extrap = poly.fit_transform(extrap_flat)
    preds_extrap = model.predict(X_poly_extrap)

    extrap_flat["Predicted_Y"] = preds_extrap
    best_row_extrap = extrap_flat.loc[preds_extrap.argmin()]

    st.markdown("### ✅ Optimal Settings with Extrapolation")
    st.write(best_row_extrap.to_frame().rename(columns={0: "Value"}))

    st.markdown("### 📉 Visual Confirmation (Extrapolated Region)")
    if len(x_cols) == 2:
        fig_extrap = px.scatter_3d(extrap_flat, x=x_cols[0], y=x_cols[1], z="Predicted_Y",
                                   color="Predicted_Y", opacity=0.6,
                                   title="RSM Prediction with 10% Extrapolation")
        fig_extrap.add_scatter3d(
            x=[best_row_extrap[x_cols[0]]],
            y=[best_row_extrap[x_cols[1]]],
            z=[best_row_extrap["Predicted_Y"]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Optimal"
        )
        st.plotly_chart(fig_extrap, use_container_width=True)
    elif len(x_cols) == 1:
        fig_1d_extrap = go.Figure()
        fig_1d_extrap.add_trace(go.Scatter(x=extrap_flat[x_cols[0]], y=extrap_flat["Predicted_Y"],
                                           mode="lines", name="Prediction"))
        fig_1d_extrap.add_trace(go.Scatter(x=[best_row_extrap[x_cols[0]]],
                                           y=[best_row_extrap["Predicted_Y"]],
                                           mode="markers", name="Optimal",
                                           marker=dict(color="red", size=8)))
        fig_1d_extrap.update_layout(title="Extrapolated Optimization Curve")
        st.plotly_chart(fig_1d_extrap, use_container_width=True)
    else:
        st.warning("Extrapolated visualization only supported for 1 or 2 inputs.")
else:
    st.warning("Please fit a model first to run extrapolated optimization.")


