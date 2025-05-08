# pages/3_RSM_and_Regression.py
import streamlit as st
import pandas as pd
import plotly.express as px
from core.rsm_model import fit_rsm_model, run_anova

st.header("🧪 RSM: Response Surface Modeling")

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

# Step 2: Define inputs using editable table
st.markdown("---")
st.subheader("🧬 Define Experimental Conditions")
group_labels = list(y_series.dropna().index)

# Let user specify which variables to include in the input table
if "rsm_input_confirmed" not in st.session_state:
    st.session_state["rsm_input_confirmed"] = False

custom_vars_raw = st.text_input("Enter names of input variables (comma-separated)", value="Fab,BSA")
confirmed = st.button("✅ Confirm Variables")
if confirmed:
    st.session_state["rsm_input_confirmed"] = True
    st.session_state["rsm_input_vars"] = [v.strip() for v in custom_vars_raw.split(",") if v.strip()]

custom_vars = st.session_state.get("rsm_input_vars", [])

if ("rsm_input_df" not in st.session_state) or (not st.session_state["rsm_input_confirmed"]) or (set(st.session_state["rsm_input_df"].columns) != set(["group"] + custom_vars)) or (set(st.session_state["rsm_input_df"].group.tolist()) != set(group_labels)):
    st.session_state["rsm_input_df"] = pd.DataFrame({"group": group_labels})
    for var in custom_vars:
        st.session_state["rsm_input_df"][var] = [None] * len(group_labels)

if st.session_state["rsm_input_confirmed"]:
    input_df = st.data_editor(
        st.session_state["rsm_input_df"].copy(),
        key="rsm_input_editor",
        num_rows="dynamic",
        use_container_width=True
    )
else:
    st.warning("Please confirm your input variables before continuing.")
# Keep table edits after confirmation only
if st.session_state["rsm_input_confirmed"]:
    st.session_state["rsm_input_df"] = input_df.copy()

if confirmed:
    st.session_state["rsm_input_df"] = input_df.copy()
    st.session_state["rsm_input_vars"] = custom_vars

# Step 3: Confirm input values separately
st.markdown("---")
if st.session_state.get("rsm_input_confirmed"):
    if st.button("✅ Confirm Input Values"):
        st.session_state["rsm_final_df"] = st.session_state["rsm_input_df"].copy()

# Step 4: Assemble model dataframe
if "rsm_final_df" not in st.session_state:
    st.warning("Please confirm input values before modeling.")
    st.stop()

input_df = st.session_state["rsm_final_df"].copy()
usable_inputs = [col for col in input_df.columns if col != "group"]
x_cols = st.multiselect("Select input variables for modeling", usable_inputs, default=usable_inputs)

input_df = st.session_state["rsm_final_df"].copy()
input_df["group"] = input_df["group"].astype(str)
y_series.index = y_series.index.astype(str)
merged = input_df.set_index("group").join(y_series.rename("Y"), how="inner")
merged.dropna(inplace=True)

st.markdown("---")
st.subheader("📈 Model Results")
if len(merged) < 3:
    st.error("Not enough data to fit a regression model. Fill in more input values.")
    st.dataframe(merged)
else:
    model, y_pred, metrics = fit_rsm_model(merged, x_cols, "Y")
    st.json(metrics)

    if len(x_cols) == 1:
        fig = px.scatter(merged, x=x_cols[0], y="Y", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    elif len(x_cols) == 2:
        import plotly.graph_objects as go

        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        import numpy as np
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
        equation = " + ".join([f"{coef:.2f}·{term}" for term, coef in metrics['coefficients'].items()])

        equation_latex = f"Y = {equation}"
        title = f"RSM Surface Fit: Y vs. {x_cols[0]} and {x_cols[1]} (R² = {metrics['R2']:.3f})"
        fig.update_layout(
            title=title,
            margin=dict(l=0, r=0, b=0, t=50),
            scene=dict(
                xaxis_title=x_cols[0],
                yaxis_title=x_cols[1],
                zaxis_title="Y"
            ),
            annotations=[
                dict(
                    showarrow=False,
                    text=f"{{{equation_latex}}}",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.05,
                    font=dict(size=12)
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("More than 2 inputs: visualization skipped. See coefficients above.")

    st.subheader("📊 ANOVA")
    anova_table = run_anova(None, input_df[x_cols], merged["Y"])


    st.dataframe(anova_table)

    st.download_button("Download Design + Output", merged.to_csv(index=False), file_name="rsm_input_output.csv")
