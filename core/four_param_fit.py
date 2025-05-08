# core/four_param_fit.py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

def four_param_logistic(x, a, b, c, d):
    return ((a - d) / (1.0 + ((x / c) ** b))) + d

def fit_4pl_and_ic50(df, x_col, y_col, group_col=None, return_plotly=False):
    results = {}
    plots = {}
    ic50_dict = {}
    overlay_fig = go.Figure()

    groups = df[group_col].unique() if group_col else ["all"]

    for group in groups:
        if group_col:
            subset = df[df[group_col] == group]
        else:
            subset = df

        subset = subset[[x_col, y_col]].dropna()
        x = subset[x_col].astype(float).values
        y = subset[y_col].astype(float).values

        try:
            p0 = [max(y), 1, np.median(x), min(y)]
            popt, _ = curve_fit(four_param_logistic, x, y, p0, maxfev=10000)
            ic50 = popt[2]
            results[str(group)] = {
                "a": popt[0], "b": popt[1], "c (IC50)": ic50, "d": popt[3]
            }
            ic50_dict[str(group)] = ic50

            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = four_param_logistic(x_fit, *popt)

            if return_plotly:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Raw Data"))
                fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name="4PL Fit"))
                fig.add_vline(x=ic50, line_dash="dash", line_color="gray", annotation_text=f"IC50={ic50:.2f}")
                fig.update_layout(title=f"4PL Fit: {group}", xaxis_title=x_col, yaxis_title=y_col)
                plots[str(group)] = fig

                overlay_fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name=f"{group}"))
            else:
                plots[str(group)] = None
        except Exception as e:
            results[str(group)] = {"error": str(e)}
            ic50_dict[str(group)] = np.nan
            plots[str(group)] = None

    return results, plots, overlay_fig, ic50_dict