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
            popt, pcov = curve_fit(four_param_logistic, x, y, p0, maxfev=10000)
            ic50 = popt[2]
            from sklearn.metrics import r2_score
            y_pred = four_param_logistic(x, *popt)
            r2 = r2_score(y, y_pred)

            # Compute CV% across each concentration level (technical replicate precision)
            subset["x"] = subset[x_col].astype(float)
            subset["y"] = subset[y_col].astype(float)
            cv_per_level = subset.groupby("x")["y"].agg(["mean", "std"])
            cv_per_level["cv"] = (cv_per_level["std"] / cv_per_level["mean"]).replace([np.inf, -np.inf], np.nan) * 100
            cv_by_concentration = cv_per_level["cv"].dropna().reset_index().rename(columns={"x": "Concentration", "cv": "CV (%)"})
            mean_cv = np.nanmean(cv_by_concentration["CV (%)"]) if not cv_by_concentration.empty else np.nan

            results[str(group)] = {
                "a": popt[0],
                "b": popt[1],
                "c (IC50)": ic50,
                "d": popt[3],
                "R2": r2,
                "Mean CV (%)": mean_cv,
                "CV by Concentration (%)": cv_by_concentration.to_dict(orient="records")
            }

            ic50_dict[str(group)] = ic50

            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = four_param_logistic(x_fit, *popt)

            from scipy.stats import t

            # Compute 95% confidence interval band
            dof = max(0, len(x) - len(popt))
            alpha = 0.05
            tval = t.ppf(1.0 - alpha / 2., dof) if dof > 0 else 1.96

            # Estimate gradient (Jacobian) for CI propagation
            J = np.zeros((len(x_fit), len(popt)))
            eps = 1e-8
            for i in range(len(popt)):
                dp = np.zeros_like(popt)
                dp[i] = eps
                y_hi = four_param_logistic(x_fit, *(popt + dp))
                y_lo = four_param_logistic(x_fit, *(popt - dp))
                J[:, i] = (y_hi - y_lo) / (2 * eps)

            se_fit = np.sqrt(np.sum(J @ pcov * J, axis=1))
            ci_upper = y_fit + tval * se_fit
            ci_lower = y_fit - tval * se_fit

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
