import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from scipy.stats import t

def five_param_logistic(x, a, b, c, d, g):
    """
    Five-parameter logistic function (5PL).
    a: Minimum asymptote
    b: Hill's slope (Steepness factor)
    c: Inflection point (IC50 or EC50)
    d: Maximum asymptote
    g: Asymmetry factor
    """
    return d + ((a - d) / ((1.0 + (x / c)**b)**g))

def fit_5pl_and_ic50(df, x_col, y_col, group_col=None, return_plotly=False):
    """
    Fits a 5PL model to the data and calculates IC50.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Name of the column containing X values (e.g., concentration).
        y_col (str): Name of the column containing Y values (e.g., signal).
        group_col (str, optional): Name of the column for grouping data. Defaults to None.
        return_plotly (bool, optional): Whether to return Plotly figure objects. Defaults to False.

    Returns:
        tuple: (results_dict, plots_dict, overlay_figure, ic50_values_dict)
               - results_dict: Dictionary containing fitted parameters and stats for each group.
               - plots_dict: Dictionary containing individual Plotly figures for each group.
               - overlay_figure: A single Plotly figure with all group fits overlaid.
               - ic50_values_dict: Dictionary of IC50 values for each group.
    """
    results = {}
    plots = {}
    ic50_dict = {}
    overlay_fig = go.Figure()

    # Determine unique groups or use 'all' if no grouping column
    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
    else:
        groups = ["all_data"] # Use a default name if no group_col or invalid

    for group in groups:
        group_results = {}
        if group_col and group_col in df.columns and group != "all_data":
            subset = df[df[group_col] == group].copy() # Use .copy() to avoid SettingWithCopyWarning
        else:
            subset = df.copy() # Use .copy()

        # Ensure x and y columns are present and drop NA values for fitting
        if x_col not in subset.columns or y_col not in subset.columns:
            results[str(group)] = {"error": f"X ('{x_col}') or Y ('{y_col}') column not found in subset for group '{group}'."}
            ic50_dict[str(group)] = np.nan
            plots[str(group)] = go.Figure().update_layout(title=f"Error: Data columns not found for {group}")
            continue

        subset = subset[[x_col, y_col]].dropna()
        
        if subset.empty:
            results[str(group)] = {"error": f"No data after NA removal for group '{group}'."}
            ic50_dict[str(group)] = np.nan
            plots[str(group)] = go.Figure().update_layout(title=f"Error: No data for {group}")
            continue

        x = subset[x_col].astype(float).values
        y = subset[y_col].astype(float).values

        if len(x) < 5: # Need at least 5 points for 5 parameters
            results[str(group)] = {"error": f"Not enough data points (need at least 5) for 5PL fit in group '{group}'. Got {len(x)}."}
            ic50_dict[str(group)] = np.nan
            plots[str(group)] = go.Figure().update_layout(title=f"Error: Not enough data for {group}")
            continue
            
        try:
            # Initial parameter guesses for 5PL
            # a (min asymptote), b (Hill's slope), c (IC50), d (max asymptote), g (asymmetry)
            # More robust initial guesses can significantly improve fitting.
            min_y_val = np.min(y) if len(y) > 0 else 0
            max_y_val = np.max(y) if len(y) > 0 else 1
            median_x_val = np.median(x) if len(x) > 0 else 1
            
            p0 = [min_y_val, 1.0, median_x_val, max_y_val, 1.0] 
            
            # Define bounds to constrain parameters if necessary, can help with convergence
            # Example: (a_min, b_min, c_min, d_min, g_min), (a_max, b_max, c_max, d_max, g_max)
            # Bounds need to be carefully chosen based on expected data characteristics.
            # For instance, c (IC50) should be positive. Slopes can be positive/negative.
            # Asymmetry g is typically > 0.
            bounds = (
                [-np.inf, -np.inf, 1e-9, -np.inf, 1e-3], # Lower bounds (c > 0, g > 0)
                [np.inf,  np.inf, np.inf, np.inf, np.inf]    # Upper bounds
            )

            popt, pcov = curve_fit(five_param_logistic, x, y, p0=p0, bounds=bounds, maxfev=20000, method='trf') # 'trf' often better for bounds
            
            ic50 = popt[2]
            y_pred = five_param_logistic(x, *popt)
            r2 = r2_score(y, y_pred)

            # Compute CV% across each concentration level
            # Add a temporary column for original x values to the subset for grouping
            subset.loc[:, '_x_original_for_cv_'] = x 
            cv_per_level = subset.groupby('_x_original_for_cv_')[y_col].agg(['mean', 'std'])
            cv_per_level['cv'] = (cv_per_level['std'].abs() / cv_per_level['mean'].abs()).replace([np.inf, -np.inf], np.nan) * 100
            cv_by_concentration = cv_per_level['cv'].dropna().reset_index().rename(columns={'_x_original_for_cv_': "Concentration", "cv": "CV (%)"})
            mean_cv = np.nanmean(cv_by_concentration["CV (%)"]) if not cv_by_concentration.empty else np.nan

            group_results = {
                "a (Min Asymptote)": popt[0],
                "b (Hill's Slope)": popt[1],
                "c (IC50)": ic50,
                "d (Max Asymptote)": popt[3],
                "g (Asymmetry Factor)": popt[4],
                "R2": r2,
                "Mean CV (%)": mean_cv,
                "CV by Concentration (%)": cv_by_concentration.to_dict(orient="records")
            }
            results[str(group)] = group_results
            ic50_dict[str(group)] = ic50

            if return_plotly:
                x_fit = np.logspace(np.log10(min(x[x>0]) if any(x>0) else 1e-9), np.log10(max(x) if any(x) else 1), 200) # Ensure x_fit is positive for log scale
                y_fit = five_param_logistic(x_fit, *popt)

                # Compute 95% confidence interval band
                # This part is complex and might need careful validation for 5PL
                # The Jacobian calculation needs to be adapted for 5 parameters.
                dof = max(0, len(x) - len(popt)) # degrees of freedom
                alpha = 0.05 # significance level
                tval = t.ppf(1.0 - alpha / 2., dof) if dof > 0 else 1.96 # t-distribution critical value

                # Estimate Jacobian (gradient) for CI propagation
                J = np.zeros((len(x_fit), len(popt)))
                eps = 1e-8 # Epsilon for finite differences
                for i in range(len(popt)):
                    p_plus = popt.copy()
                    p_minus = popt.copy()
                    p_plus[i] += eps
                    p_minus[i] -= eps
                    J[:, i] = (five_param_logistic(x_fit, *p_plus) - five_param_logistic(x_fit, *p_minus)) / (2 * eps)
                
                # Check if pcov is valid (e.g. not all inf or nan)
                if np.any(np.isinf(pcov)) or np.any(np.isnan(pcov)):
                    print(f"Warning: Covariance matrix for group '{group}' contains Inf or NaN. CI calculation might be unreliable.")
                    ci_upper = y_fit # Fallback or skip CI
                    ci_lower = y_fit
                else:
                    try:
                        # Variance of the fitted curve
                        var_fit = np.sum(J @ pcov * J, axis=1)
                        # Avoid negative variances due to numerical issues
                        var_fit[var_fit < 0] = 0 
                        se_fit = np.sqrt(var_fit) # Standard error of the fit
                        ci_upper = y_fit + tval * se_fit
                        ci_lower = y_fit - tval * se_fit
                    except Exception as e_ci:
                        print(f"Error calculating CI for group '{group}': {e_ci}. CI will not be plotted.")
                        ci_upper = y_fit # Fallback
                        ci_lower = y_fit


                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Raw Data", marker=dict(size=8)))
                fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name="5PL Fit", line=dict(color='royalblue', width=2)))
                
                # Add confidence interval band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_fit, x_fit[::-1]]), # x_fit then x_fit reversed
                    y=np.concatenate([ci_upper, ci_lower[::-1]]), # ci_upper then ci_lower reversed
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'), # No border for fill
                    hoverinfo="skip",
                    showlegend=False,
                    name='95% CI'
                ))
                
                fig.add_vline(x=ic50, line_dash="dash", line_color="firebrick", 
                              annotation_text=f"IC50={ic50:.3e}", annotation_position="top left")
                
                param_text = (f"a={popt[0]:.2e}, b={popt[1]:.2f}, c(IC50)={popt[2]:.2e}, <br>"
                              f"d={popt[3]:.2e}, g={popt[4]:.2f}, R²={r2:.3f}")

                fig.update_layout(
                    title=f"5PL Fit: {group}<br><sup>{param_text}</sup>",
                    xaxis_title=x_col + (" (log scale)" if min(x_fit)>0 else ""),
                    yaxis_title=y_col,
                    xaxis_type="log" if min(x_fit)>0 else "linear", # Use log scale if data permits
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                plots[str(group)] = fig
                
                # Add to overlay plot
                overlay_fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name=f"5PL Fit: {group} (IC50={ic50:.2e})"))

        except RuntimeError as e_runtime: # Specific error from curve_fit
            error_msg = f"RuntimeError during 5PL fitting for group '{group}': {e_runtime}. Could not converge or bad initial parameters."
            print(error_msg)
            results[str(group)] = {"error": error_msg}
            ic50_dict[str(group)] = np.nan
            # Create an empty plot with error message
            fig = go.Figure()
            fig.update_layout(title=f"5PL Fit Error: {group}", 
                              xaxis_title=x_col, yaxis_title=y_col,
                              annotations=[dict(text="Could not fit 5PL model.", showarrow=False)])
            plots[str(group)] = fig
        except Exception as e:
            error_msg = f"General error during 5PL fitting for group '{group}': {e}"
            print(error_msg)
            results[str(group)] = {"error": error_msg}
            ic50_dict[str(group)] = np.nan
            fig = go.Figure()
            fig.update_layout(title=f"5PL Fit Error: {group}", 
                              xaxis_title=x_col, yaxis_title=y_col,
                              annotations=[dict(text="An unexpected error occurred.", showarrow=False)])
            plots[str(group)] = fig
            
    # Finalize overlay plot
    overlay_fig.update_layout(
        title="Combined 5PL Overlay Plot by Group",
        xaxis_title=x_col + (" (log scale)" if df[x_col].min() > 0 else ""), # Check original data for log scale decision
        yaxis_title=y_col,
        xaxis_type="log" if df[x_col].min() > 0 else "linear",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return results, plots, overlay_fig, ic50_dict

