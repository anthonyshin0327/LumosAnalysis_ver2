# core/rsm_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.optimize import curve_fit

def expand_group_to_inputs(df, group_col, user_mapping):
    """
    Replace group labels in df[group_col] with columns representing actual variable inputs.
    user_mapping: dict of group_label -> dict of variable_name -> value
    """
    expanded = df[[group_col]].copy()
    for var in next(iter(user_mapping.values())).keys():
        expanded[var] = expanded[group_col].map(lambda g: user_mapping.get(str(g), {}).get(var, np.nan))
    return expanded

def four_param_logistic(x, a, b, c, d):
    return ((a - d) / (1.0 + ((x / c) ** b))) + d

def calculate_ic50(df, group_col, conc_col, signal_col):
    ic50_dict = {}
    for group, subset in df.groupby(group_col):
        x = subset[conc_col].astype(float).values
        y = subset[signal_col].astype(float).values
        if len(np.unique(x)) < 4:
            ic50_dict[group] = np.nan  # not enough points to fit 4PL
            continue
        try:
            p0 = [max(y), 1, np.median(x), min(y)]
            popt, _ = curve_fit(four_param_logistic, x, y, p0, maxfev=10000)
            ic50_dict[group] = popt[2]  # c = IC50
        except Exception:
            ic50_dict[group] = np.nan
    return ic50_dict

def calculate_slope_at(df, group_col, conc_col, signal_col, target_conc):
    slope_dict = {}
    for group, subset in df.groupby(group_col):
        x = subset[conc_col].astype(float).values
        y = subset[signal_col].astype(float).values
        if len(np.unique(x)) < 4:
            slope_dict[group] = np.nan
            continue
        try:
            p0 = [max(y), 1, np.median(x), min(y)]
            popt, _ = curve_fit(four_param_logistic, x, y, p0, maxfev=10000)
            a, b, c, d = popt
            # derivative of 4PL at x0
            x0 = target_conc
            term = (x0 / c) ** b
            dydx = ((a - d) * b * term) / (x0 * (1 + term) ** 2)
            slope_dict[group] = dydx
        except Exception:
            slope_dict[group] = np.nan
    return slope_dict

def fit_rsm_model(df, x_cols, y_col):
    X = df[x_cols].values
    y = df[y_col].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    metrics = {
        "R2": model.score(X_poly, y),
        "coefficients": dict(zip(poly.get_feature_names_out(x_cols), model.coef_)),
        "intercept": model.intercept_
    }

    return model, y_pred, metrics

def plot_rsm_surface(model, X, y, x_cols):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    x1, x2 = np.meshgrid(
        np.linspace(X[x_cols[0]].min(), X[x_cols[0]].max(), 50),
        np.linspace(X[x_cols[1]].min(), X[x_cols[1]].max(), 50)
    )
    X_grid = np.c_[x1.ravel(), x2.ravel()]
    X_grid_poly = poly.transform(X_grid)
    y_pred_grid = model.predict(X_grid_poly).reshape(x1.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, y_pred_grid, cmap='viridis', alpha=0.7)
    ax.scatter(X[x_cols[0]], X[x_cols[1]], y, color='red')
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    ax.set_zlabel("Response")
    ax.set_title("RSM Surface Fit")
    return fig

def run_anova(model, X, y):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np

    if X.empty or y.empty:
        raise ValueError("ANOVA error: input X or y is empty.")
    if len(X) != len(y):
        raise ValueError(f"ANOVA error: X has {len(X)} rows, but y has {len(y)}.")

    X = X.copy()
    X.columns = [str(col).replace(" ", "_").replace("^", "_pow_").replace("*", "_mul_") for col in X.columns]

    df = X.copy()
    df["Y"] = y.reset_index(drop=True)

    # Force all values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop bad rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Final check
    if df.isna().any().any():
        raise ValueError("ANOVA error: Data still contains NaNs after cleanup.")

    formula = f"Y ~ {' * '.join([col for col in df.columns if col != 'Y'])}"

    try:
        ols_model = smf.ols(formula=formula, data=df).fit()
        return sm.stats.anova_lm(ols_model, typ=2)
    except Exception as e:
        snapshot = df.to_string(index=False)
        raise ValueError(f"ANOVA model failed: {str(e)}\nFormula: {formula}\nData snapshot:\n{snapshot}")
