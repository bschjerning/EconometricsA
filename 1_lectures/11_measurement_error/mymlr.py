import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

def ols(y, X, robust=False):
    """
    Perform OLS estimation using matrix algebra.

    Parameters:
    y (pd.Series): Dependent variable, dimensions (n x 1)
    X (pd.DataFrame): Design matrix, dimensions (n x p)

    Returns:
    dict: Model results including coefficients, standard errors, residuals, SST, SSR, SSE, R²,
          and variable names for both independent and dependent variables.
    """
    # Extract variable names from X and y
    lbl_X = X.columns.tolist()  # Names of independent variables
    lbl_y = y.name               # Name of dependent variable
    # Ensure correct dimensions: X (n x p), y (n x 1)
    X = X.values  # (n x p)
    y = y.values.reshape(-1, 1)  # (n x 1)

    # OLS estimates: β = (X'X)^(-1) X'y
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y  # (p x 1)

    # Predicted values: ŷ = Xβ
    y_hat = X @ beta_hat  # (n x 1)

    # Residuals: u = y - ŷ
    u_hat = y - y_hat  # (n x 1)

    # Number of observations (n), parameters (p), and residual degrees of freedom
    n, p = X.shape
    df_resid = n - p

    # Sum of Squared Residuals (SSE): u'u
    # The matrix product is 1 x 1; squeeze removes the redundant dimensions
    SSE = np.squeeze(u_hat.T @ u_hat)  # scalar

    # Total Sum of Squares (SST): (y - ȳ)'(y - ȳ)
    y_centered = y - y.mean()
    SST = np.squeeze(y_centered.T @ y_centered)  # scalar

    # Explained Sum of Squares (SSR): SST - SSE
    SSR = SST - SSE  # scalar

    # R²: SSR/SST = 1 - SSE/SST
    R_squared = 1 - (SSE / SST)

    # Estimated error variance: sigmâ² = SSE / (n - p)
    sigma_squared = SSE / df_resid

    # Conventional or heteroskedasticity-robust variance-covariance matrix
    var_beta_hat = var_cov_matrix(X, u_hat, robust)
    se = np.sqrt(np.diag(var_beta_hat)).reshape(-1, 1)  # (p x 1)

    # t-statistics, two-sided p-values, and 95% confidence intervals
    t_stats = beta_hat / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))
    t_crit = stats.t.ppf(0.975, df=df_resid)
    conf_intervals = np.hstack([
        beta_hat - t_crit * se,
        beta_hat + t_crit * se
    ])

    # Store results in a dictionary, including variable names for both X and y
    results = {
        'beta_hat': beta_hat,               # (p x 1)
        'y_hat': y_hat,                     # (n x 1)
        'u_hat': u_hat,                     # (n x 1)
        'var_beta_hat': var_beta_hat,       # (p x p)
        'sigma_squared': sigma_squared,     # scalar
        'se': se,                           # (p x 1)
        't_stats': t_stats,                 # (p x 1)
        'p_values': p_values,               # (p x 1)
        'conf_intervals': conf_intervals,   # (p x 2)
        'SST': SST,                         # scalar
        'SSR': SSR,                         # scalar
        'SSE': SSE,                         # scalar
        'R_squared': R_squared,             # scalar
        'n': n, 'p': p,                     # observations and parameters
        'df_resid': df_resid,               # residual degrees of freedom
        'robust': robust,                   # robust standard errors requested
        'lbl_X': lbl_X, # Names of independent variables
        'lbl_y': lbl_y   # Name of dependent variable
    }

    return results

def var_cov_matrix(X, u_hat, robust=False):
    """Variance-covariance matrix for the OLS estimator."""
    n, p = X.shape
    XX_inv = np.linalg.inv(X.T @ X)

    if robust:
        # HC1: n/(n-p) times the heteroskedasticity-robust sandwich estimator
        meat = (X * (u_hat**2)).T @ X
        return (n / (n - p)) * XX_inv @ meat @ XX_inv

    sigma_squared = np.squeeze(u_hat.T @ u_hat) / (n - p)
    return sigma_squared * XX_inv

def output(results):
    """
    Prints OLS summary in a readable format.

    Parameters:
    results (dict): Model results from the OLS function.
    """
    print(f"OLS Regression Results for Dependent Variable: {results['lbl_y']}")
    print("="*98)
    print(f"Number of Observations: {results['n']}")
    print(f"Residual Degrees of Freedom: {results['df_resid']}")
    print(f"R-squared: {results['R_squared']:.4f}")
    print(f"SST: {results['SST']:.4f}, SSR: {results['SSR']:.4f}, SSE: {results['SSE']:.4f}")
    print("="*98)
    print(f"{'Variable':<20}{'Coefficient':>15}{'Std. Error':>15}{'t':>12}{'P>|t|':>12}{'95% Conf. Interval':>22}")
    print("-"*98)

    for i, var in enumerate(results['lbl_X']):
        beta = results['beta_hat'][i][0]
        std_err = results['se'][i][0]
        t_stat = results['t_stats'][i][0]
        p_value = results['p_values'][i][0]
        conf_low, conf_high = results['conf_intervals'][i]
        print(f"{var:<20}{beta:>15.4f}{std_err:>15.4f}{t_stat:>12.4f}{p_value:>12.4f}   [{conf_low:>8.4f}, {conf_high:<8.4f}]")

    print("="*98)


def summary(models, options=None):
    """
    Print a summary of multiple OLS models in a tabular format, similar to research papers.

    Parameters:
    models (list): A list of OLS result dictionaries (e.g., [mlr1, mlr2, mlr3])
    options (list, optional): Fields to include in the output (default is all fields).
                              Example: ['beta_hat', 'se', 'R_squared']
    """
    # Default fields to include if options is None
    default_fields = ['beta_hat', 'se', 'R_squared', 'SST', 'SSR', 'SSE', 'n']
    fields = options if options else default_fields

    # Collect all unique regressors across models and count their occurrences
    regressor_counts = Counter()
    for model in models:
        regressor_counts.update(model['lbl_X'])

    # Sort regressors: common variables first, rare variables last
    all_regressors = sorted(regressor_counts, key=lambda x: -regressor_counts[x])

    # Initialize table (list of lists)
    table = []

    # First row: Model names
    model_names = [f"Model {i+1}" for i in range(len(models))]
    table.append([""] + model_names)

    # Second row: Dependent variable
    dep_vars = [model['lbl_y'] for model in models]
    table.append(["Dependent variable"] + dep_vars)

    # Rows for the coefficients and standard errors, sorted by regressor frequency
    for regressor in all_regressors:
        row_coef = [regressor]  # Start with the regressor name
        row_se = [""]
        for model in models:
            if regressor in model['lbl_X']:
                idx = model['lbl_X'].index(regressor)
                coef = model['beta_hat'][idx][0]  # Coefficient
                se = model['se'][idx][0]  # Standard error
                row_coef.append(f"{coef:.4f}")  # Coefficient row
                row_se.append(f"({se:.4f})")  # Standard error row
            else:
                row_coef.append("")  # Empty if the regressor is not in the model
                row_se.append("")  # Empty if no standard error for that model
        table.append(row_coef)
        table.append(row_se)

    # Rows for scalar metrics like R_squared, SST, etc.
    scalar_metrics = ['R_squared', 'SST', 'SSR', 'SSE', 'n']
    for metric in scalar_metrics:
        if metric in fields:
            row_metric = [metric]
            for model in models:
                if not metric=='n':
                    row_metric.append(f"{model[metric]:.4f}")
                else:
                     row_metric.append(f"{model[metric]:d}")
            table.append(row_metric)

    # Convert to a pandas DataFrame for pretty display
    df = pd.DataFrame(table)

    # Print the table in a neat format
    with pd.option_context('display.colheader_justify', 'center'):
        print(df.to_string(index=False, header=False))

def Ftest(m_ur, m_r, quiet=True, title="F-test for Joint Significance"):
    """Classical F-test based on unrestricted and restricted SSE."""
    SSE_ur = m_ur['SSE']
    SSE_r = m_r['SSE']
    q = m_ur['p'] - m_r['p']
    df_resid = m_ur['df_resid']

    F_stat = ((SSE_r - SSE_ur) / q) / (SSE_ur / df_resid)
    p_value = 1 - stats.f.cdf(F_stat, q, df_resid)

    if not quiet:
        print(title)
        print(f"  F-statistic: {F_stat:.4f} ~ F({q:d}, {df_resid:d})")
        print(f"  P-value: {p_value:.4f}")
    return F_stat, p_value

def predict(m, X):
    """Linear predictions with standard errors and prediction errors."""
    X = X[m['lbl_X']]
    X_array = X.values

    y_hat = X_array @ m['beta_hat']
    var_y_hat = np.sum((X_array @ m['var_beta_hat']) * X_array, axis=1)
    se_y_hat = np.sqrt(var_y_hat)
    se_prediction = np.sqrt(var_y_hat + m['sigma_squared'])

    return pd.DataFrame({
        'y_hat': y_hat.ravel(),
        'se_y_hat': se_y_hat,
        'se_prediction': se_prediction
    }, index=X.index)
