import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

def ols(y, X, robust=False, quiet=True, title="OLS Regression Results"):
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

    if not quiet:
        output(results, title=title)
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

def output(results, title="OLS Regression Results"):
    """
    Prints OLS summary in a readable format.

    Parameters:
    results (dict): Model results from the OLS function.
    """
    print(f"{title} for Dependent Variable: {results['lbl_y']}")
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


def summary(models, options=None, column_titles=None, report_stats="se"):
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

    # First row: Model names or user-supplied column titles
    if column_titles is None:
        column_titles = [f"Model {i+1}" for i in range(len(models))]
    table.append([""] + column_titles)

    # Second row: Dependent variable
    dep_vars = [model['lbl_y'] for model in models]
    table.append(["Dependent variable"] + dep_vars)

    # Rows for the coefficients and standard errors, sorted by regressor frequency
    for regressor in all_regressors:
        row_coef = [regressor]  # Start with the regressor name
        row_stats = [""]
        for model in models:
            if regressor in model['lbl_X']:
                idx = model['lbl_X'].index(regressor)
                coef = model['beta_hat'][idx][0]  # Coefficient
                row_coef.append(f"{coef:.4f}")  # Coefficient row
                if report_stats == "se":
                    stat = model['se'][idx][0]
                    row_stats.append(f"({stat:.4f})")
                elif report_stats == "t":
                    stat = model['t_stats'][idx][0]
                    row_stats.append(f"({stat:.4f})")
                else:
                    row_stats.append("")
            else:
                row_coef.append("")  # Empty if the regressor is not in the model
                row_stats.append("")
        table.append(row_coef)
        table.append(row_stats)

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

    if report_stats == "se":
        print("Note: Standard errors are reported in parentheses.\n")
    elif report_stats == "t":
        print("Note: t-statistics are reported in parentheses.\n")

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

def first_stage(y, X1, X2, Ze, robust=False, quiet=True):
    """Regress each endogenous variable in X2 on the instruments Z = [X1, Ze]."""
    y = pd.DataFrame(y)
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)
    Ze = pd.DataFrame(Ze)

    if not (len(y) == len(X1) == len(X2) == len(Ze)):
        raise ValueError("y, X1, X2, and Ze must have the same number of observations")

    # Z contains included exogenous regressors and excluded instruments
    Z = pd.concat([X1, Ze], axis=1)

    results = []
    for var in X2.columns:
        m_ur = ols(y=X2[var], X=Z, robust=robust)
        m_r = ols(y=X2[var], X=X1, robust=robust)
        F_stat, p_value = Ftest(
            m_ur,
            m_r,
            quiet=quiet,
            title=f"First-stage F-test for excluded instruments: {var}"
        )
        m_ur['F_stat'] = F_stat
        m_ur['F_p_value'] = p_value
        results.append(m_ur)

        if not quiet:
            output(m_ur, title=f"First Stage for {var}")

    return results

def tsls(y, X1, X2, Ze, robust=False, quiet=True, title="2SLS Regression Results"):
    """Estimate a linear model by two-stage least squares."""
    y = pd.DataFrame(y)
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)
    Ze = pd.DataFrame(Ze)

    # X = [X1, X2] is the regressor matrix
    # Z = [X1, Ze] is the instrument matrix
    X = pd.concat([X1, X2], axis=1)
    Z = pd.concat([X1, Ze], axis=1)
    lbl_y = y.columns[0]
    lbl_X = X.columns.tolist()

    y = y.values
    X = X.values
    Z = Z.values

    # Projection onto the column space of Z
    Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    X_hat = Pz @ X

    # 2SLS: beta_hat = (X'PzX)^(-1)X'Pz y
    beta_hat = np.linalg.inv(X.T @ Pz @ X) @ X.T @ Pz @ y
    y_hat = X @ beta_hat
    u_hat = y - y_hat

    n, p = X.shape
    df_resid = n - p
    var_beta_hat = var_cov_matrix(X_hat, u_hat, robust)
    se = np.sqrt(np.diag(var_beta_hat)).reshape(-1, 1)
    t_stats = beta_hat / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))
    t_crit = stats.t.ppf(0.975, df=df_resid)
    conf_intervals = np.hstack([
        beta_hat - t_crit * se,
        beta_hat + t_crit * se
    ])

    SSE = np.squeeze(u_hat.T @ u_hat)
    y_centered = y - y.mean()
    SST = np.squeeze(y_centered.T @ y_centered)
    SSR = SST - SSE
    R_squared = 1 - SSE / SST
    sigma_squared = SSE / df_resid

    results = {
        'beta_hat': beta_hat,
        'y_hat': y_hat,
        'u_hat': u_hat,
        'var_beta_hat': var_beta_hat,
        'sigma_squared': sigma_squared,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'conf_intervals': conf_intervals,
        'SST': SST,
        'SSR': SSR,
        'SSE': SSE,
        'R_squared': R_squared,
        'n': n,
        'p': p,
        'df_resid': df_resid,
        'robust': robust,
        'lbl_X': lbl_X,
        'lbl_y': lbl_y,
    }

    # Sargan test for overidentifying restrictions (homoskedastic version)
    df_sargan = Ze.shape[1] - X2.shape[1]
    if df_sargan > 0:
        s2 = SSE / n
        W = np.linalg.inv(s2 * (Z.T @ Z))
        J_stat = np.squeeze(u_hat.T @ Z @ W @ Z.T @ u_hat)
        pval_Jstat = 1 - stats.chi2.cdf(J_stat, df_sargan)
    else:
        J_stat = np.nan
        pval_Jstat = np.nan

    results['J_stat'] = J_stat
    results['pval_Jstat'] = pval_Jstat
    results['df_sargan'] = df_sargan

    if not quiet:
        output(results, title=title)
        if df_sargan > 0:
            print(f"Sargan test: J = {J_stat:.4f} ~ Chi2({df_sargan:d})")
            print(f"P-value: {pval_Jstat:.4f}\n")

    return results
