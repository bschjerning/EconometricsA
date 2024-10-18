import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
# Short-hand reference for inverse matrix
inv = np.linalg.inv

def labels(vars):
    labels = [var.to_frame().columns.tolist() if isinstance(var, pd.Series) else var.columns.tolist() for var in vars]
    return labels

def pd_to_arrays(vars):
    arrays = [var.to_frame().values if isinstance(var, pd.Series) else var.values for var in vars]
    return  arrays

def ols(y,X, robust=False, quiet=True, title="OLS Regression Results"):
    lbl_y, lbl_X = labels([y,X])
    y,X= pd_to_arrays([y,X])
    
    # OLS estimator, predicted values, residuals
    beta_hat = inv(X.T @ X) @ X.T @ y   # OLS estimator: β = (X'X)^(-1)X'y
    u_hat = y - X @ beta_hat            # residuals: û = y - Xβ
    var_beta_hat = var_cov_matrix(X, u_hat, robust)     # Variance-covariance matrix of β_hat
    
    # Standard errors, t-stats, p-values, confidence intervals, TSS, RSS, ESS, R²
    results = predict(y, X, beta_hat, var_beta_hat, u_hat, lbl_X, lbl_y, robust)
    if not quiet: # Print OLS results
        output(results, title=title)  
    return results

def var_cov_matrix(X, u_hat, robust=False):
    n, k = X.shape     # scalars, # of observations (n) and parameters (k)
    if robust: # HCSE: heteroskedasticity-consistent varians-kovarians matrix
        Ainv = inv(X.T @ X / n)    # Beregn Ainv
        Sigma = (X* (u_hat**2)).T  @ X / n  # Matrix for beregning af Σ  
        var_beta_hat = Ainv @ Sigma @ Ainv / (n - k)  # Returner HCSE varians-kovarians matrix
    else: # OLS varians-kovarians matrix under homoskedasticitet
        sigma2 = np.sum(u_hat**2) / (n - k)  # Estimeret fejlledsvarians
        var_beta_hat = sigma2 * inv(X.T @ X)  # Returner standard OLS varians-kovarians matrix
    return var_beta_hat

def predict(y, X, beta_hat, var_beta_hat, u_hat, lbl_X, lbl_y, robust):
    n, k = X.shape                      # of obs (n) and parameters (k)

    # Standard errors, t-tests and confidence intervals    
    y_hat = X @ beta_hat                # predicted values: ŷ = Xβ          
    se = np.sqrt(np.diag(var_beta_hat)).reshape(-1, 1)  # Standard errors of β_hat
    t_stats = beta_hat / se                             # t-stats: β / se(β)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))     # Two-tailed p-values for t-stats
    t_crit = stats.t.ppf(1 - 0.025, df=n - k)                       # Critical t-value 
    conf_intervals = np.hstack([beta_hat - t_crit * se, beta_hat + t_crit * se]) # Confidence intervals

    # Goodness-of-fit measures: R², TSS, RSS, ESS, σ²
    RSS = float(u_hat.T @ u_hat)                # Residual Sum of Squares (RSS): u'u
    TSS = float(((y - y.mean()) ** 2).sum())    # Total Sum of Squares (TSS): (y - ȳ)'(y - ȳ)
    ESS = TSS - RSS                             # Explained Sum of Squares (ESS): TSS - RSS
    R_squared = 1 - (RSS / TSS)                 # R²: 1 - (RSS/TSS)
    sigma_squared = RSS / (n - k)               # Variance of u_hat: σ² = RSS / (n - k)

    # Store results in a dictionary, including variable names for both X and y
    results = {'beta_hat': beta_hat, 'se': se, 'sigma_squared': sigma_squared, 'var_beta_hat': var_beta_hat, 
               't_stats': t_stats,  'p_values': p_values,'conf_intervals': conf_intervals,  
               'TSS': TSS,    'RSS': RSS,    'ESS': ESS,    'R_squared': R_squared,
               'n': n, 'k': k,'lbl_X': lbl_X,'lbl_y': lbl_y,
               'y_hat': y_hat, 'u_hat': u_hat
               }
    return results

def first_stage(y, X1, X2, Ze, robust=False, quiet=True):
    y, X1, X2, Ze = [pd.DataFrame(var) for var in [y, X1, X2, Ze]]
    Z = pd.concat([X1, Ze], axis=1)
    results = []
    for i, var in enumerate(X2.columns):
        # OLS regression of each endogenous variable on all exogenous variables and instruments
        res = ols(y=X2.loc[:, var], X=Z, robust=robust, quiet=quiet, title=f"First Stage for {var}")
        results.append(res)

        # F-test for joint significance of instruments    
        res_r = ols(y=X2.loc[:, var], X=Ze, robust=robust, quiet=True)
        title_F = f"F-test for joint significance of instruments: {var}"
        F_stat, p_value= Ftest(res, res_r, quiet=quiet, title=title_F)
    return results

def tsls(y, X1, X2, Ze, robust=False, quiet=True, title="2SLS Regression Results"):
    y, X1, X2, Ze = [pd.DataFrame(var) for var in [y, X1, X2, Ze]]
    X = pd.concat([X1, X2], axis=1)     # Combine X1 and X2 (n x (k1 + k2))
    Z = pd.concat([X1, Ze], axis=1)     # Combine X1 and X2 (n x (k1 + k2))
    lbl_y, lbl_X1, lbl_X2, lbl_Ze, lbl_X, lbl_Z = labels([y, X1, X2, Ze, X, Z]) # Extract variable names
    y, X1, X2, Ze, X, Z = pd_to_arrays([y, X1, X2, Ze, X, Z]) # Convert to arrays

    # Predicted values for endogenous variables from the first stage (X2_hat)
    Pz = Z @ inv(Z.T @ Z) @ Z.T  # (n x n) projection matrix (to get X predicted by Z)
    X_hat = Pz @ X  # (n x k2)
    
    # Estimate beta_hat using "2SLS" formula: β = (X'PzX)^(-1) * X'Pz*y
    beta_2SLS = inv(X.T@Pz@X) @ X.T@Pz@y  # (k1 + k2 x 1)
    u_hat = y - X @ beta_2SLS            # residuals: û = y - Xβ (use X instead of X_hat)
    var_beta_hat = var_cov_matrix(X_hat, u_hat, robust)     # Variance-covariance matrix of β_hat

    # Standard errors, t-stats, p-values, confidence intervals, TSS, RSS, ESS, R²
    results = predict(y, X, beta_2SLS, var_beta_hat, u_hat, lbl_X, lbl_y, robust)
    
    # Saran test for overidentification
    s2=float(results['RSS']/results['n'])  # Residual variance estimate
    W = inv(s2*Z.T @ Z)  # Weighting matrix for J-statistic
    J_stat = float(u_hat.T @ Z @ W @ Z.T@u_hat)  # J-statistic for overidentification test
    results['J_stat'] = J_stat
    pval_Jstat = 1 - stats.chi2.cdf(J_stat, len(lbl_Ze)-len(lbl_X2))  # P-value for J-statistic

    results['pval_Jstat'] = pval_Jstat
    if not quiet: # Print 2SLS results
        output(results, title="2SLS Regression Results")
        print(f'Sargant test for overidentification, J={J_stat:.10f} ~ Chi2({len(lbl_Ze)-len(lbl_X2)})') 
        print(f'P-value for Sargan test: {pval_Jstat:.4f}\n')
    return results

def output(results, title="OLS Regression Results"):
    """
    Prints OLS/2SLS summary in a readable format with proper alignment of t-statistics and confidence intervals.
    
    Parameters:
    results (dict): Model results from the OLS or 2SLS function.
    title (str, optional): Title for the output. Defaults to 'OLS Regression Results'.
    """
    print(f"{title}, Dependent Variable: {results['lbl_y']}")
    print("="*98)
    print(f"Number of Observations: {results['n']}")
    print(f"Degrees of Freedom: {results['n'] - results['k']} (Residual), {results['k']} (Model)")
    print(f"R-squared: {results['R_squared']:.4f}")
    print(f"TSS: {results['TSS']:.4f}, RSS: {results['RSS']:.4f}, ESS: {results['ESS']:.4f}")
    print("="*98)
    
    # Adjusted header for the table to align the confidence interval label correctly
    print(f"{'Variable':<20}{'Coefficient':>15}{'Std. Error':>15}{'t':>12}{'P>|t|':>12}{'95% Conf. Interval':>22}")
    print("-"*98)
    
    for i, var in enumerate(results['lbl_X']):
        beta = results['beta_hat'][i][0]
        std_err = results['se'][i][0]
        t_stat = results['t_stats'][i][0]
        p_val = results['p_values'][i][0]
        conf_low = results['conf_intervals'][i][0]
        conf_high = results['conf_intervals'][i][1]
        
        # Print values with proper alignment for confidence interval
        print(f"{var:<20}{beta:>15.4f}{std_err:>15.4f}{t_stat:>12.4f}{p_val:>12.4f}   [{conf_low:>8.4f}, {conf_high:<8.4f}]")
    
    print("="*98)



def summary(models, options=None, column_titles=None, report_stats="se"):
    """
    Print a summary of multiple OLS/2SLS models in a tabular format, similar to research papers.

    Parameters:
    models (list): A list of OLS/2SLS result dictionaries (e.g., [mlr1, mlr2, mlr3]).
    options (list, optional): Fields to include in the output (default is all fields).
                              Example: ['beta_hat', 'se', 'R_squared'].
    column_titles (list, optional): Custom column titles for each model. Defaults to 'Model 1', 'Model 2', etc.
    report_stats (str, optional): Options are "se" (default), "t", or None.
                                  "se" reports standard errors in parentheses.
                                  "t" reports t-statistics in parentheses.
                                  None leaves parentheses empty.
    """
    # Default fields to include if options is None
    default_fields = ['beta_hat', 'se', 'R_squared', 'TSS', 'RSS', 'ESS', 'n']
    fields = options if options else default_fields

    # Collect all unique regressors across models and count their occurrences
    regressor_counts = Counter()
    for model in models:
        regressor_counts.update(model['lbl_X'])
    
    # Sort regressors: common variables first, rare variables last
    all_regressors = sorted(regressor_counts, key=lambda x: -regressor_counts[x])

    # Initialize table (list of lists)
    table = []

    # First row: Model names or custom titles
    if column_titles is None:
        column_titles = [f"Model {i+1}" for i in range(len(models))]
    table.append([""] + column_titles)

    # Second row: Dependent variable
    dep_vars = [model['lbl_y'] for model in models]
    table.append(["Dependent variable"] + dep_vars)

    # Rows for the coefficients and standard errors or t-statistics
    for regressor in all_regressors:
        row_coef = [regressor]  # Start with the regressor name
        row_stats = [""]  # Start with an empty space for SE/t-values or nothing
        for model in models:
            if regressor in model['lbl_X']:
                idx = model['lbl_X'].index(regressor)
                coef = model['beta_hat'][idx][0]  # Coefficient
                if report_stats == "se":
                    stat = model['se'][idx][0]  # Standard error
                    row_stats.append(f"({stat:.4f})")
                elif report_stats == "t":
                    stat = model['t_stats'][idx][0]  # t-statistic
                    row_stats.append(f"({stat:.4f})")
                else:
                    row_stats.append("")  # No parentheses content if None
                row_coef.append(f"{coef:.4f}")  # Coefficient row
            else:
                row_coef.append("")  # Empty if the regressor is not in the model
                row_stats.append("")  # Empty for SE/t-value if not in model
        table.append(row_coef)
        table.append(row_stats)

    # Rows for scalar metrics like R_squared, TSS, etc.
    scalar_metrics = ['R_squared', 'TSS', 'RSS', 'ESS', 'n']
    for metric in scalar_metrics:
        if metric in fields:
            row_metric = [metric]
            for model in models:
                if metric != 'n':
                    row_metric.append(f"{model[metric]:.4f}")
                else:
                    row_metric.append(f"{model[metric]:d}")
            table.append(row_metric)

    # Convert to a pandas DataFrame for pretty display
    df = pd.DataFrame(table)
    
    # Print the table in a neat format
    with pd.option_context('display.colheader_justify', 'center'):
        print(df.to_string(index=False, header=False))

    # Add a note about what is printed in parentheses
    if report_stats == "se":
        print("Note: Standard errors are reported in parentheses.\n")
    elif report_stats == "t":
        print("Note: t-statistics are reported in parentheses.\n")


def Ftest(m_ur, m_r, quiet=True, title="F-test for Joint Significance"):
    # m_ur, m_r: estimation results for unrestricted and restricted models
    RSS_ur, k_ur, n  = m_ur['RSS'], m_ur['k'], m_ur['n']    
    RSS_r, k_r = m_r['RSS'], m_r['k']
    q = k_ur - k_r # Number of restrictions (q) 

    F_stat = ((RSS_r - RSS_ur) / q) / (RSS_ur / (n - k_ur))
    p_value = 1 - stats.f.cdf(F_stat, q, n - k_ur)  # P-value for F-statistic
    if not quiet:
        print(f"{title}")
        print(f"  F-statistic: {F_stat:.4f} ~ F({q:d},{n - k_ur:d})")
        print(f"  P-value: {p_value:.4f}")
    return F_stat, p_value