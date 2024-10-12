import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

def ols(X, y, robust=False):
    """
    Perform OLS estimation using matrix algebra.
    
    Parameters:
    X (pd.DataFrame): Design matrix (independent variables), dimensions (n x k)
    y (pd.Series): Dependent variable, dimensions (n x 1)

    Returns:
    dict: Model results including coefficients, standard errors, residuals, TSS, RSS, ESS, R²,
          and variable names for both independent and dependent variables.
    """

    # Short-hand reference for inverse matrix
    inv = np.linalg.inv

    # Extract variable names from X and y
    lbl_X = X.columns.tolist()  # Names of independent variables
    lbl_y = y.name               # Name of dependent variable
    # Ensure correct dimensions: X (n x k), y (n x 1)
    X = X.values  # (n x k)
    y = y.values.reshape(-1, 1)  # (n x 1)

    # OLS estimates: β = (X'X)^(-1) X'y
    
    beta_hat = inv(X.T @ X) @ X.T @ y  # (k x 1)

    # Predicted values: ŷ = Xβ
    y_hat = X @ beta_hat  # (n x 1)

    # Residuals: u = y - ŷ
    residuals = y - y_hat  # (n x 1)

    # Number of observations (n) and parameters (k)
    n, k = X.shape

    # Residual Sum of Squares (RSS): u'u
    RSS = float(residuals.T @ residuals)  # scalar

    # Total Sum of Squares (TSS): (y - ȳ)'(y - ȳ)
    TSS = float(((y - y.mean()) ** 2).sum())  # scalar

    # Explained Sum of Squares (ESS): TSS - RSS
    ESS = TSS - RSS  # scalar

    # R²: 1 - (RSS/TSS)
    R_squared = 1 - (RSS / TSS)

    # Variance of residuals: σ² = RSS / (n - k)
    sigma_squared = RSS / (n - k)

    if robust:
        # HCSE: heteroskedasticity-consistent varians-kovarians matrix
        Ainv = inv(X.T @ X / n)    # Beregn Ainv
        Sigma = (X* (residuals**2)).T  @ X / n  # Matrix for beregning af Σ  
        var_beta_hat = Ainv @ Sigma @ Ainv / (n - k)  # Returner HCSE varians-kovarians matrix
    else:
        # OLS varians-kovarians matrix under homoskedasticitet
        sigma2 = np.sum(residuals**2) / (n - k)  # Estimeret fejlledsvarians
        var_beta_hat = sigma2 * inv(X.T @ X)  # Returner standard OLS varians-kovarians matrix

    # Standard errors of coefficients: sqrt(diag(σ² * (X'X)^(-1)))
    standard_errors = np.sqrt(np.diag(var_beta_hat)).reshape(-1, 1)  # (k x 1)

    # t-statistics: β / se(β)
    t_stats = beta_hat / standard_errors

    # Two-tailed p-values for t-stats
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    # Critical t-value for 95% confidence intervals
    t_critical = stats.t.ppf(1 - 0.025, df=n - k)

    # Confidence intervals
    conf_intervals = np.hstack([
        beta_hat - t_critical * standard_errors,
        beta_hat + t_critical * standard_errors
    ])

    # Store results in a dictionary, including variable names for both X and y
    results = {
        'beta_hat': beta_hat,               # (k x 1)
        'y_hat': y_hat,                     # (n x 1)
        'residuals': residuals,             # (n x 1)
        'var_beta_hat': var_beta_hat,       # (k x k)
        'sigma_squared': sigma_squared,     # scalar
        'standard_errors': standard_errors, # (k x 1)
        't_stats': t_stats,                 # (k x 1)
        'p_values': p_values,               # (k x 1)
        'conf_intervals': conf_intervals,   # (k x 2)
        'TSS': TSS,                         # scalar
        'RSS': RSS,                         # scalar
        'ESS': ESS,                         # scalar
        'R_squared': R_squared,             # scalar
        'n': n, 'k': k,                     # number of observations and parameters
        'lbl_X': lbl_X,                     # Names of independent variables
        'lbl_y': lbl_y                      # Name of dependent variable
    }
    
    return results

def tsls(X1, X2, y, Ze, robust=False):
    """
    Perform Two-Stage Least Squares (2SLS) estimation using matrix algebra 
    with options for robust, homoskedastic, and 2SLS-specific standard errors.

    Model:
    --------
    The model is a two-equation system:

    1. First stage (instrumental regression for endogenous variables):
       X2 = Ze * π + X1 * γ + error_1
       
       Where X2 are the endogenous variables and Ze are the excluded instruments.
       
    2. Second stage (regression for the dependent variable):
       y = X1 * β1 + X2_hat * β2 + u
       
       Where X2_hat are the predicted values from the first stage, 
       X1 are the exogenous variables, and y is the dependent variable.

    Parameters:
    ------------
    X1 (pd.DataFrame): Exogenous variables, dimensions (n x k1)
    X2 (pd.DataFrame): Endogenous variables, dimensions (n x k2)
    y (pd.Series): Dependent variable (y), dimensions (n x 1)
    Ze (pd.DataFrame): Excluded instruments for X2, dimensions (n x l)
    Robust: Type of standard errors to compute:
                    - True (default): Robust to heteroskedasticity and 2-stage estimation errors
                    - False: Homoskedastic standard errors  
    Returns:
    ------------
    dict: Model results including:
          - Coefficients (beta_hat)
          - Standard errors (based on the specified variance type)
          - Residuals, predicted values, confidence intervals, and R-squared.
    """
    # Short-hand reference for inverse matrix
    inv = np.linalg.inv

    # Extract variable names from X1, X2, and y
    lbl_X1 = X1.columns.tolist()  # Names of exogenous variables
    lbl_X2 = X2.columns.tolist()  # Names of endogenous variables
    lbl_X = lbl_X1 + lbl_X2  # Combine exogenous and endogenous variable names
    lbl_y = y.name  # Name of dependent variable
    lbl_Ze = Ze.columns.tolist()  # Names of excluded instruments
    
    # Convert to numpy arrays for matrix operations
    X1 = X1.values  # Exogenous variables (n x k1)
    X2 = X2.values  # Endogenous variables (n x k2)
    Ze = Ze.values  # Excluded instruments (n x l)
    y = y.values.reshape(-1, 1)  # Dependent variable (n x 1)

    # First stage: Regress X2 (endogenous variables) on Ze (excluded instruments) and X1 (exogenous variables)
    # Combine excluded instruments (Ze) and exogenous variables (X1) to form the full instrument matrix
    Z = np.hstack([Ze, X1])  # (n x (l + k1))

    # Predicted values for endogenous variables from the first stage (X2_hat)
    X2_hat = Z @ inv(Z.T @ Z) @ Z.T @ X2  # (n x k2)

    # Second stage: Regress y on X1 (exogenous) and X2_hat (predicted endogenous variables)
    X_hat = np.hstack([X1, X2_hat])  # Combine X1 and predicted X2_hat (n x (k1 + k2))
    
    # Estimate beta_hat using OLS: β = (X'X)^(-1) X'y
    beta_hat = inv(X_hat.T @ X_hat) @ X_hat.T @ y  # (k1 + k2 x 1)
    
    # Predicted values for y: ŷ = X * β
    y_hat = X_hat @ beta_hat  # (n x 1)
    
    # Residuals: u = y - ŷ
    residuals = y - y_hat  # (n x 1)

    # Number of observations (n) and parameters (k1 + k2)
    n, k = X_hat.shape

    # Total Sum of Squares (TSS): (y - ȳ)'(y - ȳ)
    TSS = float(((y - y.mean()) ** 2).sum())  # scalar

    # Residual Sum of Squares (RSS): u'u
    RSS = float(residuals.T @ residuals)  # scalar

    # Explained Sum of Squares (ESS): TSS - RSS
    ESS = TSS - RSS  # scalar

    # R²: 1 - (RSS/TSS)
    R_squared = 1 - (RSS / TSS)

    # Variance of residuals: σ² = RSS / (n - k)
    sigma_squared = RSS / (n - k)
    
    # Handle different variance estimators based on var_type argument
    if robust:
        # Robust OLS variance (heteroskedasticity-robust)
        Ainv = inv(X_hat.T @ X_hat / n)    # Inverse of (X'X)
        Sigma = (X_hat* (residuals**2)).T  @ X_hat / n  # Matrix for beregning af Σ  
        var_beta_hat = Ainv @ Sigma @ Ainv / (n - k)    
    else:
        # Homoskedastic variance-covariance matrix
        sigma_squared = float((residuals.T @ residuals) / (n - k))  # Residual variance
        var_beta_hat = sigma_squared * inv(X_hat.T @ X_hat)
    
    # Standard errors of coefficients
    standard_errors = np.sqrt(np.diag(var_beta_hat)).reshape(-1, 1)  # (k1 + k2 x 1)

    # t-statistics: β / se(β)
    t_stats = beta_hat / standard_errors

    # Two-tailed p-values for t-stats
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    # Critical t-value for 95% confidence intervals
    t_critical = stats.t.ppf(1 - 0.025, df=n - k)

    # Confidence intervals
    conf_intervals = np.hstack([
        beta_hat - t_critical * standard_errors,
        beta_hat + t_critical * standard_errors
    ])

    # Store results in a dictionary
    results = {
        'beta_hat': beta_hat,               # (k1 + k2 x 1)
        'y_hat': y_hat,                     # (n x 1)
        'residuals': residuals,             # (n x 1)
        'sigma_squared': sigma_squared,     # scalar
        'var_beta_hat': var_beta_hat,       # (k1 + k2 x k1 + k2)
        'standard_errors': standard_errors, # Standard errors based on var_type (k1 + k2 x 1)
        't_stats': t_stats,                 # (k1 + k2 x 1)
        'p_values': p_values,               # (k1 + k2 x 1)
        'conf_intervals': conf_intervals,   # (k1 + k2 x 2)
        'TSS': TSS,                         # scalar
        'RSS': RSS,                         # scalar
        'ESS': ESS,                         # scalar
        'R_squared': R_squared,             # scalar
        'n': n, 'k': k,                     # number of observations and parameters
        'lbl_X': lbl_X,                     # Combined exogenous and endogenous variable names
        'lbl_Ze': lbl_Ze,                   # Names of excluded instruments
        'lbl_y': lbl_y                  # Name of dependent variable   
   }

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
        std_err = results['standard_errors'][i][0]
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
                              Example: ['beta_hat', 'standard_errors', 'R_squared'].
    column_titles (list, optional): Custom column titles for each model. Defaults to 'Model 1', 'Model 2', etc.
    report_stats (str, optional): Options are "se" (default), "t", or None.
                                  "se" reports standard errors in parentheses.
                                  "t" reports t-statistics in parentheses.
                                  None leaves parentheses empty.
    """
    # Default fields to include if options is None
    default_fields = ['beta_hat', 'standard_errors', 'R_squared', 'TSS', 'RSS', 'ESS', 'n']
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
                    stat = model['standard_errors'][idx][0]  # Standard error
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