import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

def ols(X, y):
    """
    Perform OLS estimation using matrix algebra.
    
    Parameters:
    X (pd.DataFrame): Design matrix (independent variables), dimensions (n x k)
    y (pd.Series): Dependent variable, dimensions (n x 1)

    Returns:
    dict: Model results including coefficients, standard errors, residuals, TSS, RSS, ESS, R²,
          and variable names for both independent and dependent variables.
    """
    # Extract variable names from X and y
    lbl_X = X.columns.tolist()  # Names of independent variables
    lbl_y = y.name               # Name of dependent variable
    # Ensure correct dimensions: X (n x k), y (n x 1)
    X = X.values  # (n x k)
    y = y.values.reshape(-1, 1)  # (n x 1)

    # OLS estimates: β = (X'X)^(-1) X'y
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y  # (k x 1)

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

    # Standard errors of coefficients: sqrt(diag(σ² * (X'X)^(-1)))
    var_beta_hat = sigma_squared * np.linalg.inv(X.T @ X)  # (k x k)
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

def output(results):
    """
    Prints OLS summary in a readable format with proper alignment of t-statistics and confidence intervals.
    
    Parameters:
    results (dict): Model results from the OLS function.
    """
    print(f"OLS Regression Results for Dependent Variable: {results['lbl_y']}")
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



def summary(models, options=None):
    """
    Print a summary of multiple OLS models in a tabular format, similar to research papers.

    Parameters:
    models (list): A list of OLS result dictionaries (e.g., [mlr1, mlr2, mlr3])
    options (list, optional): Fields to include in the output (default is all fields).
                              Example: ['beta_hat', 'standard_errors', 'R_squared']
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
                se = model['standard_errors'][idx][0]  # Standard error
                row_coef.append(f"{coef:.4f}")  # Coefficient row
                row_se.append(f"({se:.4f})")  # Standard error row
            else:
                row_coef.append("")  # Empty if the regressor is not in the model
                row_se.append("")  # Empty if no standard error for that model
        table.append(row_coef)
        table.append(row_se)

    # Rows for scalar metrics like R_squared, TSS, etc.
    scalar_metrics = ['R_squared', 'TSS', 'RSS', 'ESS', 'n']
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