import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats

def ols(y, X):
    """
    Perform OLS estimation using matrix algebra.

    Parameters:
    y (pd.Series): Dependent variable, dimensions (n x 1)
    X (pd.DataFrame): Design matrix, dimensions (n x p)

    Returns:
    dict: Model results including coefficients, standard errors, residuals, SST, SSE, SSR, R²,
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

    # Sum of Squared Residuals (SSR): u'u
    # The matrix product is 1 x 1; squeeze removes the redundant dimensions
    SSR = np.squeeze(u_hat.T @ u_hat)  # scalar

    # Total Sum of Squares (SST): (y - ȳ)'(y - ȳ)
    y_centered = y - y.mean()
    SST = np.squeeze(y_centered.T @ y_centered)  # scalar

    # Explained Sum of Squares (SSE): SST - SSR
    SSE = SST - SSR  # scalar

    # R²: SSE/SST = 1 - SSR/SST
    R_squared = 1 - (SSR / SST)

    # Estimated error variance: sigmâ² = SSR / (n - p)
    sigma_squared = SSR / df_resid

    # Standard errors of coefficients: sqrt(diag(σ² * (X'X)^(-1)))
    var_beta_hat = sigma_squared * np.linalg.inv(X.T @ X)  # (p x p)
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
        'SSE': SSE,                         # scalar
        'SSR': SSR,                         # scalar
        'R_squared': R_squared,             # scalar
        'n': n, 'p': p,                     # observations and parameters
        'df_resid': df_resid,               # residual degrees of freedom
        'lbl_X': lbl_X, # Names of independent variables
        'lbl_y': lbl_y   # Name of dependent variable
    }

    return results

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
    print(f"SST: {results['SST']:.4f}, SSE: {results['SSE']:.4f}, SSR: {results['SSR']:.4f}")
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
    default_fields = ['beta_hat', 'se', 'R_squared', 'SST', 'SSE', 'SSR', 'n']
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
    scalar_metrics = ['R_squared', 'SST', 'SSE', 'SSR', 'n']
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

def Ftest(m_ur, m_r, alpha=0.05):
    """F-test based on unrestricted and restricted OLS results."""
    SSR_ur = m_ur['SSR']
    SSR_r = m_r['SSR']
    q = m_ur['p'] - m_r['p']
    df_resid = m_ur['df_resid']

    F_stat = ((SSR_r - SSR_ur) / q) / (SSR_ur / df_resid)
    p_value = stats.f.sf(F_stat, q, df_resid)
    critical_value = stats.f.ppf(1 - alpha, q, df_resid)

    print(f"F-test: F({q:d}, {df_resid:d}) = {F_stat:.4f}")
    print(f"{100 * alpha:g}% critical value = {critical_value:.4f}, p-value = {p_value:.4f}")
    return F_stat, p_value


def Waldtest(m_ur, R, r, alpha=0.05):
    """Wald test of the linear hypothesis H0: R beta = r."""
    q = R.shape[0]

    d = R @ m_ur['beta_hat'] - r
    V_d = R @ m_ur['var_beta_hat'] @ R.T
    Wald = np.squeeze(d.T @ np.linalg.solve(V_d, d))
    p_value = stats.chi2.sf(Wald, q)
    critical_value = stats.chi2.ppf(1 - alpha, q)

    print(f"Wald test: chi2({q:d}) = {Wald:.4f}")
    print(f"{100 * alpha:g}% critical value = {critical_value:.4f}, p-value = {p_value:.4f}")
    return Wald, p_value


def bin_means(x, y, bins=20):
    """Group means using quantile bins, or each x value when bins=None."""
    data = pd.DataFrame({
        'x': np.asarray(x).reshape(-1),
        'y': np.asarray(y).reshape(-1)
    }).dropna()
    if bins is None:
        return data.groupby('x', as_index=False)['y'].mean()
    data['bin'] = pd.qcut(data['x'], bins, duplicates='drop')
    return data.groupby('bin', observed=True)[['x', 'y']].mean()


def binscatter(x, y, bins=20, ax=None, xlabel=None, ylabel=None,
               title=None, raw=True, regression=True):
    """Plot binned means, raw observations, and a simple OLS regression line."""
    x_name = getattr(x, 'name', None) or 'x'
    y_name = getattr(y, 'name', None) or 'y'
    data = pd.DataFrame({
        'x': np.asarray(x).reshape(-1),
        'y': np.asarray(y).reshape(-1)
    }).dropna()

    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(5, 4))

    means = bin_means(data['x'], data['y'], bins)
    if raw:
        ax.scatter(data['x'], data['y'], s=5, alpha=0.08, color='grey')
    mean_label = 'Means by x value' if bins is None else 'Binned means'
    ax.scatter(means['x'], means['y'], color='tab:blue', label=mean_label)

    if regression:
        X = pd.DataFrame({'const': 1.0, x_name: data['x']})
        y_series = pd.Series(data['y'].to_numpy(), name=y_name)
        model = ols(y_series, X)
        b0, b1 = model['beta_hat'].ravel()
        line = np.linspace(data['x'].min(), data['x'].max(), 100)
        label = rf'OLS: slope = {b1:.3f}, $R^2$ = {model["R_squared"]:.3f}'
        ax.plot(line, b0 + b1 * line, color='tab:red', label=label)

    x_label = xlabel or x_name
    y_label = ylabel or y_name
    ax.set(xlabel=x_label, ylabel=y_label,
           title=title or f'{y_label} vs. {x_label}')
    ax.legend()

    if own_figure:
        plt.tight_layout()
        plt.show()


def residual_variance_plot(m, z, bins=10, axes=None, xlabel=None,
                           title=None, titles=None):
    """Plot OLS residuals and binned mean squared residuals against z."""
    u_hat = m['u_hat'].ravel()
    z_name = getattr(z, 'name', None) or 'z'
    residual_means = bin_means(z, u_hat, bins)
    means = bin_means(z, u_hat ** 2, bins)

    own_figure = axes is None
    if own_figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = axes[0].figure

    default_title = ('Mean squared residuals by x value' if bins is None
                     else 'Mean squared residuals by bins')
    titles = titles or ('Residuals', default_title)

    axes[0].scatter(z, u_hat, s=5, alpha=0.08)
    axes[0].scatter(residual_means['x'], residual_means['y'],
                    color='tab:red', label=('Means by x value' if bins is None
                                            else 'Binned means'))
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set(xlabel=xlabel or z_name, ylabel='Residual', title=titles[0])
    axes[0].legend()

    axes[1].scatter(means['x'], means['y'], color='tab:blue')
    axes[1].plot(means['x'], means['y'], color='tab:blue')
    axes[1].set(xlabel=xlabel or z_name,
                ylabel=r'Mean of $\hat u^2$',
                title=titles[1])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if own_figure:
        plt.show()

    variance_ratio = means['y'].max() / means['y'].min()
    groups = 'x values' if bins is None else 'bins'
    print(f'Max/min mean(u_hat^2) by {groups} = {variance_ratio:.2f}')
