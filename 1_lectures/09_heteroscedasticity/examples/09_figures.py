# %% 
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(0)

# Generate the x values
x = 2 + 4 * np.random.rand(400)

# Generate the y values
y = 2 + 2 * x

# Generate the u values for different cases (A, B, C, D)
uA = np.random.normal(0, 0.25 + 0.75 * (x - 2), size=len(x))
uB = np.random.normal(0, 1, size=len(x))
uC = np.random.normal(0, 2.5, size=len(x))
uD = np.random.normal(0, 0.25 + 0.75 * (6 - x), size=len(x))

# Generate y values for different cases
yA = y + uA
yB = y + uB
yC = y + uC
yD = y + uD

# Define colors and transparency
plt.rcParams.update({'font.size': 12})
scatter_color = 'navy'  # Color for scatter points
line_color = 'darkred'  # Color for the regression line
alpha_value = 0.4  # Transparency for the scatter points

# Fit a linear model for plotting the regression line
def fit_and_plot(x, y, ax, label, xtics=True, ytics=True):
    b1, b0 = np.polyfit(x, y, 1)
    order = np.argsort(x)
    
    # Plot scatter points with transparency and regression line
    ax.scatter(x, y, color=scatter_color, alpha=alpha_value)
    ax.plot(x[order], b0 + b1 * x[order], color=line_color)
    ax.set_title(label, fontsize=20)
    ax.set_xlim([2, 6])
    ax.set_ylim([0, 20])
    if xtics:
        xlabel = ax.set_xlabel('x')
    if ytics:
        ylabel = ax.set_ylabel('y')
    ax.set_yticks([0, 4, 8, 12, 16, 20])

# Create the 2x2 plot
fig, axs = plt.subplots(2, 2, figsize=(9, 6))

fit_and_plot(x, yA, axs[0, 0], 'A', xtics=False)
fit_and_plot(x, yB, axs[0, 1], 'B', xtics=False, ytics=False)
fit_and_plot(x, yC, axs[1, 0], 'C', xtics=True)
fit_and_plot(x, yD, axs[1, 1], 'D', xtics=True, ytics=False)

plt.tight_layout()
plt.savefig('examples/Quiz_Heteroskedasticitet.pdf')
plt.show()


# %% Figure 9.2: Heteroscedasticity depending on the distance from the mean
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
n = 1000
x = 2 + 4 * np.random.rand(n)
y_true = 2 + 2 * x

# Case 1: Variance increasing with distance from mean
u_increasing = np.random.normal(0, np.abs(0.25 + 0.75 * (x - x.mean())), size=n)
y_hetero_increasing = y_true + u_increasing

# Case 2: Variance decreasing with distance from mean
u_decreasing = np.random.normal(0, np.abs(-1.5 + 0.75 * (x.mean() - np.abs(x - x.mean()))), size=n)
y_hetero_decreasing = y_true + u_decreasing

# Create 1x2 plot
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot for increasing variance
axs[0].scatter(x, y_hetero_increasing, label='Data', color='blue', alpha = 0.4)
axs[0].plot(x, y_true, color='red', label='E(y|x)')
axs[0].set_title(r"Varians STIGENDE med afstand fra $\bar{x}$", fontsize=12)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

# Plot for decreasing variance
axs[1].scatter(x, y_hetero_decreasing, label='Data', color='green',alpha = 0.4)
axs[1].plot(x, y_true, color='red', label='E(y|x)')
axs[1].set_title(r"Varians FALDENDE med afstand fra $\bar{x}$", fontsize=12)
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.savefig('examples/heteroscedasticity_dist_to_xbar.pdf')
plt.show()

# %% Plotting the F-distribution
import scipy.stats as stats

# Define x range for plotting
x = np.linspace(0, 10, 1000)

# Define degrees of freedom for different F distributions
df_values = [10, 20, 30, 50, 100, 200]

# Chi-square distribution with df = 5
q = 10  # numerator degrees of freedom
chi2_dist = stats.chi2.pdf(x * q, df=q) * q

# Create the figure
plt.figure(figsize=(10, 6))

# Plot Chi-square distribution
plt.plot(x, chi2_dist, label=r'$\chi^2(5)$', linestyle='--', color='b', linewidth=3)

# Plot F distributions for different values of degrees of freedom
for df in df_values:
    f_dist = stats.f.pdf(x, q, df)  # F distribution with numerator df = 5
    plt.plot(x, f_dist, label=f'F(5, {df})')

# Add labels and title
plt.title("Convergence of F-distribution to Chi-square distribution", fontsize=14)
plt.xlabel("Value", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title='Distribution', fontsize=10)
plt.grid(True)

# If you are using this in your local environment, you can display the plot like this:
plt.show()

# %% -------------------------------------------
# Figure 9.4: y, residuals and residuals^2 vs. x plot
# ----------------------------------------------# 
import numpy as np
import matplotlib.pyplot as plt
import mymlr as mlr
import pandas as pd

# Generer og forbered data
np.random.seed(0)
x = 2 + 4 * np.random.rand(400)
y = 2 + 3 * x
uA = np.random.normal(0, 0.25 + 0.75 * (x - 2), size=len(x))
uB = np.random.normal(0, 1, size=len(x))
uC = np.random.normal(0, 2.5, size=len(x))
uD = np.random.normal(0, 0.25 + 0.75 * (6 - x), size=len(x))
data = pd.DataFrame({'x': x, 'yA': y + uA, 'yB': y + uB, 'yC': y + uC, 'yD': y + uD, 'const': 1})

# Plot y, residualer eller kvadrerede residualer for de fire DGP'er
def plot_diagnostics(df, xvar, yvar, mode='y', bins=10, raw=True,
                     binned=True, regression=True, labels=None, save=None):
    labels = yvar if labels is None else labels
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    for i, (x_col, y_col, ax) in enumerate(zip(xvar, yvar, axes.ravel())):
        m = mlr.ols(df[y_col], df[['const', x_col]])
        if mode == 'y':
            z, ylabel = df[y_col], 'y'
        elif mode == 'residuals':
            z, ylabel = m['u_hat'].ravel(), r'Residual $\hat u$'
            ax.axhline(0, color='black', linewidth=1)
        elif mode == 'squared residuals':
            z, ylabel = m['u_hat'].ravel() ** 2, r'Squared residual $\hat u^2$'
        else:
            raise ValueError("mode must be 'y', 'residuals' or 'squared residuals'")

        # Gør residualerne tydeligere end observationerne i det almindelige scatterplot
        raw_size, raw_alpha = (12, 0.25) if mode != 'y' else (5, 0.08)
        mlr.binscatter(df[x_col], z, bins=bins, ax=ax, raw=raw,
                       binned=binned, regression=regression,
                       raw_size=raw_size, raw_alpha=raw_alpha, raw_color='navy',
                       xlabel=x_col, ylabel=ylabel, title=labels[i])
        ax.set_xlim(2, 6)

    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    plt.show()

# Brug af funktionen
yvar = ['yA', 'yB', 'yC', 'yD']
xvar = ['x', 'x', 'x', 'x']
labels = ['A', 'B', 'C', 'D']

# Plot y med regressionslinjer
plot_diagnostics(data, xvar, yvar, mode='y', labels=labels,
                 save='examples/Heteroskedasticitet_y.pdf')

# Plot residualer uden regressionslinjer
plot_diagnostics(data, xvar, yvar, mode='residuals', regression=False,
                 labels=labels, save='examples/Heteroskedasticitet_residuals.pdf')

# Plot kvadrerede residualer med regressionslinjer
plot_diagnostics(data, xvar, yvar, mode='squared residuals', labels=labels,
                 save='examples/Heteroskedasticitet_squared residuals.pdf')
# %% -------------------------------------------
# Breusch-Pagan testet
# ----------------------------------------------
import mymlr as mlr
from scipy.stats import chi2  # Import chi2 for the Breusch-Pagan test


# Funktion til Breusch-Pagan testet
def breusch_pagan_test(df, y_col, x_col):
    # Estimer OLS modellen og gem residualerne
    res = mlr.ols(y=df[y_col], X=df[[x_col, 'const']])
    residuals = res['u_hat']

    # Kvadrerede residualer
    df['u2'] = residuals ** 2

    # Estimer en model med de kvadrerede residualer som afhængig variabel
    aux_model = mlr.ols(y=df['u2'], X=df[[x_col, 'const']])
    r_squared = aux_model['R_squared']

    # Antal observationer (n) og antal forklarende variable (k)
    n = len(df)
    k = len(df.columns) - 1  # minus 'y'

    # Beregn LM-teststørrelsen (Breusch-Pagan test)
    LM_stat = n * r_squared
    return LM_stat, 1 - chi2.cdf(LM_stat, df=k-1)  # Returner LM-statistikken og p-værdi

# Brug af funktionen til yA, yB, yC og yD
for y_col in ['yA', 'yB', 'yC', 'yD']:
    LM_stat, p_value = breusch_pagan_test(data, y_col, 'x')
    print(f"Breusch-Pagan test for {y_col}: LM-stat = {LM_stat:.2f}, p-værdi = {p_value:.4f}")

# %%
