#%% Figure 2.1: Conditional Expectation of u given x, E(u|x)
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 1000  # Number of observations
min_x, max_x = -3, 3  # Range for uniform distribution of x
std_u1 = 1  # Standard deviation of u1 (Dataset 1)
std_u2 = 1  # Standard deviation of u2 (Dataset 2)
non_linear_factor = 0.5  # Degree of non-linear relationship in Dataset 2

# Generate random variables x and u for Dataset 1 (E(u|x) = 0)
np.random.seed(42)  # For reproducibility
x1 = np.random.uniform(min_x, max_x, n)
u1 = np.random.normal(0, std_u1, n)  # E(u|x) = 0

# Generate random variables x and u for Dataset 2 (E(u|x) ≠ 0)
x2 = np.random.uniform(min_x, max_x, n)
u2 = non_linear_factor * (x2 ** 2 - np.mean(x2 ** 2)) + np.random.normal(0, std_u2, n)  # Non-linear relationship

# Adjust u2 to ensure E(u2) = 0
u2 = u2 - np.mean(u2)

# Verify cov(x, u) = 0 for both datasets
print("Covariance of x1 and u1:", np.cov(x1, u1)[0, 1])
print("Covariance of x2 and u2:", np.cov(x2, u2)[0, 1])

# Plotting
plt.figure(figsize=(16, 8))

# Plot for Dataset 1
plt.subplot(1, 2, 1)
plt.scatter(x1, u1, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("E(u|x) = 0, E(u) = 0, cov(u,x)=0", fontsize=30)
plt.xlabel("x", fontsize=30)
plt.ylabel("u", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# Plot for Dataset 2
plt.subplot(1, 2, 2)
plt.scatter(x2, u2, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("E(u|x) ≠ 0, E(u) = 0, cov(u,x)=0", fontsize=30)
plt.xlabel("x", fontsize=30)
plt.ylabel("u", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.tight_layout()

# now save the figure as a pdf
plt.savefig("E[u|x]_plots.pdf")

plt.show()
# %% Figure 2.2: Linear regression with different R-squared values
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Parameters
printR2 = True  # Print R-squared values
beta0, beta1, n = 2, 3, 1000  # Linear model parameters and sample size
sigma_u = [5, 4]  # Standard deviations of noise for the two datasets
sigma_x = [1, 1]  # Standard deviations of x for the two datasets
# Prepare figure
plt.figure(figsize=(14, 6))

# Loop through the two scenarios
for i in range(2):
    x = np.random.normal(0, sigma_x[i], n)
    u = np.random.normal(0, sigma_u[i], n)
    y = beta0 + beta1 * x + u
    
    # Fit the linear model
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    R2 = model.score(x.reshape(-1, 1), y)
    
    # Plot
    plt.subplot(1, 2, i+1)
    plt.scatter(x, y, alpha=0.5)
    plt.plot(x, y_pred, color='red', linewidth=2)
    plt.title(f"Eksempel {i+1}", fontsize=30)
    plt.xlabel("x", fontsize=30)
    plt.ylabel("y", fontsize=30)

    # Print the R-squared value
    if printR2:
        print(f"R-squared for example {i+1}: {R2:.2f}")


plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_linear_regression(beta0=2, beta1=3, n=1000, sigma_u=[5, 3], sigma_x=[1, 1], printR2=False):
    np.random.seed(42)
    
    plt.figure(figsize=(14, 6))
    
    for i in range(2):
        x = np.random.normal(0, sigma_x[i], n)
        u = np.random.normal(0, sigma_u[i], n)
        y = beta0 + beta1 * x + u

        # Calculate OLS estimates using explicit formulas
        var_x = np.sum((x - np.mean(x)) ** 2) / n
        cov_xy = np.sum((x - np.mean(x)) * (y - np.mean(y))) / n
        beta1_hat = cov_xy / var_x
        beta0_hat = np.mean(y) - beta1_hat * np.mean(x)
        y_pred = beta0_hat + beta1_hat * x

        # Calculate R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        R2 = 1 - (ss_residual / ss_total)
        
        # Plot
        plt.subplot(1, 2, i + 1)
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, y_pred, color='red', linewidth=2)
        
        # Create subtitle with parameter values
        subtitle = fr"$\sigma_x={sigma_x[i]}$, $\sigma_u={sigma_u[i]}$, " \
                   fr"$\beta_0={beta0}$, $\beta_1={beta1}$"
        
        plt.title(subtitle, fontsize=30)
        plt.xlabel("x", fontsize=30)
        plt.ylabel("y", fontsize=30)
        plt.grid(True)

        # Print the parameters
        print(f"Eksempel {i + 1}: \nbeta0: {beta0_hat:.2f}, beta1: {beta1_hat:.2f}, sigma_u: {sigma_u[i]}, sigma_x: {sigma_x[i]}")
        if printR2:
            print(f"R-squared: {R2:.2f}\n")
    
    plt.suptitle(r"$y = \beta_0 + \beta_1 x + u, \quad x \sim N(0, \sigma_x^2), \quad u \sim N(0, \sigma_u^2)$", fontsize=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return plt

# Plot for slides on goodness-of-fit
plt = plot_linear_regression(beta0=1, beta1=2, sigma_u=[4, 2], sigma_x=[2, 2])
plt.savefig("R2_example.pdf")
plt.show()


# %% 
