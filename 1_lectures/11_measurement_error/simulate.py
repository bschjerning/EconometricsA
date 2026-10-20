# %% simulate.py
# Author: Bertel Schjerning
# Monet Carlo simulation to illustrate properties of OLS
# Part 1: Simulate data
# - Simulate data from a linear model with two rhs variables 
# - Plot histograms of the data
# - Estimate the model using OLS
# 
# Part 2: Monte Carlo simulation
# - Simulate data S times and estimate the model each time
# - Plot histograms of the estimated coefficients
# - Investigate the effect of sample size on the estimated coefficients
# Properties of OLS to illustrate:
# - Unbiasedness: E(β̂) = β
# - Consistency: plim(β̂) = β
# - Asymptotic Normality and Root-n consistency 
#   sqrt(n)(β̂ - β) ->d N(0, n σ²(X'X)^(-1))
#   Show that sqrt(n)se(β̂) is constant, but se(β̂) decreases with n
#   Show that distribution of β̂ is normal when n is large
#   Show that distribution of β̂ is normal when n is small and errors are normal
# Test of hypothesis
#   Show that t-ratios are t-distributed in small samples, but asymptotically standard normal
#   Type I error: Under the null, p-values should equal the significance level if the null is true
#   Power: Compare the probability of rejecting the null when it is false
#   Show that power increases with sample size, effect size, and significance level, but decreases with noise level
# Violation of assumptions
#   Show that OLS is unbiased, but inefficient when errors are heteroskedastic
#   Show that OLS is unbiased and inconsistent when errors are correlated with the rhs variables
#   Perfect multicollinearity: Show that standard errors are infinite
#   Multicollinearity: Show that OLS variance of β̂ increases with correlation between rhs variables





# Part 2: Monte Carlo simulation




# %% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import mymlr as mlr
# %% Part 1: Simulate data
def simulate(n=100, beta=[0, 1, 2], sigma_u=1,
             mu_x1=0, sigma_x1=1, mu_eta=2, sigma_eta=1,
             rho_x=1, rho_xu=0, dist='normal'):
    
    if dist == 'normal':
        gen = np.random.normal
    if dist == 'uniform':
        gen = np.random.uniform(-np.sqrt(3), np.sqrt(3), size)*sigma + mu
    elif dist == 't':
        gen = lambda mu, sigma, size: np.random.standard_t(3, size)*sigma + mu

    const = np.ones(n)
    u = gen(0, sigma_u, size=n)
    x1 = gen(mu_x1, sigma_x1, size=n)
    eta = gen(mu_eta, sigma_eta, size=n)
    x2 = rho_x * x1 + rho_xu * u + eta
    y = beta[0] + beta[1] * x1 + beta[2] * x2 + u
    
    return pd.DataFrame({'y': y,'const': const,'x1': x1,'x2': x2, 'u': u, 'eta': eta})

# %% Plot histogram of statistics 
import matplotlib.pyplot as plt
from scipy.stats import norm
def histogram(stat, truestat=None, title='Histogram', xlim=None, bins=100, normdensity=True):
    plt.rcParams.update({'font.size': 16})
    if truestat is None:
        truestat = stat.mean()
    fig, ax = plt.subplots(1, len(stat.columns), figsize=(15, 5))
    for i, lbl in enumerate(stat.columns):
        mcsd = stat[lbl].std();
        ax[i].hist(stat[lbl], bins=100, density=True, alpha=0.7)
        if normdensity:
            ax[i].axvline(x=truestat[i], color='red', linestyle='--')
            x = np.linspace(truestat[i]-4*mcsd, truestat[i]+4*mcsd, 100)
            y = norm.pdf(x, loc=stat[lbl].mean(), scale=mcsd)
            ax[i].plot(x, y, color='black')
        ax[i].set_title(lbl)
        if xlim is not None:
            ax[i].set_xlim(xlim)
        else:
            ax[i].set_xlim(truestat[i]-4*mcsd, truestat[i]+4*mcsd)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

# Simulate data
df = simulate(n=10000, dist='t', sigma_u=.1)
histogram(df[['y','x1','x2']], title='Histogram of y, x1, and x2');
histogram(df[['eta','u']], title=f'Histogram of $\eta$ and u');

# %% Estimate the model using OLS
df = simulate(n=10)
res=mlr.ols(y=df['y'], X=df[['const', 'x1', 'x2']])
mlr.output(res)
# %% Part 2: Monte Carlo simulation
def monte_carlo(simulator, estimator, S=1000):
    # Loop over repetitions
    for s in range(S):
        # Simulate data
        data = simulator()        

        # Estimate model
        res = estimator(data['y'], data[['const', 'x1', 'x2']])

        # Store results
        if s==0:
            beta_hat = pd.DataFrame(index=range(S), columns=res['lbl_X'])
            se = pd.DataFrame(index=range(S), columns=res['lbl_X'])

        beta_hat.loc[s]= res['beta_hat'].T[0]
        se.loc[s]= res['se'].T[0]

    return beta_hat, se

# Perform Monte Carlo simulation
simulator1 = lambda: simulate(n=100, beta=[0, 1, 2], sigma_u=1)  
b, se = monte_carlo(simulator1, mlr.ols, S=10000) 
# %% Plot histogram of estimated coefficients
import matplotlib.pyplot as plt
from scipy.stats import norm

beta0 = [0, 1, 2]
simulator1 = lambda: simulate(n=5, beta=beta0, sigma_u=1, dist='t')  
b, se = monte_carlo(simulator1, mlr.ols, S=10000)
# Create a summary table
summary_table = pd.DataFrame({
    'Parameter': b.columns,
    'True Value': [beta0],
    'Mean Estimate': b.mean(),
    'MC Standard Deviation': b.mean(),
    'Average SE': se.mean()
})

histogram(stat=b, truestat=beta0, title='Estimated coefficients');
histogram(se, title='Estimated Standard Errors');
histogram((b-beta0)/se,  np.mean((b-beta0)/se), title='T-ratios');

# %%  
def summary_table(b,se): 
    sumtab = {'Parameter': b.columns,'True Value': beta0, 'Mean Estimate': b.mean(),
        'MC Standard Deviation': b.mean(),'Average SE': se.mean()}
    display(pd.DataFrame(sumtab))
summary_table(b,se)

# %% Investigate the effect of sample size
beta0 = [0, 1, 2]
N1 = np.arange(5, 51, 1)
N2 = np.arange(60, 101, 10)
N3 = np.arange(200, 1001, 100); # start, stop, step
N = np.concatenate([N1, N2, N3])
mean_b = np.zeros((len(N), len(beta0)))
mean_se = np.zeros((len(N), len(beta0)))
mean_t = np.zeros((len(N), len(beta0)))
mean_mcse = np.zeros((len(N), len(beta0)))
for i, n in enumerate(N):
    simulator1 = lambda: simulate(n=n, beta=beta0, sigma_u=1, dist='t')  
    b, se = monte_carlo(simulator1, mlr.ols, S=1000)
    mean_b[i] = b.mean()
    mean_se[i] = se.mean()
    mean_t[i] = (b-beta0).mean()/se.mean()
    mean_mcse[i] = se.mean()

# %% Plot results
plt.plot(N, mean_b[:,0])
plt.plot(N, 0*mean_b[:,0])
# add confidence intervals as shaded area
plt.fill_between(N, mean_b[:,0]-1.96*mean_se[:,0], mean_b[:,0]+1.96*mean_se[:,0], alpha=0.2)
plt.xlabel('Sample size')
plt.ylabel('Estimated coefficients')
plt.legend(['beta0'])
plt.show()

plt.plot(N, mean_se)
plt.xlabel('Sample size')
plt.ylabel('Estimated standard errors')
plt.legend(['beta0', 'beta1', 'beta2'])
plt.show()

plt.plot(N, mean_t)
plt.xlabel('Sample size')
plt.ylabel('Estimated t-ratios')
plt.legend(['beta0', 'beta1', 'beta2'])
plt.show()

plt.plot(N, mean_mcse)
plt.xlabel('Sample size')
plt.ylabel('Mean standard errors')
plt.show()

plt.plot(N, np.sqrt(N).reshape(-1, 1)*mean_se)
plt.xlabel('Sample size')
plt.ylabel('$\sqrt{n}$se($\hat{\\beta}$)')
plt.show()

# %%
