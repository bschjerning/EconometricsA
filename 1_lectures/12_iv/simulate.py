

# %%
import numpy as np
import pandas as pd
import mymlr as mlr


# %% 
def setparam():
    theta = {
        'alpha':   -5,  # Price coefficient in utility
        'beta':    5,   # Coefficients on product characteristics
        'gamma_x': 5,   # Coefficients on cost characteristics
        'gamma_w': .5   # Coefficient on cost shifters
    }
    mu = {
        'x':    5,      # Mean of product characteristics
        'w':    5       # Mean of cost shifters
    }
    sigma = {
        'x':  1,        # Standard deviation of product characteristics
        'w':  1,        # Standard deviation of cost shifters
        'xi': 1,        # Std dev of demand shock
        'nu': 1         # Std dev of supply shock
    }
    
    return theta, mu, sigma

# Marginal Cost Function
def f_marginal_cost(x_jm, w_jm, gamma_x, gamma_w, nu_jm):
    mc_jm = x_jm * gamma_x + w_jm * gamma_w + nu_jm
    return mc_jm

# Demand Function (Logit demand)
def f_demand_S(p_jm, x_jm, beta, alpha, xi_jm, M, J):
    # Compute utility for each product in each market 
    u_jm = p_jm*alpha + x_jm * beta + xi_jm         

    # Compute market shares, S_jm = exp(u_jm) / (sum_j(exp(u_jm)))
    exp_u=np.exp(u_jm)                                  
    logsum = np.sum(exp_u.reshape(M, J), axis=1)        
    S_jm = exp_u / np.repeat(1+logsum, J).reshape(-1,1)             
    return S_jm

# Inverse supply function: returns price given the quantity supplied (S_jm).
def f_supply_P(mc_jm, S_jm, alpha):
    # Derivative of logit demand with respect to price
    dS_dp = -alpha * S_jm * (1 - S_jm)  
    # First-order condition: p = mc + S / (dS/dp)
    p_jm = mc_jm + S_jm / dS_dp
    return p_jm


# Direct supply function: returns quantity supplied (S_jm) given price.
# Derived from FOC: p = mc + S / (dS/dp).
def f_supply_S(p_jm, mc_jm, alpha):
    # Price-cost margin determines supplied quantity
    eta = alpha * (p_jm - mc_jm)  
    # Derived from FOC: S_jm = 1 + 1 / eta
    S_jm = np.maximum(0, 1 + 1 / eta)  # Ensure quantity is non-negative
    return S_jm

# Equilibrium Computation
def compute_equilibrium(mc_jm, x_jm, beta, alpha, xi_jm, M, J, tol=1e-6, max_iter=1000):
    # Solve for equilibrium prices using iterative fixed-point algorithm

    p_guess = mc_jm +  0.1  # Initialize prices (initial guess)
    # Iteratively adjust prices to find equilibrium
    for iteration in range(max_iter):
        # Compute market shares from consumer demand
        S_jm = f_demand_S(p_guess, x_jm, beta, alpha, xi_jm, M, J)

        # Calculate optimal prices based demand
        p_jm = f_supply_P(mc_jm, S_jm, alpha)
        
        # Check convergence (if the change in prices is below tolerance)
        if np.max(np.abs(p_jm - p_guess)) < tol:
            print(f"Equilibrium reached in {iteration} iterations.")
            break
        
        # Update prices for the next iteration
        p_guess = p_jm
    else:
        print("Equilibrium not reached within max iterations.")
    return p_jm

def sim_exog(mu, sigma, M=1, J=5): 
    # Simulate product characteristics, costs, and shocks
    norm = np.random.normal
    x_jm = norm(size=(M * J, 1), loc=mu['x'], scale=sigma['x'])
    w_jm = norm(size=(M * J, 1), loc=mu['w'], scale=sigma['w'])
    xi_jm = norm(scale=sigma['xi'], size=(M * J,1))
    nu_jm = norm(scale=sigma['nu'], size=(M * J,1))

    return x_jm, xi_jm, w_jm, nu_jm

# Simulator Function
def simulate_market(theta, mu, sigma, M=50, J=10):

    # Unpack parameters
    alpha = theta['alpha']
    beta = theta['beta']
    gamma_x = theta['gamma_x']
    gamma_w = theta['gamma_w']
                        
    # Step 1: Simulate product characteristics, costs, and shocks
    x_jm, xi_jm, w_jm, nu_jm= sim_exog(mu, sigma, M, J) 

    # Step 2: Compute marginal costs
    mc_jm = f_marginal_cost(x_jm, w_jm, gamma_x, gamma_w, nu_jm)

    # Step 3: Compute equilibrium prices and market shares
    p_jm = compute_equilibrium(mc_jm, x_jm, beta, alpha, xi_jm, M, J, tol=1e-6, max_iter=1000);  
    
    # Step 4: Compute market shares in equilibrium
    S_jm = f_demand_S(p_jm, x_jm, beta, alpha, xi_jm, M, J)

    # Create DataFrame with simulated data
    df = pd.DataFrame({
        'price': p_jm.flatten(),
        'share': S_jm.flatten(),
        'x1': x_jm.flatten(),
        'xi': xi_jm.flatten(),
        'w': w_jm.flatten(),
        'nu': nu_jm.flatten(),
        'mc': mc_jm.flatten(),
        'markup': (p_jm - mc_jm).flatten()
    })
    df['const'] = 1
    # df.dropna()  # Drop any missing values
    
    return df

# Call the simulate_market function
theta, mu, sigma = setparam()

theta['alpha'] = -.5
theta['beta'] = 0.1
theta['gamma_x'] = 0.1
theta['gamma_w'] = .5
sigma['xi'] = 2
sigma['nu'] = .5
sigma['x'] = 1
sigma['w'] = 1

df = simulate_market(theta, mu, sigma, M=500, J=10)

# Display the first few rows of the simulated data
print(df.describe())
# Estimate the demand function using OLS
y= np.log(df['share'])
m_ols=mlr.ols(y=y, X=df[['const', 'x1', 'price']])
mlr.output(m_ols)
df[['w2']]=df[['w']]**2
df[['w3']]=df[['w']]**3
m_iv = mlr.tsls(y=y, X1=df[['const', 'x1']], X2=df[['price']], Ze=df[['w', 'w2', 'w3']])
# m_iv = mlr.tsls(y=y, X1=df[['const', 'x1']], X2=df[['price']], Ze=df[['w']])
mlr.output(m_iv, title='IV Demand Estimation')


import matplotlib.pyplot as plt

plt.scatter( df[['price']], np.log(df['share']), label="Demand", color='b', alpha=0.5, s=5)

# Add labels and title
plt.xlabel('Price')
plt.ylabel('Log of Market Share')
plt.title('Simulated Demand: Price vs Log Market Share')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
plt.scatter( df[['price']], df['w'], label="Demand", color='b', alpha=0.5, s=5)
plt.show()

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_supply_demand(alpha, beta, gamma_x, gamma_w, J, j=0):
    M=1

    x_jm, xi_jm, w_jm, nu_jm=sim_exog(
        M=M, J=J, mu_x=[5, 5], mu_w=5,
        sigma_x=[0, 0], sigma_w=0, 
        sigma_xi=0, sigma_nu=0
        )
    
    mc_jm = f_marginal_cost(x_jm, w_jm, gamma_x, gamma_w, nu_jm)
    p_jm = compute_equilibrium(mc_jm, x_jm, beta, alpha, xi_jm, M, J, tol=1e-6, max_iter=1000);  

    demand_quantities = []
    supply_quantities = []

    prices = np.linspace(mc_jm[j,0], mc_jm[j,0]+2, 100)  # Create a range of prices from 1 to 10
    # Loop through each price and calculate corresponding demand and supply
    for p in prices:
        # Simulate demand and supply for this price
        p_jm[j,0] = p
        demand = f_demand_S(p_jm, x_jm, beta, alpha, xi_jm, M, J)
        supply = f_supply_S(p_jm, mc_jm, alpha)

        demand_quantities.append(demand[j])
        supply_quantities.append(supply[j])

    # Plot the supply and demand curves
    plt.figure(figsize=(10, 6))
    
    plt.scatter(prices, demand_quantities, label="Demand", color='b')
    plt.scatter(prices, supply_quantities, label="Supply", color='r')

    plt.xlabel(f'Price, product {j+1}')
    plt.ylabel(f'Quantity, product {j+1}')
    plt.title(f'Supply and Demand Curves for product {j+1}')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Example of calling the plot function with parameters
alpha = -2
plot_supply_demand(alpha, beta=[1, 1], gamma_x=[0.3], gamma_w=0.1, J=4, j=0)
# %%
