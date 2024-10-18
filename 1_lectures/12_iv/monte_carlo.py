import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate(simulator, estimator, S=1000):
    # Loop over repetitions
    for s in range(S):
        # Simulate data
        df = simulator()        

        # Estimate model
        res = estimator(df)

        # Store results
        if s==0:
            beta_hat = pd.DataFrame(index=range(S), columns=res['lbl_X'])
            se = pd.DataFrame(index=range(S), columns=res['lbl_X'])

        beta_hat.loc[s]= res['beta_hat'].T[0]
        se.loc[s]= res['se'].T[0]

    return beta_hat, se

def summary_table(b,se, beta0): 
    sumtab = {'Parameter': b.columns,'True Value': beta0, 'Mean Estimate': b.mean(),
        'MC Standard Deviation': b.std(),'Average SE': se.mean()}
    display(pd.DataFrame(sumtab))

def histogram(stat, truestat=None, title='Histogram', xlim=None, bins=50, normdensity=True):
    plt.rcParams.update({'font.size': 16})
    if truestat is None:
        truestat = stat.mean()
    fig, ax = plt.subplots(1, len(stat.columns), figsize=(15, 4))
    for i, lbl in enumerate(stat.columns):
        mcsd = stat[lbl].std();
        ax[i].hist(stat[lbl], bins=bins, density=True, alpha=0.7)
        ax[i].axvline(x=truestat[i], color='red', linestyle='--')
        if normdensity:
            x = np.linspace(truestat[i]-4*mcsd, truestat[i]+4*mcsd, 100)
            y = norm.pdf(x, loc=truestat[i], scale=mcsd)
            ax[i].plot(x, y, color='black')
        ax[i].set_title(lbl)
        if xlim is not None:
            ax[i].set_xlim(xlim)
        else:
            ax[i].set_xlim(np.minimum(truestat[i]-4*mcsd, stat[lbl].mean()-4*mcsd), 
                           np.maximum(truestat[i]+4*mcsd, stat[lbl].mean()+4*mcsd))

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

