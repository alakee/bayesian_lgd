#============= load packages ============#

import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random as rd

#=======================================#


#=========== define functions ==========#

def simulate_lgd(mu, psi, n):
    """
    Simulates LGD data. Assumes LGD follows a beta distribution. 
    Note: Beta distribuion is parameterised as beta(mu * psi, (1 - mu) * psi)
    so that the expected value is mu.
    """
    
    alpha0 = mu * psi
    beta0 = (1 - mu) * psi
    
    df = pd.DataFrame({"x": np.random.beta(alpha0,beta0, n)})
    
    # return the simulated values in a dataframe 
    return(df)

def plot_trace(param, param_name='parameter'):
  """
  Plot the trace and posterior of a parameter.
  """
  
  # Summary statistics
  mean = np.mean(param)
  median = np.median(param)
  cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
  
  # Plotting
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(param)
  plt.xlabel('Iteration')
  plt.ylabel('Posterior Samples {}'.format(param_name))
  plt.axhline(mean, color='r', lw=2, linestyle='--')
  plt.axhline(median, color='c', lw=2, linestyle='--')
  plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
  plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
  plt.title('Trace and Posterior Distribution for {}'.format(param_name))

  plt.subplot(2,1,2)
  plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
  plt.xlabel('Posterior Samples {}'.format(param_name))
  plt.ylabel('Density')
  plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
  plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
  plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
  plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
  
  plt.gcf().tight_layout()
  plt.legend()
  
  

def bootstrap_lgd(mu, psi, samples, n_sims):
    """
   Generates bootstrapped LGD estimates.
  """
        
    rd.seed(314)
    bootstrap_data = pd.DataFrame({"x": np.random.beta(mu*psi,(1 - mu) * psi, samples)})
    
    lgd_list = []
    
    for i in range(0, n_sims):
        
        resampled_data = bootstrap_data.sample(n = len(bootstrap_data), replace = True)
        
        lgd_list.append(   (np.mean(resampled_data)))
        
    return np.array(lgd_list)


#=====================================#


    

#======== Define the model =========#
  
lgd_model = """
data {

  int<lower=0> N;
  real<lower=0> expected_lgd;   // what we expect the LGD to be
  real<lower=0> precision_lgd;  // how sure we are about our expectations
  real<lower=0> gamma_shape;   
  real<lower=0> gamma_scale;  
  real x[N];
    
}
parameters {
  real<lower = 0> mu;
  real<lower = 0> psi;
}
model {

  // truncate the normal distribution to (0, 1) - LGD bounds
  mu ~ normal(expected_lgd, precision_lgd) T[0, 1];
  psi ~ gamma(gamma_shape, gamma_scale) ;

  // this reparameterisation ensures E(X) = mu 
  // (we're basically sampling the expected LGD)
  x ~ beta(mu*psi, (1 - mu) * psi);
}
"""
# write the model to a stan file
text_file = open("C:/Users/aaron/lgd_model.stan", "w")
text_file.write(lgd_model)
text_file.close()

# compile the model
stan_model = ps.StanModel(file="C:/Users/aaron/lgd_model.stan")




#============== Set up the data ===========#
df = simulate_lgd(mu = 0.1,psi = 0.15, n = 100);

# plot of the simulated data
plt.figure()
plt.hist(x = df['x'])
plt.xlabel("Simulated LGD")
plt.ylabel("Count")
plt.title("Simulated Loss Given Default (LGD)")


# run hamiltonian monte carlo (stan implements no-u-turn extension)
data_dict = {"x": df["x"],
             "N": len(df),
             "expected_lgd":0.25,
             "precision_lgd":0.01, 
             "gamma_shape":2,
             "gamma_scale":0.1}

stan_fit = stan_model.sampling(data=data_dict)


# summarise the sampling chains
stan_results = pd.DataFrame(stan_fit.extract())
print(stan_results.describe())



# plot the results
plot_trace(stan_fit['mu'], "$\mu$")
plot_trace(stan_fit['psi'], "$\psi$")


#========== Bootstrap =============#

run_bootstrap = True


if run_bootstrap:
    
    # run the bootstrap
    bs_50 = bootstrap_lgd(mu = 0.1, psi = 0.2, samples = 50, n_sims = 1000)
    bs_100 = bootstrap_lgd(mu = 0.1, psi = 0.2, samples = 100, n_sims = 1000)
    bs_500 = bootstrap_lgd(mu = 0.1, psi = 0.2, samples = 500, n_sims = 1000)
    
    # plot the results
    plt.figure()
    plt.subplot(3,1,1)
    plt.hist(bs_50)
    plt.xlabel("Average LGD")
    plt.ylabel("Count")
    plt.title("Bootstrapped Average LGD (50 Observations)")
    
    plt.subplot(3,1,2)
    plt.hist(bs_100)
    plt.xlabel("Average LGD")
    plt.ylabel("Count")
    plt.title("Bootstrapped Average LGD (100 Observations)")
    
    plt.subplot(3,1,3)
    plt.hist(bs_500)
    plt.xlabel("Average LGD")
    plt.ylabel("Count")
    plt.title("Bootstrapped Average LGD (500 Observations)")
