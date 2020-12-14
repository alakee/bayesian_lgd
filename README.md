# bayesian_lgd

When write-off data is scarce, prior beliefs can be used to help estimate Loss Given Default (LGD). These prior beliefs can be incorporated using Bayesian methods. Note that if you have ‘unclear’ prior beliefs out the LGD, and not a lot of data, the likelihood will dominate the posterior. This means the LGD estimate will be approximately equal to the maximum likelihood estimate (MLE). 

It is assumed that the LGD follows a beta distribution.  The beta distribution has been reparameterised so that prior beliefs can be easily incorporated. Note that by assuming a beta distribution for the LGD, we implicitly force the LGD to be truncated between 0 and 1. This is not overly restrictive since the LGD rarely exceeds 1. 
Prior beliefs about the expected value of the LGD are incorporated using a truncated normal distribution on mu. The gamma distribution is a natural choice of prior for psi, and is used to impose beliefs on the variance. We use a gamma distribution since only positive values are allowed for the variance (although, any distribution with positive support would do do). 

To estimate the LGD under the prior beliefs, PyStan is used. PyStan implements a variant of the Hamiltonian Monte Carlo (HMC) algorithm to efficiently draw samples from the posterior distribution. 

