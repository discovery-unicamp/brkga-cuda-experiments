import numpy as np
from scipy import stats
import statsmodels.stats.weightstats as smws

nobs_all = [10, 100, 1000]
sigma = 0.5

seed = 628561  #chosen to produce nice result in small sample
for nobs in nobs_all:
    np.random.seed(seed)
    payoff_s1 = sigma * np.random.randn(nobs)
    payoff_s2 = 0.1 + sigma * np.random.randn(nobs)

    p1 = stats.ttest_ind(payoff_s1, payoff_s2)[1]
    p2 = smws.ttost_ind(payoff_s1, payoff_s2, -0.5, 0.5, usevar='pooled')[0]

    print ('nobs:', nobs, 'diff in means:', payoff_s2.mean() - payoff_s1.mean())
    print ('ttest:', p1,    ['not different', 'different    '][p1 < 0.05])
    print ('   tost:', p2, ['different    ', 'not different'][p2 < 0.05])
