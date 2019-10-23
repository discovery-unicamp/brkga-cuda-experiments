'''Test of Equivalence and Non-Inferiority

currently only TOST for paired sample
Application for example bioequivalence



Author: Josef Perktold
License: BSD-3

'''


import numpy as np
from scipy import stats
from random import randint as rand

def tost_paired(y, x, low, upp, transform=None):
    '''test of (non-)equivalence for paired sample

    TOST: two one-sided t tests

    null hypothesis  x - y < low or x - y > upp
    alternative hypothesis:  low < x - y < upp

    If the pvalue is smaller than a threshold, then we reject the hypothesis
    that there is difference between the two samples larger than the one
    given by low and upp.

    Parameters
    ----------
    y, x : array_like
        two paired samples
    low, upp : float
        equivalence interval low < x - y < upp
    transform : None or function
        If None (default), then the data is not transformed. Given a function
        sample data and thresholds are transformed. If transform is log the
        the equivalence interval is in ratio: low < x / y < upp

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1 : tuple of floats
        test statistic and pvalue for lower threshold test
    t2, pv2 : tuple of floats
        test statistic and pvalue for upper threshold test

    Notes
    -----
    tested on only one example
    uses stats.ttest_1samp which doesn't have a real one-sided option

    '''
    if transform:
        y = transform(y)
        x = transform(x)
        low = transform(low)
        upp = transform(upp)
    t1, pv1 = stats.ttest_1samp(x - y, low)
    t2, pv2 = stats.ttest_1samp(x - y, upp)
    return max(pv1, pv2)/2., (t1, pv1 / 2.), (t2, pv2 / 2.)
    

if __name__ == '__main__':

    #example from http://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_ttest_sect013.htm
    # raw = np.array('''\
    #    103.4 90.11  59.92 77.71  68.17 77.71  94.54 97.51
    #    69.48 58.21  72.17 101.3  74.37 79.84  84.44 96.06
    #    96.74 89.30  94.26 97.22  48.52 61.62  95.68 85.80'''.split(), float)

    # x, y = raw.reshape(-1,2).T

    # print (tost_paired(y, x, 0.8, 1.25, transform=np.log))

    x = np.array([932403.19, 137043.38, 587952.88, 2548211.5, 3200853.75, 70627200.0, 20244162.0, 9340685.0, 13156370.0, 7937487.0, 10408383.0, 54301180.0, 48480740.0, 21154658.0, 90703736.0, 30736370.0, 34199256.0, 16671520.0, 54033320.0, 94026648.0, 90368224.0, 118205640.0])
    y = np.array([948092.81, 126053.04, 579833.19, 2465718.0, 3200197.5, 70621040.0, 20468628.0, 9406424.0, 12508235.0, 7895304.0, 10367363.0, 53765484.0, 48044316.0, 21144074.0, 86311328.0, 30961720.0, 33993396.0, 16695521.0, 53788440.0, 94970560.0, 90397880.0, 117965896.0])

    print (tost_paired(y, x, 0.95, 1.05,transform=np.log))
