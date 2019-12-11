import itertools

import pandas as pd
import statsmodels
import statsmodels.api as sm
import numpy as np
import hurst
from math import log, e
# import pyrem
from collections import Counter
from rpy2.robjects import r, pandas2ri, FloatVector
from rpy2.robjects import IntVector, Formula, packages
from rpy2.robjects.packages import importr

stats = packages.importr('stats')
tseries = packages.importr('tseries')
forecast = packages.importr('forecast')
base = importr('base')
# _as = importr('as.ts')
from rpy2.robjects.packages import STAP
mfunc = 'myasts <- function(dobj){return(as.ts(dobj))}'
myasts = STAP(mfunc, "myasts")
# np.array(myasmatrix.myasmatrix(rres))
pandas2ri.activate()

from arch import arch_model
from entropy import spectral_entropy, perm_entropy, svd_entropy, app_entropy, sample_entropy, lziv_complexity
from arch.unitroot import PhillipsPerron
# import fracdiff



ts = pd.read_csv("./test_set.csv")
ts.reset_index(inplace=True)
ts['barTimestamp'] = pd.to_datetime(ts['barTimestamp'])
ts = ts.set_index('barTimestamp')
#ts = ts['close']

# ts = ts.set_index('barTimestamp')


def acf_features(x):
    m = 1

    acfx = sm.tsa.stattools.acf(x, nlags=max(m, 10), missing='none')
    acfdiff1x = sm.tsa.stattools.acf(x.diff(periods=1)[1:], nlags=10, missing='none')

    x2 = x.diff(periods=1)[1:]
    acfdiff2x = sm.tsa.stattools.acf(x2.diff(periods=1)[1:], nlags=10, missing='none')

    x_acf1 = acfx[1]
    x_acf10 = sum(acfx[1:]**2)
    diff1_acf1 = acfdiff1x[1]
    diff2_acf1 = acfdiff2x[1]
    diff1_acf10 = sum(acfdiff1x[1:]**2)
    diff2_acf10 = sum(acfdiff2x[1:]**2)

    return x_acf1, x_acf10, diff1_acf1, diff1_acf10, diff2_acf1, diff2_acf10

def pacf_features(x):
    m = 1

    x_pacf5 = sm.tsa.stattools.pacf(x, nlags=max(m, 5), method='ywm')

    x_pacf5 = sum((x_pacf5[1:6])**2)

    diff1x_pacf5 = sm.tsa.stattools.pacf(x.diff(periods=1)[1:], nlags=5, method='ywm')
    diff1x_pacf5 = sum(diff1x_pacf5[1:] ** 2)

    x2 = x.diff(periods=1)[1:]
    diff2x_pacf5 = sm.tsa.stattools.pacf(x2.diff(periods=1)[1:], nlags=5, method='ywm')
    diff2x_pacf5 = sum(diff2x_pacf5[1:] ** 2)

    return x_pacf5, diff1x_pacf5, diff2x_pacf5

#
# holt_parameters <- function(x) {
#   # parameter estimates of holt linear trend model
#   fit <- forecast::ets(x, model = c("AAN"))
#   params <- c(fit$par["alpha"], fit$par["beta"])
#   names(params) <- c("alpha", "beta")
#   return(params)
# }

def holt_parameters(x):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    print(x)
    ets_model = ExponentialSmoothing(x, trend='add')#, seasonal='None')
    ets_fit = ets_model.fit()
    alpha, beta = ets_fit.params['smoothing_level'], ets_fit.params['smoothing_slope']
    return alpha, beta



def entropy4(x, normalize = False, base=None):

    value, counts = np.unique(x, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    entr = -(norm_counts * np.log(norm_counts)/np.log(base)).sum()
    if normalize is True:
        entr /= np.log2(len(norm_counts))
    print(entr)
    entr += spectral_entropy(x, sf=len(x), method='fft', normalize=normalize)
    return entr / 2


def entropy(x):
    print(x)
    return spectral_entropy(x, sf=len(x), method='fft', normalize=True)
# print(acf_features(ts['close']))
# print(pacf_features(ts['close']))

def scale_ts(x):
    std = np.std(x)
    mean = np.mean(x)
    x = (x - mean) / std
    return x

def lumpiness(x):
    width = 10
    x = scale_ts(x)
    nr = len(x)
    lo = [i for i in range(0, nr, width)]
    up = [i for i in range(width, nr + width, width)]
    varx = [np.var(x[lo[idx]:up[idx]]) for idx in [i for i in range(0, int(nr / width))]]
    _lumpiness = np.var(varx)
    return _lumpiness

def stability(x):
    width = 10
    x = scale_ts(x)
    nr = len(x)
    lo = [i for i in range(0, nr, width)]
    up = [i for i in range(width, nr + width, width)]
    meanx = [np.mean(x[lo[idx]:up[idx]]) for idx in [i for i in range(0, int(nr / width))]]
    _stability = np.var(meanx)
    return _stability

def crossing_points(x):
    midline = np.median(x)
    ab = x <= midline
    lenx = len(x)
    p1 = ab[0:lenx-1]
    p2 = ab[1:lenx]
    p1 = np.array(p1)
    p2 = np.array(p2)
    not_p1 = np.logical_not(p1)
    not_p2 = np.logical_not(p2)
    a = np.logical_and(p1, not_p2)
    b = np.logical_and(p2, not_p1)
    crossingpoints = sum(np.logical_or(a, b))
    return crossingpoints
# print(entropy(ts['close']))
# print(entropy4(ts['close'], normalize=True))


def flat_spots(x):
    def run_lengths(lst):
        return max(sum(1 for _ in l) for n, l in itertools.groupby(lst))

    cutx = np.array(pd.cut(x, bins=10, labels=[i for i in range(10)]))
    flatspots = run_lengths(cutx)
    return flatspots

def kpss(x):
    return statsmodels.tsa.stattools.kpss(x)[0]

def pp(x):
    p = PhillipsPerron(x)
    return p._stat_rho + p._stat_tau

def hurst_coeff(x):
    h, C, data = hurst.compute_Hc(x)
    return h + 0.5

import pyper
from pyper import *


def arch_stat(x, lags=12, demean=True):
    if demean:
        x = x - np.mean(x)
    embed = r['embed']
    mat = embed(FloatVector(x ** 2), lags + 1)
    fmla = Formula('y ~ x')
    env = fmla.environment
    env['x'] = mat[:, 1:]
    env['y'] = mat[:, 0]
    fit = stats.lm(fmla)
    modsum = base.summary(fit)
    rsquared = modsum.rx2('r.squared')
    arch_lm = rsquared
    return arch_lm

def heterogeneity(x):
    x_archtest = arch_stat(x)
    arch_r2 = x_archtest[0]
    LBstat = sum(sm.tsa.stattools.acf(x**2, nlags=12, missing='none')[1:]**2)
    arch_acf = LBstat
    garch_fit = tseries.garch(x, trace=False)
    residuals = r['residuals']
    garch_fit_std = residuals(garch_fit)[1:]
    x_garch_archtest = arch_stat(garch_fit_std)
    garch_r2 = x_garch_archtest[0]
    LBstat2 = sum(sm.tsa.stattools.acf(garch_fit_std**2, nlags=12, missing='none')[1:]**2)
    garch_acf = LBstat2
    return arch_acf, garch_acf, arch_r2, garch_r2

def stl_features(x):
    import statsmodels.api as sm
    def poly(x, p):
        x = np.array(x)
        X = np.transpose(np.vstack((x ** k for k in range(p + 1))))
        return np.linalg.qr(X)[0][:, 1:]
    # dta = sm.datasets.co2.load_pandas().data
    # deal with missing values. see issue
    # dta.co2.interpolate(inplace=True)
    #t = pd.DataFrame(x)
    # print(x.close)
    trend, spike, linearity, curvature, e_acf1, e_acf10 = 0, 0, 0, 0, 0, 0
    msts = 0
    nperiods = 0
    stlfit = forecast.mstl(x)
    # stlfit = sm.tsa.seasonal_decompose(x, freq=1, model='additive')
    trend0 = stlfit[:,1]
    remainder = stlfit[:,2]
    detrend = x - trend0
    deseason = x
    fits = x - remainder
    n = len(x)
    varx = np.var(x)
    vare = np.var(remainder)
    vardetrend = np.var(detrend)
    vardeseason = np.var(deseason)

    if (vardeseason / varx < 1e-10):
        trend = 0
    else:
        trend = max(0, min(1,1 - vare / vardeseason))

    d = (remainder - np.mean(remainder))**2
    varloo = (vare * (n-1) - d) / (n - 2)
    spike = np.var(varloo)
    e_acf1, e_acf10, _ , _, _, _ = acf_features(pd.DataFrame(data=remainder.flatten()))

    fmla = Formula('y ~ x')
    env = fmla.environment
    #poly = r['poly']
    seq = r['seq']
    seqn = seq(n)
    env['x'] = poly(seqn, 2)
    env['y'] = trend0
    fit = stats.lm(fmla)
    coef = r['coef']
    fit = coef(fit)
    linearity = fit[0]
    curvature = fit[1]
    return trend, spike, e_acf1, e_acf10, linearity, curvature

print(hurst_coeff(ts['close']))