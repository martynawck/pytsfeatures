import pandas as pd
import statsmodels.api as sm

ts = pd.read_csv("/home/martyna/sharedData/test2.csv")
ts.reset_index(inplace=True)
ts['barTimestamp'] = pd.to_datetime(ts['barTimestamp'])
ts = ts.set_index('barTimestamp')
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

def stl_features(x):
    import statsmodels.api as sm

    # dta = sm.datasets.co2.load_pandas().data
    # deal with missing values. see issue
    # dta.co2.interpolate(inplace=True)
    #t = pd.DataFrame(x)
    # print(x.close)
    result = sm.tsa.seasonal_decompose(x.close, freq=1, model='additive')
    print('trend')
    print(result.trend)
    print(result.seasonal)
    # print(result.resid)
    # print(result.observed)

print(stl_features(ts))