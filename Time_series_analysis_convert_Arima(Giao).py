import pandas as pd
import matplotlib.pyplot as plt

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA

from math import sqrt

import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import seaborn as sns
from random import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error


df = pd.read_csv('QS(1).csv', encoding= 'unicode_escape')
df = df.set_index('year_dateformat')
ts = df['Quantity']
plt.figure(1)
plt.title("Quần Short (QS)")
ts.plot.line(figsize=(15, 6))

def test_stationarity(time_series):
  #Perform Dickey-Fuller test:
  print('Results of Dickey-Fuller Test:')
  adfuller_test = adfuller(time_series, autolag='AIC')
  adfuller_test_output = pd.Series(adfuller_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
  for key,value in adfuller_test[4].items():
      adfuller_test_output['Critical Value (%s)'%key] = value

  print(adfuller_test_output)

# test_stationarity(ts)

# First we penalize higher values by taking log
plt.figure(2)
ts_log = np.log(ts)
ts_log.plot(figsize=(15, 6))

# Take 1st differencing on transformed time series
plt.figure(3)
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True) # drop NA values
ts_log_diff.plot(figsize=(15, 6))

test_stationarity(ts_log_diff.dropna()) #.dropna(inplace=True))

# Take 2nd differencing on transformed time series - we get better result
plt.figure(4)
ts_log_diff = ts_log - ts_log.shift(2)
ts_log_diff.dropna(inplace=True)  # drop NA values
ts_log_diff.plot(figsize=(15, 6))

test_stationarity(ts_log_diff.dropna()) #.dropna(inplace=True))

# Using decomposition method to decompose time series
from pylab import rcParams
plt.figure(5)
rcParams['figure.figsize'] = 15, 6
#decomposition = sm.tsa.seasonal_decompose(ts_log, model = 'additive')
decomposition = sm.tsa.seasonal_decompose(ts_log, freq=12, model = 'additive')
decomposition.plot()
### Commented: ở đây decomposition ko có giá trị trả về. Tức là hàm seasonal_decompose ko trả ra kết quả gì cả
### nguyên nhân là do: gọi chuỗi dừng trong Arima, vì ts_log ko có tính dừng, nên khi gọi hàm này ra, kết quả ko có


# Build ARIMA model
arima_model = ARIMA(ts_log, order=(2, 1, 2))  
arima_model_fit = arima_model.fit(disp=-1) 

plt.figure(6)
plt.plot(ts_log_diff)
plt.plot(arima_model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f'% np.nansum((arima_model_fit.fittedvalues-ts_log_diff)**2))

# Read summary of ARIMA model
print(arima_model_fit.summary())


# Convert predicted values to original scale
predictions_ARIMA_diff = pd.Series(arima_model_fit.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())  # these are fitted values on the transformed data


# Cumulative sum to reverse differencing
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

# Adding 1st month value - was previously removed while differencing
predictions_ARIMA_log = pd.Series(ts_log[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
print(predictions_ARIMA_log.head())


# Take exponential to reverse Log Transform
predictions_ARIMA = np.exp(predictions_ARIMA_log)

# Compare with the original time series
plt.figure(7)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.xticks(rotation=90, fontsize=6)


# Auto ARIMA
# Divide into train and validation set
plt.figure(8)
train = ts[:int(0.75*(len(ts)))]
valid = ts[int(0.75*(len(ts))):]
#train.plot()
#valid.plot()
plt.plot(train)
plt.plot(valid)
plt.xticks(rotation=90, fontsize=6)

plt.show()