import pandas as pd
import numpy as np

para_m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

# Feature selection for random forest


# Intraday returns
def intraday_ret(t):
    data_open = pd.read_csv('./Open_price.csv', index_col='Date')
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    data_intraday_ret = pd.DataFrame(columns=data_open.columns, index=para_m)
    for i in range(data_intraday_ret.shape[1]):
        for j in range(data_intraday_ret.shape[0]):
            data_intraday_ret.iloc[j, i] = (data_close.iloc[t-para_m[j]-1, i] / data_open.iloc[t-para_m[j]-1, i]) - 1
    return data_intraday_ret


# Returns with respect to last closing price
def close_ret(t):
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    data_cr_ret = pd.DataFrame(columns=data_close.columns, index=para_m)
    for i in range(data_cr_ret.shape[1]):
        for j in range(data_cr_ret.shape[0]):
            data_cr_ret.iloc[j, i] = (data_close.iloc[t-2, i] / data_close.iloc[t-para_m[j]-2, i]) - 1
    return data_cr_ret


# Returns with respect to opening price:
def open_ret(t):
    data_open = pd.read_csv('./Open_price.csv', index_col='Date')
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    data_or_ret = pd.DataFrame(columns=data_open.columns, index=para_m)
    for i in range(data_or_ret.shape[1]):
        for j in range(data_or_ret.shape[0]):
            data_or_ret.iloc[j, i] = (data_open.iloc[t-1, i] / data_close.iloc[t-para_m[j]-1, i]) - 1
    return data_or_ret


# Feature selection for LSTM
def feature_lstm(t, stock_num):
    f_lstm = []
    for i in range(240):
        # m = 1
        f = [intraday_ret(t-i).iloc[0, stock_num], close_ret(t-i).iloc[0, stock_num], open_ret(t-i).iloc[0, stock_num]]
        # Robust Scaler standardization
        for j in range(len(f)):
            f[j] = (f[j]-np.median(f))/(np.max(f)-np.min(f))
        f_lstm.append(f)
    return f_lstm



