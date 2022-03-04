import pandas as pd
import numpy as np


# Feature selection for Random Forest

# Parameter of lag of features
para_m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]


def intraday_ret(t):
    """

    :param t: the date
    :return: Feature1: the intraday returns of the date
    """
    data_open = pd.read_csv('./Open_price.csv', index_col='Date')
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    data_intraday_ret = pd.DataFrame(columns=data_open.columns, index=para_m)
    for i in range(data_intraday_ret.shape[1]):
        for j in range(data_intraday_ret.shape[0]):
            data_intraday_ret.iloc[j, i] = (data_close.iloc[t-para_m[j]-1, i] / data_open.iloc[t-para_m[j]-1, i]) - 1
    return data_intraday_ret


def close_ret(t):
    """

    :param t: the date
    :return: Feature2: returns with respect to last closing price of the date
    """
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    data_cr_ret = pd.DataFrame(columns=data_close.columns, index=para_m)
    for i in range(data_cr_ret.shape[1]):
        for j in range(data_cr_ret.shape[0]):
            data_cr_ret.iloc[j, i] = (data_close.iloc[t-2, i] / data_close.iloc[t-para_m[j]-2, i]) - 1
    return data_cr_ret


def open_ret(t):
    """

    :param t: the date
    :return: Feature3: returns with respect to open price of the date
    """
    data_open = pd.read_csv('./Open_price.csv', index_col='Date')
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    data_or_ret = pd.DataFrame(columns=data_open.columns, index=para_m)
    for i in range(data_or_ret.shape[1]):
        for j in range(data_or_ret.shape[0]):
            data_or_ret.iloc[j, i] = (data_open.iloc[t-1, i] / data_close.iloc[t-para_m[j]-1, i]) - 1
    return data_or_ret


# Feature selection for LSTM
def feature_lstm(t, stock_num):
    """

    :param t: the date
    :param stock_num: the sequential number of the stock in the stock basket
    :return: Feature4: an array(240, 3) with returns
    """
    data_open = pd.read_csv('./Open_price.csv', index_col='Date')
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    f_lstm = []
    # para_m = 1
    for i in range(240):
        f = [data_close.iloc[t - i - 2, stock_num] / data_open.iloc[t - i - 2, stock_num] - 1,
             data_close.iloc[t - i - 2, stock_num] / data_close.iloc[t - i - 3, stock_num] - 1,
             data_open.iloc[t - i - 1, stock_num] / data_close.iloc[t - i - 2, stock_num] - 1]
        # Robust Scaler standardization
        a = np.median(f)
        b = np.max(f)
        c = np.min(f)
        for j in range(len(f)):
            f[j] = (f[j]-a)/(b-c)
        f_lstm.append(f)
    return f_lstm



