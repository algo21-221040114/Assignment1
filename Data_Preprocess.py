import pandas as pd
from pandas_datareader import data
import datetime


def get_data(stock_basket, start_date, end_date):
    """

    :param stock_basket: stocks name list
    :param start_date: starting time
    :param end_date: ending time
    :return:
    """
    # Build empty data set
    data_open = pd.DataFrame()
    data_close = pd.DataFrame()

    # Use pandas_reader.data.DataReader to load the desired data
    for i in range(len(stock_basket)):
        df = data.DataReader(stock_basket[i], 'yahoo', start_date, end_date)
        data_open[stock_basket[i]] = pd.DataFrame(df['Open'])
        data_close[stock_basket[i]] = pd.DataFrame(df['Adj Close'])
    # print(data_open.shape[0])
    # print(len(data_open['2019']))

    # Save data
    data_open.to_csv('Open_price.csv', sep=',', header=True, index=True)
    data_close.to_csv('Adj_close_price.csv', sep=',', header=True, index=True)









