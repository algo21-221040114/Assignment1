import pandas as pd
from pandas_datareader import data
import datetime

# Define the instruments to download.  S&P500
stock_basket = ['AAPL', 'FLS', 'GME', 'HRL', 'HSY', 'MSFT', 'SPG', 'SPY', 'TER', 'X']
start_date = datetime.datetime(2016, 1, 4)
end_date = datetime.datetime(2021, 12, 31)

# Build empty data set
data_open = pd.DataFrame()
data_close = pd.DataFrame()

# Use pandas_reader.data.DataReader to load the desired data
for i in range(len(stock_basket)):
    df = data.DataReader(stock_basket[i], 'yahoo', start_date, end_date)
    data_open[stock_basket[i]] = pd.DataFrame(df['Open'])
    data_close[stock_basket[i]] = pd.DataFrame(df['Adj Close'])

# Save data
data_open.to_csv('Open_price.csv', sep=',', header=True, index=True)
data_close.to_csv('Adj_close_price.csv', sep=',', header=True, index=True)





