import datetime
from Data_Preprocess import get_data
from Target_Decision import target_decision
from Random_Forest_Model import random_forest
from LSTM_Model import lstm

# Define the instruments to download.  randomly selected from S&P500
# Define dataset time period
stock_basket = ['AAPL', 'FLS', 'GME', 'HRL', 'HSY', 'MSFT', 'SPG', 'SUN', 'TER', 'X']
start_date = datetime.datetime(2016, 1, 4)
end_date = datetime.datetime(2019, 12, 31)

# General Parameters
train_min_t = 241  # the maximum lag is 240
train_max_t = 754  # the first three years are for feature selection
test_max_t = 1006  # the forth year is for test
stock_num = 10  # total amount of stocks

# Random Forest Model Parameters
n_est = 100
max_dep = 4
max_fea = 9

# LSTM Parameters
node1 = 16  # LSTM parameter
node2 = 2  # LSTM parameter


if __name__ == "__main__":
    get_data(stock_basket, start_date, end_date)
    target_decision()
    random_forest(train_min_t, train_max_t, test_max_t, n_est, max_dep, max_fea)
    lstm(train_min_t, train_max_t, test_max_t, stock_num, node1, node2)
