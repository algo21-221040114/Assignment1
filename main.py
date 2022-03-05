import datetime
from Data_Preprocess import get_data
from Target_Decision import target_decision
from Random_Forest_Model import random_forest


# Define the instruments to download.  randomly selected from S&P500
# Define dataset time period
stock_basket = ['AAPL', 'FLS', 'GME', 'HRL', 'HSY', 'MSFT', 'SPG', 'SPY', 'TER', 'X']
start_date = datetime.datetime(2016, 1, 4)
end_date = datetime.datetime(2019, 12, 31)

# Time Parameters
train_min_t = 241  # the maximum lag is 240
train_max_t = 754  # the first three years are for feature selection
test_max_t = 1006  # the forth year is for test

# Random Forest Model Parameters
n_est = 100
max_dep = 4
max_fea = 9

if __name__ == "__main__":
    get_data(stock_basket, start_date, end_date)
    target_decision()
    random_forest(train_min_t, train_max_t, test_max_t, n_est, max_dep, max_fea)
