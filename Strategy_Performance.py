import pandas as pd


def strategy_performance(buy_order_list, sell_order_list, max_t):
    """

    :param buy_order_list: the stock number to buy
    :param sell_order_list:  the stock number to sell
    :param max_t: the test starting date
    :return: the buy-sell portfolio average daily return (or cumulative return)
    """
    data_open = pd.read_csv('./Open_price.csv', index_col='Date')
    data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
    portfolio_ret = []
    cum_ret = 1
    daily_ret = 0
    avg_daily_ret = 0
    for i in range(len(buy_order_list)):
        # if the stock number in buy_order, buy with open price and sell it with close price
        up_ret = data_close.iloc[max_t+i, buy_order_list[i]]/data_open.iloc[max_t+i, buy_order_list[i]]-1
        # if the stock number in sell_order, sell with open price and buy it with close price
        down_ret = (data_open.iloc[max_t+i, sell_order_list[i]]/data_close.iloc[max_t+i, sell_order_list[i]])-1
        daily_ret = (up_ret+down_ret)/2
        avg_daily_ret += daily_ret
        # cum_ret = cum_ret*(1+daily_ret)
        portfolio_ret.append(daily_ret)
    avg_daily_ret = avg_daily_ret/len(buy_order_list)
    print(avg_daily_ret)
    # print(cum_ret)