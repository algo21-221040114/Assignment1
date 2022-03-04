from sklearn.ensemble import RandomForestClassifier
from Feature_Selection import intraday_ret
from Feature_Selection import close_ret
from Feature_Selection import open_ret
import pandas as pd

# Parameters
train_min_t = 241  # the maximum lag is 240
train_max_t = 754  # the first three years are for feature selection

# Training set, Feature
# for t in range(train_min_t, train_max_t):
#     df1 = pd.DataFrame(intraday_ret(t).values.T)
#     df2 = pd.DataFrame(close_ret(t).values.T)
#     df3 = pd.DataFrame(open_ret(t).values.T)
#     if t == train_min_t:
#         x = pd.concat([df1, df2, df3], axis=1)
#     else:
#         df4 = pd.concat([df1, df2, df3], axis=1)
#         x = x.append(df4)
# # the feature data is big, reserve it to save time
# x.to_csv('x.csv', sep=',', header=True, index=True)
x = pd.read_csv('./x.csv')
x = x.iloc[:, 1:]

# Training set, Target
target = pd.read_csv('./Target.csv', index_col='Date')
for t in range(train_min_t, train_max_t):
    if t == train_min_t:
        y = pd.DataFrame(target.iloc[t, :].values.T)
    else:
        df1 = pd.DataFrame(target.iloc[t, :].values.T)
        y = y.append(df1)

# Random Forest Model
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4, max_features=9,
                             min_samples_split=2, random_state=0)
clf.fit(x, y.values.ravel())

# Test set and Prediction
test_max_t = 1006  # the forth year is for test
buy_order_list = []
sell_order_list = []
for t in range(754, test_max_t):
    df1 = pd.DataFrame(intraday_ret(t).values.T)
    df2 = pd.DataFrame(close_ret(t).values.T)
    df3 = pd.DataFrame(open_ret(t).values.T)
    x_test = pd.concat([df1, df2, df3], axis=1)
    prediction = clf.predict_proba(x_test)
    max_prob = prediction[0][1]
    min_prob = prediction[0][1]
    buy_order = 0
    sell_order = 0
    for j in range(1, len(prediction)):
        prob = prediction[j][1]
        if prob > max_prob:
            max_prob = prob
            buy_order = j
        elif prob < min_prob:
            min_prob = prob
            sell_order = j
    buy_order_list.append(buy_order)
    sell_order_list.append(sell_order)

# Strategy Performance
data_open = pd.read_csv('./Open_price.csv', index_col='Date')
data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
portfolio_ret = []
cum_ret = 1
daily_ret = 0
avg_daily_ret = 0
for i in range(len(buy_order_list)):
    # if the stock number in buy_order, buy with open price and sell it with close price
    up_ret = data_close.iloc[train_max_t+i, buy_order_list[i]]/data_open.iloc[train_max_t+i, buy_order_list[i]]-1
    # if the stock number in sell_order, sell with open price and buy it with close price
    down_ret = (data_open.iloc[train_max_t+i, sell_order_list[i]]/data_close.iloc[train_max_t+i, sell_order_list[i]])-1
    daily_ret = (up_ret+down_ret)/2
    avg_daily_ret += daily_ret
    # cum_ret = cum_ret*(1+daily_ret)
    portfolio_ret.append(daily_ret)
avg_daily_ret = avg_daily_ret/len(buy_order_list)
print(avg_daily_ret)
# print(cum_ret)
# print(len(portfolio_ret))









