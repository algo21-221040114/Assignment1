from sklearn.ensemble import RandomForestClassifier
from Feature_Selection import intraday_ret
from Feature_Selection import close_ret
from Feature_Selection import open_ret
from Strategy_Performance import strategy_performance
import pandas as pd


def random_forest(train_min_t, train_max_t, test_max_t, estimators, depth, features):
    """

    :param train_min_t: the beginning date of train
    :param train_max_t:  the end date of train
    :param test_max_t:  the end date of test
    :param estimators: n_estimators, parameter for Random Forest
    :param depth: max_depth, parameter for Random Forest
    :param features: max_feature, parameter for Random Forest
    :return: the buy-sell portfolio average daily return (or cumulative return)
    """
    # Training set, Feature
    for t in range(train_min_t, train_max_t):
        df1 = pd.DataFrame(intraday_ret(t).values.T)
        df2 = pd.DataFrame(close_ret(t).values.T)
        df3 = pd.DataFrame(open_ret(t).values.T)
        if t == train_min_t:
            x = pd.concat([df1, df2, df3], axis=1)
        else:
            df4 = pd.concat([df1, df2, df3], axis=1)
            x = x.append(df4)
    # the feature data is big, reserve it to save time to train model
    x.to_csv('x.csv', sep=',', header=True, index=True)
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
    clf = RandomForestClassifier(n_estimators=estimators, criterion='gini', max_depth=depth, max_features=features,
                                 min_samples_split=2, random_state=0)
    clf.fit(x, y.values.ravel())

    # Test set and Prediction
    buy_order_list = []
    sell_order_list = []
    for t in range(train_max_t, test_max_t):
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
    strategy_performance(buy_order_list, sell_order_list, train_max_t)










