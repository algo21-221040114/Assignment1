import pandas as pd
import numpy as np
from Feature_Selection import feature_lstm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from Strategy_Performance import strategy_performance
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only for Mac


def lstm(train_min_t, train_max_t, test_max_t, stock_num, node1, node2):
    """

    :param train_min_t: the beginning date of train
    :param train_max_t:  the end date of train
    :param test_max_t:  the end date of test
    :param stock_num:  the number of total stocks
    :param node1: output_size parameter
    :param node2: output_size parameter
    :return: the buy-sell portfolio average daily return (or cumulative return)
    """
    # Training set, Target
    target = pd.read_csv('./Target.csv', index_col='Date')
    y = []
    for t in range(train_min_t, train_max_t):
        target_series = target.iloc[t, :].values.T
        for i in range(len(target_series)):
            y.append([1-target_series[i], target_series[i]])
    y = np.array(y)

    # Training set, Feature
    x = np.zeros(((train_max_t-train_min_t)*stock_num, 240, 3))
    for t in range(train_min_t, train_max_t):
        for s in range(stock_num):
            x[(t-train_min_t)*stock_num+s] = np.array(feature_lstm(t, s))

    # LSTM model
    model = Sequential()
    model.add(LSTM(node1, input_shape=(240, 3)))
    model.add(Dense(node2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
    model.fit(x, y, batch_size=32, validation_split=0.2, callbacks=[callback])

    # Test set and Prediction
    buy_order_list = []
    sell_order_list = []
    for t in range(train_max_t, test_max_t):
        x_test = np.zeros((stock_num, 240, 3))
        for s in range(stock_num):
            x_test[s] = np.array(feature_lstm(t, s))
        prediction = model.predict(x_test)
        max_prob = prediction[0][1]
        min_prob = prediction[0][1]
        buy_order = 0
        sell_order = 0
        for i in range(1, len(prediction)):
            prob = prediction[i][1]
            if prob > max_prob:
                max_prob = prob
                buy_order = i
            elif prob < min_prob:
                min_prob = prob
                sell_order = i
        buy_order_list.append(buy_order)
        sell_order_list.append(sell_order)

    # Strategy Performance
    strategy_performance(buy_order_list, sell_order_list, train_max_t)



