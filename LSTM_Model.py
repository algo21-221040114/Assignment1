import pandas as pd
import numpy as np
import pickle
from Feature_Selection import feature_lstm
from keras.models import Sequential
from keras.layers import Dense, Embedding
# from keras.layers import CuDNNLSTM as LSTM
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only for Mac


# Training set, Target
max_t = 242
target = pd.read_csv('./Target.csv', index_col='Date')
y = []
for t in range(241, max_t):
    target_series = target.iloc[t, :].values.T
    for i in range(len(target_series)):
        y.append([1-target_series[i], target_series[i]])
y = np.array(y)
print('over1')

# Training set, Feature
stock_num = 10
x = np.zeros(((max_t-241)*stock_num, 240, 3))
for t in range(241, max_t):
    for s in range(stock_num):
        x[(t-241)*10+s] = np.array(feature_lstm(t, s))
print('over2')

# LSTM model
model = Sequential()
model.add(LSTM(16, input_shape=(240, 3)))  # dropout=0.1, recurrent_dropout=0.1
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
callback = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
model.fit(x, y, batch_size=32, validation_split=0.2, callbacks=[callback])
print('over4')

# Test set and Prediction
t_test = 757
x_test = np.zeros((stock_num, 240, 3))
for s in range(stock_num):
    x_test[s] = np.array(feature_lstm(t_test, s))
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
print(buy_order, sell_order)



