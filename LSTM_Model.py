import pandas as pd
import numpy as np
from Feature_Selection import feature_lstm
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import CuDNNLSTM as LSTM
from keras.callbacks import EarlyStopping

# LSTM model
model = Sequential()
model.add(Embedding(240, 3))
model.add(LSTM(16))  # dropout=0.1, recurrent_dropout=0.1
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training set, Target
# target = pd.read_csv('./Target.csv', index_col='Date')
# for t in range(241, 754)
#     if t == 241:
#         y = pd.DataFrame(target.iloc[t, :].values.T)
#     else:
#         df1 = pd.DataFrame(target.iloc[t, :].values.T)
#         y = y.append(df1)
# print('over1')

# Training set, Feature
# max_t = 242
# stock_num = 10
# x = np.zeros(((max_t-241)*stock_num, 240, 3))
# for t in range(241, max_t):
#     for s in range(stock_num):
#         x[(t-241)*10+s] = np.array(feature_lstm(t, s))
# print(x)
# print('over2')
# x = np.zeros((2, 4, 3))
# x[0][1] = np.array([[8, 9, 10]])
# print(x)
# callback = EarlyStopping(monitor='val_acc', patience=10, mode='max', restore_best_weights=True)
# model.fit(x, y, batch_size=32, callbacks=[callback])
