import pandas as pd
import numpy as np
import pickle
from Feature_Selection import feature_lstm
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import CuDNNLSTM as LSTM
from keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only for Mac


# LSTM model
model = Sequential()
model.add(Embedding(240, 3))
model.add(LSTM(16))  # dropout=0.1, recurrent_dropout=0.1
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Training set, Target
max_t = 242
target = pd.read_csv('./Target.csv', index_col='Date')
for t in range(241, max_t):
    if t == 241:
        y = pd.DataFrame(target.iloc[t, :].values.T)
    else:
        df1 = pd.DataFrame(target.iloc[t, :].values.T)
        y = y.append(df1)
print('over1')

# Training set, Feature
stock_num = 10
x = np.zeros(((max_t-241)*stock_num, 240, 3))
for t in range(241, max_t):
    for s in range(stock_num):
        x[(t-241)*10+s] = np.array(feature_lstm(t, s))
print('over2')

data_output = open('feature_lstm.pkl', 'wb')
pickle.dump(x, data_output)
data_output.close()

data_input = open('feature_lstm.pkl', 'rb')
x_lstm = pickle.load(data_input)
data_input.close()
print('over3')

callback = EarlyStopping(monitor='val_acc', patience=10, mode='max', restore_best_weights=True)
model.fit(x_lstm, y, batch_size=32, callbacks=[callback])
print('over4')
t_test = 757
x_test = np.zeros((stock_num, 240, 3))
for s in range(stock_num):
    x_test[s] = np.array(feature_lstm(t_test, s))
y_prob = model.predict(x_test)
y_class = y_prob.argmax(axis=-1)
print(y_prob, y_class)

