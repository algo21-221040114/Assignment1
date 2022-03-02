import numpy as np
import pandas as pd

data_open = pd.read_csv('./Open_price.csv', index_col='Date')
data_close = pd.read_csv('./Adj_close_price.csv', index_col='Date')
data_target = data_open.copy(deep=True)
for i in range(data_open.shape[0]):
    ir_list = []
    for j in range(data_open.shape[1]):
        ir_list.append(data_close.iloc[i, j]/data_open.iloc[i, j]-1)
    for j in range(data_open.shape[1]):
        if ir_list[j] >= np.median(ir_list):
            data_target.iloc[i, j] = 1
        else:
            data_target.iloc[i, j] = 0
data_target.to_csv('Target.csv', sep=',', header=True, index=True)