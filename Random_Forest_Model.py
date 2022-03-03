from sklearn.ensemble import RandomForestClassifier
from Feature_Selection import intraday_ret
from Feature_Selection import close_ret
from Feature_Selection import open_ret
import pandas as pd

# Training set, Target
target = pd.read_csv('./Target.csv', index_col='Date')
for t in range(241, 754):
    if t == 241:
        y = pd.DataFrame(target.iloc[t, :].values.T)
    else:
        df1 = pd.DataFrame(target.iloc[t, :].values.T)
        y = y.append(df1)
print('over1')

# Training set, Feature
# for t in range(241, 754):
#     df1 = pd.DataFrame(intraday_ret(t).values.T)
#     df2 = pd.DataFrame(close_ret(t).values.T)
#     df3 = pd.DataFrame(open_ret(t).values.T)
#     if t == 241:
#         x = pd.concat([df1, df2, df3], axis=1)
#     else:
#         df4 = pd.concat([df1, df2, df3], axis=1)
#         x = x.append(df4)
# x.to_csv('x.csv', sep=',', header=True, index=True)
x = pd.read_csv('./x.csv')
x = x.iloc[:, 1:]
print('over2')

# Random Forest Model
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, max_features=9,
                             min_samples_split=2, random_state=0)
clf.fit(x, y.values.ravel())
# print(clf.feature_importances_)

# Test set and Prediction
for t in range(754, 755):
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
    print(buy_order, sell_order)







