# Assignment1

## Introduction

This assignment refer to a paper, 'Forecasting directional movements of stock prices for intraday trading using LSTM
and random forests', written by Pushpendu Ghosh, Ariel Neufeld, Jajati Keshari Sahoo. 
You can access to the paper through https://paperswithcode.com/paper/forecasting-directional-movements-of-stock.
The main proposal is to predict the up or down direction of stock prices, with Random Forest and LSTM.

## Environment

Pycharm (Professional Edition)
Python 3.7

## Requirements

Numpy
Pandas
Keras
Sklearn

## Model Elaboration

Data used in the model is the stock open price and close price, obtained from the open database, Yahoo Finance. 
Due to equipment limit, this assignment is based on ten stocks from 2016.01.01 to 2019.12.31.
The first three years are for feature selection and model training; the forth year is for testing.
After data preprocessing and feature selection, you can separately run the two models, Random_Forest.py and LSTM.py.
Models will classify the stock into down-trend or up-trend, 0 or 1.

If you have GPU or Cloud Sever, you can enlarge the stock basket, extend time period and increase the training and 
testing rolling windows,which may contribute to better model. With bigger dataset, you should change the parameters 
including stock_num, stock_basket, train_min_t, train_max_t and test_max_t. Also, when you get GPU, CuDNNLSTM will be 
available for you to enhance model efficiency.

## Conclusion

The Random Forest model performs better, providing average daily return about 6.69% in the test data, compared with -4.17%, 
the average daily return in 2019 of all stocks. However, the LSTM model always divide stocks into two classes with the probability around 50%. One reason may be that 
after Robust Standardisation, the 3-dimension features have little difference, and after 240 iterations, the difference 
may be eliminated again. New features should be consideration for the LSTM model. Also, with more rolling training periods 
supported with GPU, the LSTm may perform better.

