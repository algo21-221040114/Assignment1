# -*- coding: utf-8 -*-
import pandas as pd
import tushare as ts
from datetime import datetime

# Tushare API init
my_token = 'fce2186abb3aa71fa472c6762b0f79244960ab004cf97ef77352ec92'
ts.set_token(my_token)
pro = ts.pro_api()


def get_data(stock_basket, back_test_start_date, back_test_end_date):
    """

    :param stock_basket: stock basket
    :param back_test_start_date: back test starting date
    :param back_test_end_date: back test ending date
    :return: data (DataFrame)
    """
    for i in range(len(stock_basket)):
        df = pro.daily(ts_code=stock_basket[i], start_date=back_test_start_date, end_date=back_test_end_date,
                       fields='ts_code, trade_date, close')
        if i != 0:
            pass
        else:
            time_period = []
            for j in range(len(df)):
                time_period.append(df.iloc[j, 1])  # trade date
            data = pd.DataFrame(columns=stock_basket, index=time_period)
        for k in range(len(time_period)):
            data.iloc[k, i] = df.iloc[k, 2]  # close price
    return data



def st_time(stock_code, back_test_start_date, back_test_end_date):
    """

    :param stock_code: stock code
    :param back_test_start_date: back test starting date
    :param back_test_end_date: back test ending date
    :return:
    """
    back_test_start = datetime.strptime(back_test_start_date, '%Y%m%d')
    back_test_end = datetime.strptime(back_test_end_date, '%Y%m%d')
    df = pro.namechange(ts_code=stock_code, fields='ts_code, name, start_date, end_date')
    st = df[df.name.str.contains('ST')]
    print(st)
    for i in range(len(st)):
        st_start = datetime.strptime(st.iloc[i, 2], '%Y%m%d')
        if st.iloc[i, 3] is None:
            st_end = datetime.now()
        else:
            st_end = datetime.strptime(st.iloc[i, 3], '%Y%m%d')
        if st_start <= back_test_end and st_end >= back_test_start:
            if st_start <= back_test_start:
                st_start = back_test_start
            if st_end >= back_test_end:
                st_end = back_test_end
            print(st_start, "|", st_end)


# if __name__ == "__main__":
    # get_data(['000001.SZ', '600000.SH'], '20160101', '20211231')
    # st_time('000005.SZ', '20160101', '20211231')
