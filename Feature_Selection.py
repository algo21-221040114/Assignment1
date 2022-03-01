# Feature Selection for Random Forest
para_m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

# Intraday returns
def intraday_return(para_m, T):
    ir_list = []
    for t in range(241, T):
        for m in para_m:
            ir = (data.iloc[t-m, -1]/data.iloc[t-m, 0])-1
            ir_list.append(ir)
    return ir_list


# Returns with respect to last closing price
def ret_last_close(para_m, T):
    cr_list = []
    for t in range(241, T):
        for m in para_m:
            cr = (data.iloc[t-1, -1] / data.iloc[t-1-m, -1]) - 1
            cr_list.append(cr)
    return cr_list

# Returns with respect to opening price:
def ret_open(para_m, T):
    or_list = []
    for t in range(241, T):
        for m in para_m:
            open_r = (data.iloc[t, 0] / data.iloc[t-m, -1]) - 1
            or_list.append(open_r)
    return or_list

# Feature Selection for LSTM
def order():