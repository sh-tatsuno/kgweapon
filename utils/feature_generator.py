import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gc

# add time features
def add_time(df, time_col, params):
    df[time_col]= pd.to_datetime(df[time_col])
    dt= df[time_col].dt
    for col in params:
        if col == 'year':
            df[col] = eval("dt."+col+".astype('uint16')")
        else:
            df[col] = eval("dt."+col+".astype('uint8')")
    del(dt)
    return

# add count feature
def add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+'_count'] = counts[unqtags]
    del(unq); del(unqtags); del(counts);
    return

# add agg features
def add_agg(df, groupby_cols, target, metric, postfix=''):
    name = target + '_' + '_'.join(groupby_cols) + '_' + postfix
    gb = df.groupby(groupby_cols, as_index=False)[target].agg(metric).reset_index(drop=True)
    gb.columns = [col if col != gb.columns.values[-1] else name for col in gb.columns.values]
    df_ = pd.merge(df, gb, how='left', on=groupby_cols)
    df[name] = df_[name]
    return

def add_mean_enc(df, col, target):
    cumsum = df.groupby(col)[target].cumsum() - df[target]
    cumcnt = df.groupby(col)[target].cumcount()
    df[col + '_target_enc'] = cumsum /cumcnt
    df[col + '_target_enc'].fillna(df[col + '_target_enc'].mean(), inplace=True)
    return

def add_month_timeblock(df, timestamp):
    base = df[timestamp].min()
    base = base.year *12 + base.month # minimum time block
    df["timeblock"] = df[timestamp].apply(lambda x: x.year *12 + x.month - base)
    return

def add_shift(df, index_cols, value_cols, shift):
    df_shift = df[index_cols + value_cols].copy()
    df_shift.timeblock = df_shift.timeblock + shift
    value_cols_rename = [col + "_shift_" + shift for col in value_cols]
    df_shift.columns = index_cols +  value_cols_rename
    df = pd.merge(df, df_shift, on =index_cols, how = "left")
    return
