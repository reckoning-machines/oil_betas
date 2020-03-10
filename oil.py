import numpy as np
import pandas as pd
import pandas_datareader as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from collections import OrderedDict
import streamlit as st
"""
To run from command line, install streamlit, then type:
streamlit run oil.py
"""


tickers = ["USO",
           'XOM',
           "BOKF",
          "CADE",
          "CFR",
          "IBTX",
          "FHN",
          "CMA",
          "EWBC",
          "ZION"]

stocks = web.get_data_yahoo(tickers,
start = "2015-01-01",
end = "2020-03-06")
daily_returns = stocks['Adj Close'].pct_change().reset_index()
daily_returns = daily_returns.dropna()

df_out = pd.DataFrame()
for item in daily_returns.columns:
    if item is not "Date" and item is not "USO" and item is not "index":
        endog = daily_returns[item]
        exog = sm.add_constant(daily_returns['USO'])
        rols = RollingOLS(endog, exog, window=60)
        rres = rols.fit()
        df_rolling = rres.params
        daily_returns['index'] = daily_returns.index
        df_rolling['index'] = df_rolling.index
        daily_betas = pd.merge(daily_returns,df_rolling,how = 'inner',left_on='index',right_on='index')
        daily_betas[item] = daily_betas['USO_y']
        daily_betas = daily_betas[['Date',item]]
        if len(df_out) == 0:
            df_out = daily_betas
        else:
            df_out = pd.merge(df_out,daily_betas,how='inner',left_on='Date',right_on='Date')

list_out = []
df_stats = pd.DataFrame()
for col in df_out.columns:
    if "Date" not in col:
        stock_return = daily_returns[col].tail(1).iloc[0]
        index_return = daily_returns['USO'].tail(1).iloc[0]
        stock_beta = df_out[col].tail(1).iloc[0]
        alpha = float(stock_return-index_return*stock_beta)
        data = [['Current 60 day beta vs USO',df_out[col].tail(1).iloc[0]],
                ['mean',df_out[col].mean()],
               ['std',df_out[col].std()],
                ['actual 1 day return',stock_return],
                ['estimated 1 day return',index_return*stock_beta],
               ['alpha',alpha]]
        r = pd.DataFrame(data,columns=['stat',col])
        list_out.append(r)
for r in list_out:
    if len(df_stats) == 0:
        df_stats = r
    else:
        df_stats = pd.merge(df_stats,r,how='inner',left_on = 'stat',right_on='stat')

import matplotlib.pyplot as plt
#%matplotlib inline
x_data = df_out['Date']
fig, ax = plt.subplots()
for column in df_out.columns:
    if "Date" not in column:
        ax.plot(x_data, df_out[column],label=column)
ax.set_title('Rolling Betas Vs OIL ETF ')
ax.legend()
st.write(fig)
st.write(df_stats)
