import numpy as np
import pandas as pd
import pandas_datareader as web
import statsmodels.api as sm
from collections import OrderedDict
import streamlit as st

#credit to https://stackoverflow.com/users/3437787/vestland
def RegressionRoll(df, subset, dependent, independent, const, win, parameters):
    """
    RegressionRoll takes a dataframe, makes a subset of the data if you like,
    and runs a series of regressions with a specified window length, and
    returns a dataframe with BETA or R^2 for each window split of the data.

    Parameters:
    ===========

    df: pandas dataframe
    subset: integer - has to be smaller than the size of the df
    dependent: string that specifies name of denpendent variable
    inependent: LIST of strings that specifies name of indenpendent variables
    const: boolean - whether or not to include a constant term
    win: integer - window length of each model
    parameters: string that specifies which model parameters to return:
                BETA or R^2

    Example:
    ========
        RegressionRoll(df=df, subset = 50, dependent = 'X1', independent = ['X2'],
                   const = True, parameters = 'beta', win = 30)

    """

    # Data subset
    if subset != 0:
        df = df.tail(subset)
    else:
        df = df

    # Loopinfo
    end = df.shape[0]
    win = win
    rng = np.arange(start = win, stop = end, step = 1)

    # Subset and store dataframes
    frames = {}
    n = 1

    for i in rng:
        df_temp = df.iloc[:i].tail(win)
        newname = 'df' + str(n)
        frames.update({newname: df_temp})
        n += 1

    # Analysis on subsets
    df_results = pd.DataFrame()
    for frame in frames:
        #print(frames[frame])

        # Rolling data frames
        dfr = frames[frame]
        y = dependent
        x = independent

        if const == True:
            x = sm.add_constant(dfr[x])
            model = sm.OLS(dfr[y], x).fit()
        else:
            model = sm.OLS(dfr[y], dfr[x]).fit()

        if parameters == 'beta':
            theParams = model.params[0:]
            coefs = theParams.to_frame()
            df_temp = pd.DataFrame(coefs.T)

            indx = dfr.tail(1).index[-1]
            df_temp['Date'] = indx
            df_temp = df_temp.set_index(['Date'])

        if parameters == 'R2':
            theParams = model.rsquared
            df_temp = pd.DataFrame([theParams])
            indx = dfr.tail(1).index[-1]
            df_temp['Date'] = indx
            df_temp = df_temp.set_index(['Date'])
            df_temp.columns = [', '.join(independent)]
        df_results = pd.concat([df_results, df_temp], axis = 0)

    return(df_results)

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
        df_rolling = RegressionRoll(df=daily_returns, subset = 0, dependent = item, independent = ['USO'], const = False, parameters = 'beta',win = 60)
        df_rolling.reset_index()
        daily_returns['index'] = daily_returns.index
        daily_betas = pd.merge(daily_returns,df_rolling,how = 'inner',left_on='index',right_on='Date')
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
