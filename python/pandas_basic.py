# Pandas 基础

import pandas as pd
import numpy as np

stock_day_change = np.load('./stock_day_change.npy')
stock_day_change.shape

pd.DataFrame(stock_day_change).head()
pd.DataFrame(stock_day_change).head(10)
pd.DataFrame(stock_day_change)[:5]

stock_symbols = ['股票' + str(x) for x in range(stock_day_change.shape[0])]

pd.DataFrame(stock_day_change, index=stock_symbols).head(2)

days = pd.date_range('2017-1-1', periods=stock_day_change.shape[1], freq='1d')

df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
df.head(2)

# 进行数据转置
df = df .T
df.head()

# 这个用法不推荐了
df_20 = df.resample('21D',how='mean')
df_20.head()

df_20 = df.resample('21D').mean()
df_20.head()

df_stock0 = df['股票0']
print(type(df_stock0))
df_stock0.head()

df_stock0.cumsum().plot()


# 使用 ohlc 方法重采样 open, high, low, close
# 周线
df_stock0_5 = df_stock0.cumsum().resample('5D').ohlc()
# 月线
df_stock0_20 = df_stock0.cumsum().resample('21D').ohlc()
df_stock0_5.head()

from abupy import ABuMarketDrawing


ABuMarketDrawing.plot_candle_stick(df_stock0_5.index, df_stock0_5['open'].values, df_stock0_5['high'].values, df_stock0_5['low'].values, df_stock0_5['close'].values, np.random.random(len(df_stock0_5)), None, 'stock', day_sum = False, html_bk = False, save = False)

print(type(df_stock0_5['open'].values))
print(df_stock0_5['open'].index)
print(df_stock0_5.columns)
