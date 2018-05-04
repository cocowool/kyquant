# 可视化的内容

import pandas as pd
import numpy as np
from abupy import ABuSymbolPd
import matplotlib.pyplot as plt

tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds = 2)
tsla_df.tail()

def plot_demo(axs = None, just_series=False):
	drawer = plt if axs is None else axs
	drawer.plot(tsla_df.close, c='r')
	if not just_series:
		drawer.plot(tsla_df.close.index, tsla_df.close.values + 10, c='g')

		drawer.plot(tsla_df.close.index.tolist(), (tsla_df.close.values+20).tolist(), c='b')
	plt.xlabel('time')
	plt.ylabel('close')
	plt.title('TSLA CLOSE')
	plt.grid(True)

plot_demo()

# 子画布以及 loc 的指定
_, axs = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
# 画布0, loc 0
drawer = axs[0][0]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], loc=0)

# 画布1, loc 1
drawer = axs[0][1]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], loc=1)

# 画布2, loc 2
drawer = axs[1][0]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], loc=2)

# 画布3, loc 2
drawer = axs[1][1]
plot_demo(drawer)
drawer.legend(['Series', 'Numpy', 'List'], bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)

# 绘制K线图，通过 matplotlib.finance.candlestick_ochl 以及 matplotlib.finance.candlestick2_ohlc 函数
import matplotlib.finance as mpf
__colorup__ = "red"
__colordown__ = "green"
tsla_part_df = tsla_df[:30]

fig, ax = plt.subplots(figsize=(14,7))
quotes = []
for index, (d,o,c,h,l) in enumerate(
	zip(tsla_part_df.index, tsla_part_df.open, tsla_part_df.close, tsla_part_df.high, tsla_part_df.low)):
	d = mpf.date2num(d)
	val = (d,o,c,h,l)
	quotes.append(val)
mpf.candlestick_ochl(ax,quotes,width=0.6,colorup=__colorup__,colordown=__colordown__)
ax.autoscale_view()
ax.xaxis_date()

# 引用 Bokeh 支持交互可视化
from abupy import ABuMarketDrawing
ABuMarketDrawing.plot_candle_form_klpd(tsla_df,html_bk=True)

# 使用 Pandas 可视化数据
demo_list = np.array([2,4,16,20])
demo_window = 3
pd.rolling_std(demo_list, window=demo_window, center=False)*np.sqrt(demo_window)

tsla_df_copy = tsla_df.copy()
# 计算投资回报
tsla_df_copy['return'] = np.log(tsla_df['close'] / tsla_df['close'].shift(1))
# 移动收益标准差
tsla_df_copy['mov_std'] = pd.rolling_std(tsla_df_copy['return'], window=20, center=False) * np.sqrt(20)
# 加权移动收益标准差
tsla_df_copy['std_ewm'] = pd.ewmstd(tsla_df_copy['return'], span=20, min_periods=20, adjust=True) * np.sqrt(20)

tsla_df_copy[['close', 'mov_std', 'std_ewm', 'return']].plot(subplots=True, grid=True)

# 绘制均线
tsla_df.close.plot()
# ma 30
pd.rolling_mean(tsla_df.close, window=30).plot()
# ma 60
pd.rolling_mean(tsla_df.close, window=60).plot()
# ma 90
pd.rolling_mean(tsla_df.close, window=90).plot()
plt.legend(['close', '30 mv', '60 mv', '90 mv'], loc='best')

# 验证低开高走第二天趋势


# searborn
import seaborn as sns
sns.distplot(tsla_df['p_change'], bins=80)