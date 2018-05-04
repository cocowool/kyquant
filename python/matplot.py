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