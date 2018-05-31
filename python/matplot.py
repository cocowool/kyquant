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
low_to_high_df = tsla_df.iloc[tsla_df[(tsla_df.close > tsla_df.open) & ( tsla_df.key != tsla_df.shape[0] -1)].key.values + 1]
change_ceil_floor = np.where(low_to_high_df['p_change'] > 0, np.ceil(low_to_high_df['p_change']), np.floor(low_to_high_df['p_change']))
change_ceil_floor = pd.Series(change_ceil_floor)
print('低开高收下一个交易日所有下跌的跌幅取整数和 sum:' + str(change_ceil_floor[change_ceil_floor < 0].sum()))
print("低开高收下一个交易日所有上涨的涨幅取整数和 sum:" + str(change_ceil_floor[change_ceil_floor > 0].sum()))
# 2*2 4张子图
_, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
# 柱图
change_ceil_floor.value_counts().plot(kind='bar', ax=axs[0][0])
# 水平柱状图
change_ceil_floor.value_counts().plot(kind='barh', ax=axs[0][1])
# 概率密度图
change_ceil_floor.value_counts().plot(kind='kde', ax=axs[1][0])
# 饼图
change_ceil_floor.value_counts().plot(kind='pie', ax=axs[1][1])


# searborn
import seaborn as sns
sns.distplot(tsla_df['p_change'], bins=80)

sns.boxplot(x='date_week', y='p_change', data=tsla_df)

# 可视化两组数据的相关性及概率密度分布
sns.jointplot(tsla_df['high'], tsla_df['low'])

change_df = pd.DataFrame({'tsla':tsla_df.p_change})
# join google
change_df = change_df.join(pd.DataFrame({'goog':ABuSymbolPd.make_kl_df('usGOOG', n_folds=2).p_change}), how='outer')
# join apple
change_df = change_df.join(pd.DataFrame({'aapl':ABuSymbolPd.make_kl_df('usAAPL', n_folds=2).p_change}), how='outer')
# join facebook
change_df = change_df.join(pd.DataFrame({'fb':ABuSymbolPd.make_kl_df('usFB', n_folds=2).p_change}), how='outer')
# join baidu
change_df = change_df.join(pd.DataFrame({'bidu':ABuSymbolPd.make_kl_df('usBIDU', n_folds=2).p_change}), how='outer')
change_df = change_df.dropna()
change_df.head()

corr = change_df.corr()
_, ax = plt.subplots(figsize(8,5))
sns.heatmap(corr,ax=ax)


# CASE 1 可视化量化策略的交易区间
def plot_trade(buy_date, sell_date):
	start = tsla_df[tsla_df.index == buy_date].key.values[0]
	end = tsla_df[tsla_df.index == sell_date].key.values[0]

	plot_demo(just_series=True)
	plt.fill_between(tsla_df.index, 0, tsla_df['close'], color='blue', alpha=.08)

	# 标注股票持有周期为绿色，使用 start end 切片周期
	if tsla_df['close'][end] < tsla_df['close'][start]:
		plt.fill_between(tsla_df.index[start:end], 0, tsla_df['close'][start:end], color='green', alpha=.38)
	else:
		plt.fill_between(tsla_df.index[start:end], 0, tsla_df['close'][start:end], color='red', alpha=.38)

	plt.ylim(np.min(tsla_df['close']) -5, np.max(tsla_df['close']) + 5 )

	plt.legend(['close'], loc='best')

# 注意日期不能是周末
plot_trade('2018-01-29', '2018-04-16')

# CASE 2 标注卖出的原因
def plot_trade_with_annotate(buy_date, sell_date, annotate):
	plot_trade(buy_date, sell_date)
	plt.annotate(annotate, xy=(sell_date, tsla_df['close'].asof(sell_date)), arrowprops=dict(facecolor='yellow'), horizontalalignment='left', verticalalignment='top')

plot_trade_with_annotate('2018-01-29', '2018-04-16', '止损卖出')


# Example 2 标准化两个股票的观察周期
goog_df = ABuSymbolPd.make_kl_df('usGOOG', n_folds=2)
print(round(goog_df.close.mean(), 2))
print(round(goog_df.close.median(), 2))
goog_df.tail()

def plot_two_stock(tsla, goog, axs=None):
	drawer = plt if axs is None else axs
	drawer.plot(tsla, c='r')
	drawer.plot(goog, c='g')

	drawer.grid(True)
	drawer.legend(['tsla', 'google'], loc='best')

plot_two_stock(tsla_df.close, goog_df.close)
plt.title('TSLA and Google Close')
plt.xlabel('time')
plt.ylabel('close')

# 四种数据标准化处理的方式
def two_mean_list(one, two, type_look='look_max'):
	one_mean = one.mean()
	two_mean = two.mean()

	if type_look == 'look_max':
		one, two = (one, one_mean/two_mean * two) if one_mean > two_mean else ( one * two_mean/one_mean, two)
	elif type_look == 'look_min':
		one, two = (one * two_mean/one_mean ,two) if one_mean > two_mean else ( one, two * one_mean/two_mean)

def regular_std(group):
	return (group - group.mean()) / group.std()

def regular_mm(group):
	return (group - group.min())/(group.max() - group.min())

_, axs = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
drawer = axs[0][0]
plot_two_stock(regular_std(tsla_df.close), regular_std(goog_df.close), drawer)
drawer.set_title('(group-group.mean())/group.std()')

drawer = axs[0][1]
plot_two_stock(regular_mm(tsla_df.close), regular_mm(goog_df.close), drawer)
drawer.set_title('(group-group.min())/(group.max() - group.min())')

drawer = axs[1][0]
one, two = two_mean_list(tsla_df.close, goog_df.close, type_look='look_max')
plot_two_stock(one, two, drawer)
drawer.set_title('two_mean_list type_look=look_max')

drawer = axs[1][1]
one, two = two_mean_list(tsla_df.close, goog_df.close, type_look='look_min')
plot_two_stock(one, two, drawer)
drawer.set_title('two_mean_list type_look=look_min')













