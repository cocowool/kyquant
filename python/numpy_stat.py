from timeit import timeit
import numpy as np

stock_day_change = np.load('stock_day_change.npy')

#### 数据准备
# 用切片保留前4只股票，前4天的涨跌幅数据
stock_day_change_four = stock_day_change[:4,:4]

# 生成a、b两组数据，假设a、b两人100次交易结果为赚100，a的稳定性不好，振幅即标准差为50，b的稳定性稍好，振幅为20
a_investor = np.random.normal(loc=100,scale=50,size=(100,1))
b_investor = np.random.normal(loc=100,scale=20,size=(100,1))

# 输出数据的标准差、方差及期望数据
print('a 交易者期望{0:.2f}元，标准差{1:.2f}，方差{2:.2f}'.format(a_investor.mean(), a_investor.std(), a_investor.var()))
print('b 交易者期望{0:.2f}元，标准差{1:.2f}，方差{2:.2f}'.format(b_investor.mean(), b_investor.std(), b_investor.var()))

# 画出两者的获利图，更直观了解
a_mean = a_investor.mean()
a_std = a_investor.std()
plt.plot(a_investor)
plt.axhline(a_mean + a_std, color='r')
plt.axhline(a_mean, color='y')
plt.axhline(a_mean - a_std, color='g')

b_mean = b_investor.mean()
b_std = b_investor.std()
plt.plot(b_investor)
plt.axhline(b_mean + b_std, color='r')
plt.axhline(b_mean, color='y')
plt.axhline(b_mean - b_std, color='g')

# 正态分布
# 200 只股票
stock_count = 200
# 504 个交易日
trade_days = 504

stock_day_change = np.random.standard_normal((stock_count, trade_days))

import scipy.stats as scs
# 均值期望
stock_mean = stock_day_change[0].mean()
# 标准差
stock_std = stock_day_change[0].std()
print('股票 0 mean均值期望：{:.3f}'.format(stock_mean))
print('股票 0 std振幅标准差：{:.3f}'.format(stock_std))
plt.hist(stock_day_change[0],bins=50,normed=True)

fit_linspace = np.linspace(stock_day_change[0].min(), stock_day_change[0].max())
# 概率密度函数（PDF）
pdf = scs.norm(stock_mean, stock_std).pdf(fit_linspace)
plt.plot(fit_linspace, pdf, lw=2, c='r')


# 正态分布买入策略
# 保留最后50天数据作为验证
keep_days = 50
# 统计前454天中的200只股票的涨跌幅数据，通过切片实现
stock_day_change_test = stock_day_change[:stock_count, 0:trade_days - keep_days]

# 打印跌幅最大的3只股票，总跌幅通过 np.sum 计算 np.sort 对数据进行排序
print(np.sort(np.sum(stock_day_change_test, axis=1))[:3])

# 使用 np.argsort 对股票跌幅进行排序，并返回序号
stock_lower_array = np.argsort(np.sum(stock_day_change_test, axis=1))[:3]

stock_lower_array

# 定义盈亏比例变量
profit = 0
for stock_ind in stock_lower_array:
	profit += show_buy_lower(stock_ind)

print('买入第 {} 只股票，从第454个交易日开始持有盈亏：{:.2f}%'.format(stock_lower_array, profit))

def show_buy_lower(stock_ind):
	# stock_ind 表示股票序号，后续可以使用股票代码
	# 设置一个一行两列的图表
	_, axs = plt.subplots(nrows=1,ncols=2,figsize=(16,5))
	# 绘制走势图，即涨跌幅连续求和
	axs[0].plot(np.arange(0, trade_days - keep_days), stock_day_change_test[stock_ind].cumsum())
	# 最后50天的走势
	cs_buy = stock_day_change[stock_ind][trade_days - keep_days:trade_days].cumsum()
	# 绘制图表
	axs[1].plot(np.arange(trade_days - keep_days, trade_days), cs_buy)

	return cs_buy[-1]


# 实例2，如何在交易中获取优势
# 设置100个赌徒
gamblers = 100
def casino(win_rate, win_once = 1, loss_once = 1, commission=0.01):
	# 赌场：假设每个赌徒都有1000000元，并且每个赌徒都想玩10000000次，但如果没钱了就不能玩
	# win_rage 输赢的概率
	# win_once 每次赢的钱数
	# loss_once 每次输的钱数
	# commission 手续费
	my_money = 1000000
	play_cnt = 10000000
	commission = commission
	for _ in np.arange(0, play_cnt):
		# 使用伯努利分布，根据 win_rate 来获取输赢
		w = np.random.binomial(1, win_rate)
		if w:
			my_money += win_once
		else:
			my_money -= loss_once
		my_money -= commission
		if my_money <= 0:
			break

	return my_money

# 天堂赌场，胜率 0.5,赔率 1 ，没有手续费
heaven_moneys = [casino(0.5,commission=0) for _ in np.arange(0,gamblers)]

cheat_moneys = [casino(0.4,commission=0) for _ in np.arange(0,gamblers)]