
from timeit import timeit
import numpy as np

### numpy 的效率很高，可以使用下面的例子来体会

# 普通方法
# normal_list = range(10000)
# %timeit [i**2 for i in range(10000)]
# numpy 方法
# np_list = np.arange(10000)
# %timeit np_list**2

### 初始化操作

# 100个0
np.zeros(100)

# shape: 3行2列，全是0
np.zeros((3,2))

# shape: 3行2列，全是1
np.ones((3,2))

# shape: x = 2, y = 3, z = 3 值随机
np.empty((2,3,3))

# eye() 得到对角线值全是1的矩阵
np.eye(3)

# 可以将普通list做为参数， 通过 np.array 来初始化 np array
data = [[1,2,3,4],[5,6,7,8]]
arr_np = np.array(data)
arr_np

#### linspace()

# linspace() 方法可以在两个值之间，按照固定的间隔生成序列
np.linspace(1,100,100)

#### 随机生成200只股票504个交易日服从正态分布的涨跌幅数据。504个交易日即2年。

# 200 只股票
stock_count = 200
# 504 个交易日
trade_days = 504
# 生成服从正态分布： 均值期望 ＝ 0，标准差 ＝ 1 的序列
stock_day_change = np.random.standard_normal((stock_count, trade_days))

# 打印shape(200,504) 200行504列
print(stock_day_change.shape)

# 打印第一只股票，前五个交易日的涨跌幅
print(stock_day_change[0:1,:5])

### 索引选取和切片选择

# 上面例子中 ```stock_day_change[0:1,:5]``` 就是一种索引选取和切片选择的方式，0:1 表示第一行，:5 表示前5列。
# 负数代表从后向前，-1 表示最后一个。

# 倒数第一只、第二只股票，最后五个交易日的数据
print(stock_day_change[-2:,-5:])

# 交换前两只和最后两只股票的数据
tmp = stock_day_change[0:2, 0:5].copy()
stock_day_change[0:2,0:5] = stock_day_change[-2:,-5:]
stock_day_change[-2:,-5:] = tmp

# 数据转换，将涨跌幅使用 astype(int) 转换为int
print(stock_day_change[0:2, 0:5])
stock_day_change[0:2, 0:5].astype(int)

# 使用 np.around() 函数，规整为float类型，只保留两位小数
np.around(stock_day_change[0:2,0:5],2)

# numpy 中 nan 表示数据缺失，下面的例子通过手工创建一个缺失数据，并用 nan_to_num 函数来用 0 填充 na
tmp_test = stock_day_change[0:2, 0:5].copy()
tmp_test[0][0] = np.nan
tmp_test

tmp_test = np.nan_to_num(tmp_test)
tmp_test

# 逻辑条件进行数据筛选，找出 涨幅超过 0.5 的股票时段，返回的 mask 是 bool 数组
mask = stock_day_change[0:2,0:5] > 0.5
print(mask)

# 使用 mask 筛选出符合条件的数据
tmp_test = stock_day_change[0:2, 0:5]
tmp_test[mask]

# 实际情况下，经常是筛选出符合条件的数据，并且重新赋值
tmp_test[tmp_test > 0.5] = 1
tmp_test

# 可以使用 | & 完成复合逻辑，每个条件都要用括号括起来
tmp_test = stock_day_change[-2:,-5:]
print(tmp_test)
tmp_test[(tmp_test > 1) | (tmp_test < -1)]

### 通用序列函数
# NumPy提供了一些可以高效处理序列的函数，下面是一些常用的
#### np.all()
# np.all 判断序列中的所有元素是否全部为 True
np.all(stock_day_change[0:2,0:5] > 0)

#### np.any()
# np.any 函数判断序列中是否有元素为 True
np.any(stock_day_change[0:2,0:5] > 0)

#### maximum() 与 minimum()
# 对两个序列对应的元素两两比较，maximum 结果集取大， minimum 结果集取小
np.maximum(stock_day_change[0:2,0:5], stock_day_change[-2:,-5:])
np.minimum(stock_day_change[0:2,0:5], stock_day_change[-2:,-5:])

#### np.unique() 
# 序列中数值唯一且不重复的值组成新的序列
change_int = stock_day_change[0:2,0:5].astype(int)
print(change_int)
np.unique(change_int)

#### np.diff() 函数
# Diff() 对前后两个临近数值进行减法运算，默认情况下 axis = 1
np.diff(stock_day_change[0:2, 0:5])
