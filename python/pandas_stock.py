# 使用真正的股票数据构建 DataFrame 对象

import pandas as pd
import numpy as np
from abupy import ABuSymbolPd

# n_folds = 2 两年
tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds = 2)
tsla_df.tail()

tsla_df[['close','volume']].plot(subplots=True, style=['r','g'], grid=True)

# 查看DF对象的信息
tsla_df.info()

# 查看DF对象的统计信息
tsla_df.describe()

# 使用 loc 配合行名称、列名称进行索引选取和切片选择
tsla_df.loc['2018-01-01':'2018-02-01','open']

# 如果不传入列名默认表示取所有列
tsla_df.loc['2018-01-01':'2018-02-01']

# 使用行索引数及列索引数
tsla_df.iloc[1:5, 2:6]

# 混合使用方法
tsla_df[['close','high','low']][0:3]

### 逻辑条件进行数据筛选
tsla_df[np.abs(tsla_df.netChangeRatio) > 8]

# 可以使用 | & 完成复合逻辑，各个条件需要使用括号括起来
[(np.abs(tsla_df.netChangeRatio) > 8) & (tsla_df.volume > 2.5 * tsla_df.volume.mean())]

### 数据转换与规整

# 数据排序
tsla_df.sort_index(by='netChangeRatio')[:5]

# 数据排序，倒序
tsla_df.sort_index(by='netChangeRatio', ascending=False)[:5]

# pct_change() 函数对序列从第二项开始向前做减法
tsla_df.close[:3]

# 正对收盘价格做运算后得出涨跌幅
tsla_df.close.pct_change()[:3]

change_ratio = tsla_df.close.pct_change()
change_ratio.tail()

# 数据持久化，本地序列化
tsla_df.to_csv('tsla_df.csv', columns=tsla_df.columns, index=True)

# 数据读取
tsla_df_load = pd.read_csv('tsla_df.csv', parse_dates=True, index_col = 0)
tsla_df_load.tail()

### 寻找股票异动涨跌幅阈值
tsla_df.netChangeRatio.hist(bins=80)
cats = pd.qcut(np.abs(tsla_df.netChangeRatio), 10)
cats.value_counts()

# 可以自定义分类标准，将数据进行分类统计
bins = [-np.inf, -7.0, -5,-3,0,3,5,7,np.inf]
cats = pd.cut(tsla_df.netChangeRatio, bins)
cats.value_counts()

# pd.cut() 经常与 pd.get_dummies() 配合使用，将数据由连续数值类型变成离散类型，数据离散化后的哑变量矩阵多用于机器学习中监督学习问题分类，用来作为训练数据

change_ratio_dummies = pd.get_dummies(cats, prefix='cr_dummies')
change_ratio_dummies.tail()

# 通过 where 条件添加新列
tsla_df['positive'] = np.where(tsla_df.netChangeRatio > 0 , 1, 0)
tsla_df.tail()

# 构建交叉表
xt = pd.crosstab(tsla_df.date_week, tsla_df.positive)

# 对交叉表进行转换，计算所占的比例
xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
xt_pct

# 可视化结果
xt_pct.plot( figsize = (8,5), kind='bar', stacked = True, title = 'date_week -> positive')
plt.xlabel('date_week')
plt.ylabel('positive')

# 也可以通过 pivot_table 和 groupby 来实现构建透视表的过程
tsla_df.pivot_table(['positive'],index=['date_week'])
tsla_df.groupby(['date_week','positive'])['positive'].count()
