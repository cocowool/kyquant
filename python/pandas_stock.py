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

# 使用
