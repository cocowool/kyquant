# 使用 numpy 模拟 GRE 中的随机选择算法，并使用 pyplot绘图

import numpy as np
from numpy import random

r = random.randint(1,301,size = (300,225) )
a = {}
for i in r:
    for j in i:
        if(j in a.keys()):
            a[j] = a[j] + 1
        else:
            a[j] = 0

height = []
z = a.values()
for i in z:
    height.append(i)

height.sort()
x = np.arange(1,301)

plt.bar(x,height)
plt.axis([0,301,0,280])
plt.grid(True)
plt.title("75%子集，225个后端")