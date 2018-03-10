
from timeit import timeit
import numpy as np

normal_list = range(10000)
t = timeit('[i**2 for i in range(10000)]', number=1)
print(t)

x = timeit('np.arange(10000)**2', number=1)
print(x)
#%timeit [i**2 for i in normal_list]
