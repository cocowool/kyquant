import pandas as pd
import seaborn as sns
import numpy as np
from abupy import ABuSymbolPd
import statsmodels.api as sm
from statsmodels import regression

# kl_pd = ABuSymbolPd.make_kl_df('usMSFT', n_folds=2)
# kl_pd = ABuSymbolPd.make_kl_df('usGSUM', n_folds=2)
kl_pd = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)
kl_pd.tail()

sns.set_context(rc={'figure.figsize':(14,7)})
sns.regplot(x=np.arange(0,kl_pd.shape[0]), y=kl_pd.close.values, marker='+')
plt.show()

def calc_regress_deg(y_arr, show=True):
    x = np.arange(0, len(y_arr))
    zoom_factor = x.max() / y_arr.max()

    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y_arr,x).fit()
    rad = model.params[1]

    deg = np.rad2deg(rad)

    if show:
        intercept = model.params[0]
        reg_y_fit = x * rad + intercept
        plt.plot(x,y_arr)
        plt.plot(x, reg_y_fit)
        plt.title('deg = ' + str(deg))
        plt.show()

    return deg

deg = calc_regress_deg(kl_pd.close.values)
