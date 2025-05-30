import numpy as np
import pandas as pd
from scipy.stats import norm

from payoffs import *
from interpolations import *


def get_interp_vol(vol, k, t): # put TtM in cols
    """
    k: list or float; t float
    """
    return np.sqrt(linear_interpolation(t, vol.columns.values, [cubic_interpolation([k], vol.index.values, vol[e].values)**2 for e in vol]))


def dupire(f, k, vol, t, dt=0.001, dk=0.001):

    minVol = 0.005
    maxVol = 2
    vol.columns = t
    LV = np.zeros([len(k), len(t)])
    
    for j in range(len(t)):
        for i in range(len(k)):

            strike = k[i]
            y = np.log(strike/f.forward[j])

            sigma = vol.iloc[i, j]
            w = (sigma**2) * t[j]

            # vol derivatives
            dT = (linear_interpolation(t[j] + dt, t, vol.iloc[i, :].tolist()) - linear_interpolation(t[j] - dt, t, vol.iloc[i, :].tolist())) / (2*dt)
            dK = (cubic_interpolation([strike + dk], k, vol.iloc[:, j].tolist()) - cubic_interpolation([strike - dk], k, vol.iloc[:, j].tolist())) / (2*dk)
            dKK = (cubic_interpolation([strike + dk], k, vol.iloc[:, j].tolist()) - 2*sigma + cubic_interpolation([strike - dk], k, vol.iloc[:, j].tolist())) / (dk**2)

            # tvar derivatives
            dwT = sigma**2 + 2*sigma*dT*t[j] # *(r*strike*dK)
            dwK = 2*t[j]*sigma*dK
            dwKK = 2*t[j]*(dwK**2 + sigma*dKK)

            # mu = np.log(f.forward[j] / f.forward[j-1]) / (1/365)

            num = dwT # dwdt - dwdy * dydt
            g = (1 - (y/(2*w))*dwK)**2 - 0.25*(0.25 + 1/w)*(dwK**2) + 0.5*dwKK # den = 1 - (y/w)*dwdy + 0.25*(-0.25 - 1/w + (y/w)**2)*(dwdy**2) + 0.5*d2wdy2

            LV[i, j] = min(max(minVol, np.sqrt(max(num, 0.00001) / max(g, 0.00001))), maxVol)

            # print('for T=', t[j]*365, '& K=', strike, 'num=', num, 'den=', g, 'LV=', LV[i, j])
            # print('dT=', dT, 'dK=', dK)

    return pd.DataFrame(LV, columns=t, index=k)


def dupire_vol(vol, k, t):

    # calibrate market vols to local vols using dupire formula
    # when diffusing the spot, use interpolation to find the LV in the LV grid

    lv = get_interp_vol(vol, k, t)
    # dupire(f, k, vol, t, dt=0.001, dk=0.001)
    
    return lv


def dupire_price(vol, f, k, t, dt, dk):

    num = (black_forward_price(f, k, vol, t + dt, 0, option_type='call') - black_forward_price(f, k, vol, t - dt, 0, option_type='call')) / (2*dt)
    den = (black_forward_price(f, k-dk, vol, t, 0, option_type='call') + 2*black_forward_price(f, k, vol, t, 0, option_type='call') - black_forward_price(f, k+dk, vol, t, 0, option_type='call')) / (dk**2)

    return np.sqrt(num/(0.5*k*k*den))
