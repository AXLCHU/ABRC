import numpy as np
import pandas as pd

from SLV import *
from payoffs import *


def GBM(r, sigma, T, S0, time_step, nbr_sim):

    dt = T / time_step
    gbm = np.zeros([time_step+1, nbr_sim])
    gbm[0] = S0
    for t in range(1, time_step+1):
        rand = np.random.standard_normal(nbr_sim)
        gbm[t] = gbm[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * rand)
    return gbm


def LV(r, v, T, S0, time_step, nbr_sim):

    dt = T / time_step
    spot_prices = np.zeros([time_step+1, nbr_sim])
    spot_prices[0] = S0

    for i in range(1, time_step+1):
        for n in range(nbr_sim):
            local_vol = dupire_vol(v, spot_prices[i - 1, n], i/time_step)
            W_S = np.random.standard_normal(1)
            spot_prices[i, n] = spot_prices[i - 1, n] + spot_prices[i - 1, n] * (r * dt + max(local_vol, 0) * np.sqrt(dt) * W_S)
    
    return spot_prices


def HESTON(r, v, T, S0, V0, rho, kappa, theta, xi, time_step, nbr_sim):
    
    dt = T / time_step

    spot_prices = np.zeros([time_step+1, nbr_sim])
    spot_prices[0] = S0
    vol_paths = np.zeros([time_step+1, nbr_sim])
    vol_paths[0] = get_interp_vol(v, S0, 0)

    for n in range(nbr_sim):
        for i in range(1, time_step+1):

            gauss_bm1 = np.random.standard_normal(1)
            gauss_bm2 = np.random.standard_normal(1)
            gauss_bm2 = rho * gauss_bm1 + np.sqrt(1 - rho * rho) * gauss_bm2

            drift = np.exp(dt * (r - 0.5 * vol_paths[i - 1, n] * vol_paths[i - 1, n]))
            vol = vol_paths[i - 1, n] * np.sqrt(dt)

            spot_prices[i, n] = spot_prices[i - 1, n] * drift * np.exp(vol * gauss_bm1)
            vol_paths[i, n] = vol_paths[i - 1, n] + kappa * (theta - vol_paths[i - 1, n]) * dt + xi * np.sqrt(max(vol_paths[i - 1, n], 0) * dt) * gauss_bm2

    return spot_prices, vol_paths


def LSV(r, v, T, S0, V0, kappa, theta, xi, rho, time_step, nbr_sim, nbr_bins):

    dt = T / time_step
    spot_prices = np.zeros([time_step+1, nbr_sim])
    spot_prices[0] = S0
    vol_paths = np.zeros([time_step+1, nbr_sim])
    vol_paths[0] = get_interp_vol(v, S0, 0)

    L = vol_paths[0, 0]
    for i in range(1, time_step+1):
        for n in range(nbr_sim):
            local_vol = dupire_vol(v, spot_prices[i - 1, n], i/time_step) ###
            vol1 = local_vol / np.sqrt(L)

            W_S = np.random.standard_normal(1)
            W_V = np.random.standard_normal(1)
            W_V = rho * W_S + np.sqrt(1 - rho * rho) * W_V

            spot_prices[i, n] = spot_prices[i - 1, n] + spot_prices[i - 1, n] * (r * dt + vol1 * max(vol_paths[i - 1, n], 0) * np.sqrt(dt) * W_S)
            vol_paths[i, n] = vol_paths[i - 1, n] + kappa * (theta - vol_paths[i - 1, n]) * dt + xi * np.sqrt(max(vol_paths[i - 1, n], 0) * dt) * W_V
        
        spot_vol = pd.concat([pd.DataFrame(spot_prices[i, :]), pd.DataFrame(vol_paths[i, :])], axis=1)
        spot_vol.columns = ['spot', 'vol']
        spot_vol_sorted = spot_vol.sort_values('spot')
        spot_vol_sorted = spot_vol_sorted.reset_index(drop=True)
        bin = 0
        for l in range(nbr_bins):
            bin_mean = spot_vol_sorted.loc[l*nbr_bins:(l+1)*nbr_bins, 'vol'].mean() 
            bin += bin_mean
        L = bin

    return spot_prices, vol_paths
