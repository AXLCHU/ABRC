import numpy as np
import pandas as pd
from datetime import datetime
import os
from time import time
import matplotlib.pyplot as plt

from payoffs import *
from diffusions import *
from LV import *


# Import market IV surface
COB = '2024-06-17'
COB_P = datetime.strptime(COB, '%Y-%m-%d').date()

inpath = r'...\inputs'
outpath = r'...\outputs'

vols = pd.read_csv(os.path.join(inpath, 'vols_sp500.csv'), index_col=0, delimiter=',')
F = pd.read_csv(os.path.join(inpath, 'forward_sp500.csv'), delimiter=',')

eq_expiries = [(datetime.strptime(d, '%Y-%m-%d').date() - COB_P).days for d in vols.columns.values]
e = [eq / 365 for eq in eq_expiries]
vols.columns = e

k = sorted(set(vols.index), key=float)
k = [float(s) for s in k]
k_arr = np.array(k)

# Simulation parameters
time_step = 50
nbr_sim = pow(10, 4) 
nbr_bins = int(nbr_sim / pow(10, 3)) 
nbr_bins = 5

# Asset parameters
K = 5500
S0 = 5473.23
r = 0.0425
T = 1
v = get_interp_vol(vols, S0, 0)[0]
act = np.exp(-r*T)

# SV parameters
V0 = v
kappa = 1
theta = 0.3 #0.25
xi = 0.4 #0.02
rho = 0.05

T_E = 12
T_O = 3
N = 4
K = 1
H = 0.8
Y = 0.0719
H_AC = 1
eps = 0.01
spots = [i/200 + 0.5 for i in range(201)]

autocall_geo, autocall_sv, autocall_lv, autocall_slv = [], [], [], []
delta_geo, delta_sv, delta_lv, delta_lsv = [], [], [], []
for s in spots:
    print('St=', s)
    S0 = s
    print('GBM at ', time())
    spot_gbm = GBM(r, v, T, S0, time_step, nbr_sim)
    print('SV at ', time())
    spot_sv, vol_sv = HESTON(r, vols, T, S0, V0, rho, kappa, theta, xi, time_step, nbr_sim)
    print('LV at ', time())
    spot_lv = LV(r, vols, T, S0, time_step, nbr_sim)
    print('LSV at ', time())
    spot_lsv, vol_lsv = LSV(r, vols, T, S0, V0, kappa, theta, xi, rho, time_step, nbr_sim, nbr_bins)
    
    minS_geo = spot_gbm[-1, :].min()
    minS_sv = spot_sv[-1, :].min()
    minS_lv = spot_lv[-1, :].min()
    minS_slv = spot_lsv[-1, :].min()
    
    abrc_geo, abrc_sv, abrc_lv, abrc_slv = 0, 0, 0, 0
    delta_geo_up, delta_geo_down, delta_sv_up, delta_sv_down = 0, 0, 0, 0
    delta_lv_up, delta_lv_down, delta_lsv_up, delta_lsv_down = 0, 0, 0, 0
    for i in range(nbr_sim):
        abrc_geo += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_gbm[-1, i], minS_geo) / nbr_sim
        abrc_sv += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_sv[-1, i], minS_sv) / nbr_sim
        abrc_lv += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lv[-1, i], minS_sv) / nbr_sim
        abrc_slv += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lsv[-1, i], minS_sv) / nbr_sim

        delta_geo_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_gbm[-1, i] + eps, minS_geo) / nbr_sim
        delta_geo_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_gbm[-1, i] - eps, minS_geo) / nbr_sim

        delta_sv_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_sv[-1, i] + eps, minS_geo) / nbr_sim
        delta_sv_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_sv[-1, i] - eps, minS_geo) / nbr_sim

        delta_lv_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lv[-1, i] + eps, minS_geo) / nbr_sim
        delta_lv_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lv[-1, i] - eps, minS_geo) / nbr_sim

        delta_lsv_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lsv[-1, i] + eps, minS_geo) / nbr_sim
        delta_lsv_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lsv[-1, i] - eps, minS_geo) / nbr_sim

    autocall_geo.append([s, abrc_geo])
    autocall_sv.append([s, abrc_sv])
    autocall_lv.append([s, abrc_lv])
    autocall_slv.append([s, abrc_slv])

    delta_geo.append([s, (delta_geo_up - delta_geo_down) / (2 * eps)])
    delta_sv.append([s, (delta_sv_up - delta_sv_down) / (2 * eps)])
    delta_lv.append([s, (delta_lv_up - delta_lv_down) / (2 * eps)])
    delta_lsv.append([s, (delta_lsv_up - delta_lsv_down) / (2 * eps)])

    print('ABRC GBM  at t=', s, '=', round(abrc, 3))
autocall_geo = pd.DataFrame(autocall_geo, columns=['St', 'P'])
autocall_sv = pd.DataFrame(autocall_sv, columns=['St', 'P'])
autocall_lv = pd.DataFrame(autocall_lv, columns=['St', 'P'])
autocall_slv = pd.DataFrame(autocall_slv, columns=['St', 'P'])

delta_geo = pd.DataFrame(delta_geo, columns=['St', 'Delta'])
delta_sv = pd.DataFrame(delta_sv, columns=['St', 'Delta'])
delta_lv = pd.DataFrame(delta_lv, columns=['St', 'Delta'])
delta_lsv = pd.DataFrame(delta_lsv, columns=['St', 'Delta'])

# Prices
plt.plot(autocall_geo['St'], autocall_geo['P'], linewidth=0.8) # label='GBM', 
plt.plot(autocall_sv['St'], autocall_sv['P'], label='SV', linewidth=0.8)
plt.plot(autocall_lv['St'], autocall_lv['P'], label='LV', linewidth=0.8)
plt.plot(autocall_slv['St'], autocall_slv['P'], label='LSV', linewidth=0.8)
plt.xlabel('S0')
plt.ylabel('Price')
plt.grid(True)
plt.legend(loc='upper right')
plt.title('Price Autocall')
plt.savefig(r'...\abrc_price.jpg')
plt.show()
plt.close()

# Deltas
plt.plot(delta_geo['St'], delta_geo['Delta'], linewidth=0.8) # label='GBM', 
plt.plot(delta_sv['St'], delta_sv['Delta'], label='SV', linewidth=0.8)
plt.plot(delta_lv['St'], delta_lv['Delta'], label='LV', linewidth=0.8)
plt.plot(delta_lsv['St'], delta_lsv['Delta'], label='LSV', linewidth=0.8)
plt.xlabel('S0')
plt.ylabel('Delta')
plt.grid(True)
plt.legend(loc='upper right')
plt.title('Delta Autocall')
plt.savefig(r'...\abrc_delta.jpg')
plt.show()
plt.close()

# Vega
print('Calculating Vega...')

vega_geo, vega_sv, vega_lv, vega_lsv = [], [], [], []
for s in spots:
    print('St=', s)
    S0 = s
    print('GBM at ', time())
    spot_gbm_up = GBM(r, v + eps, T, S0, time_step, nbr_sim)
    spot_gbm_down = GBM(r, v - eps, T, S0, time_step, nbr_sim)
    print('SV')
    spot_sv_up, vol_sv_up = HESTON(r, vols + eps, T, S0, V0, rho, kappa, theta, xi, time_step, nbr_sim)
    spot_sv_down, vol_sv_down = HESTON(r, vols - eps, T, S0, V0, rho, kappa, theta, xi, time_step, nbr_sim)
    print('LV')
    spot_lv_up = LV(r, vols + eps, T, S0, time_step, nbr_sim)
    spot_lv_down = LV(r, vols - eps, T, S0, time_step, nbr_sim)
    print('LSV')
    spot_lsv_up, vol_lsv_up = LSV(r, vols + eps, T, S0, V0, kappa, theta, xi, rho, time_step, nbr_sim, nbr_bins)
    spot_lsv_down, vol_lsv_down = LSV(r, vols - eps, T, S0, V0, kappa, theta, xi, rho, time_step, nbr_sim, nbr_bins)

    minS_geo = spot_gbm[-1, :].min()
    minS_sv = spot_sv[-1, :].min()
    minS_lv = spot_lv[-1, :].min()
    minS_lsv = spot_lsv[-1, :].min()

    vega_geo_up, vega_geo_down, vega_sv_up, vega_sv_down = 0, 0, 0, 0
    vega_lv_up, vega_lv_down, vega_lsv_up, vega_lsv_down = 0, 0, 0, 0
    for i in range(nbr_sim):
        vega_geo_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_gbm_up[-1, i], minS_geo) / nbr_sim
        vega_geo_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_gbm_down[-1, i], minS_geo) / nbr_sim

        vega_sv_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_sv_up[-1, i], minS_geo) / nbr_sim
        vega_sv_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_sv_down[-1, i], minS_geo) / nbr_sim

        vega_lv_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lv_up[-1, i], minS_geo) / nbr_sim
        vega_lv_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lv_down[-1, i], minS_geo) / nbr_sim

        vega_lsv_up += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lsv_up[-1, i], minS_geo) / nbr_sim
        vega_lsv_down += act * ABRC(K, H, H_AC, Y, T_E, T_O, N, v, spot_lsv_down[-1, i], minS_geo) / nbr_sim

    vega_geo.append([s, (vega_geo_up - vega_geo_down) / (2 * eps)])
    vega_sv.append([s, (vega_sv_up - vega_sv_down) / (2 * eps)])
    vega_lv.append([s, (vega_lv_up - vega_lv_down) / (2 * eps)])
    vega_lsv.append([s, (vega_lsv_up - vega_lsv_down) / (2 * eps)])

vega_geo = pd.DataFrame(vega_geo, columns=['St', 'Vega'])
vega_sv = pd.DataFrame(vega_sv, columns=['St', 'Vega'])
vega_lv = pd.DataFrame(vega_lv, columns=['St', 'Vega'])
vega_lsv = pd.DataFrame(vega_lsv, columns=['St', 'Vega'])


# Vegas
plt.plot(vega_geo['St'], vega_geo['Vega'], linewidth=0.8) # , label='GBM',
plt.plot(vega_sv['St'], vega_sv['Vega'], label='SV', linewidth=0.8)
plt.plot(vega_lv['St'], vega_lv['Vega'], label='LV', linewidth=0.8)
plt.plot(vega_lsv['St'], vega_lsv['Vega'], label='LSV', linewidth=0.8)
plt.xlabel('S0')
plt.ylabel('Vega')
plt.grid(True)
plt.legend(loc='upper right')
plt.title('Vega Autocall')
plt.savefig(r'...\abrc_vega.jpg')
plt.show()
plt.close()

# plt.plot(GBM(r, v, T, 1, time_step, nbr_sim))
plt.plot([H for i in range(time_step)])
plt.plot([H_AC for i in range(time_step)])
plt.show()
plt.close()
