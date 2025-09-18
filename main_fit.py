from datetime import datetime
import numpy as np
import pandas as pd
import configparser
import os

from def_fcts import *
from svi_utils import *
from LV import *


config = configparser.ConfigParser()
config.read('parameters.ini')

inpath = r'C:\Users\axelc\OneDrive\Documents\PY\ABRC\inputs'
outpath = r'C:\Users\axelc\OneDrive\Documents\PY\ABRC\outputs\20250630'

COB = '2025-06-30'
COB_P = datetime.strptime(COB, '%Y-%m-%d').date()

vols = pd.read_csv(os.path.join(inpath, COB.replace('-',''), 'vols_sp500.csv'), index_col=0, delimiter=';')
F = pd.read_csv(os.path.join(inpath, COB.replace('-',''), 'forward_sp500.csv'), delimiter=';')

spot = 6204.95 # 20250630

eq_expiries = [(datetime.strptime(d, '%d/%m/%Y').date() - COB_P).days for d in vols.columns.values]
# eq_expiries = [(datetime.strptime(d, '%Y-%m-%d').date() - COB_P).days for d in vols.columns.values]
e = [eq / 365 for eq in eq_expiries]

vol_atf, k, k_arr, deltas, z_deltas, log_forward_deltas, w = get_variables(spot, F, vols, e)

cal_arb, nbr_cal_arb, sum_neg_cal_arb = calendar_arbitrage(vols, k, e, log_forward_deltas) 
but_arb, nbr_but_arb, sum_neg_but_arb = butterfly_arbitrage(vols, k, e, F)     

vols.index = [float(k) for k in vols.index]
vols.columns = e
tenors = [t * 365 for t in e]
exp_grid, exp_grid_y = get_new_expiry_grid(tenors)

grid = pd.read_csv(os.path.join(inpath, COB.replace('-',''), 'grid_test.csv'), delimiter=';')

############################################################################################################################
# Calibrate LV
############################################################################################################################

calib_grid = lv_grid(spot, vols, F, grid)
calib_grid.to_csv(os.path.join(outpath, 'lv_sp500.csv'), index=None)

# calls, St, lv_calls, lv_puts, calib_vols_call, calib_vols_put = LV_IV(spot, vols, F, calib_grid, nb_paths=50000)

# LV = calib_vols_put.copy()
# for j in range(len(LV.columns)):
#     for i in range(len(LV.index)):
#         if vols.index[i] >= F.forward[j]:
#             LV.iloc[i, j] = calib_vols_call.iloc[i, j]

# LV.to_csv(os.path.join(outpath, 'LV_sp500.csv'))
# calib_vols_call.to_csv(os.path.join(outpath, 'LV_call_sp500.csv'))
# calib_vols_put.to_csv(os.path.join(outpath, 'LV_put_sp500.csv'))
# lv_calls.to_csv(os.path.join(outpath, 'calls_sp500.csv'))
# lv_puts.to_csv(os.path.join(outpath, 'puts_sp500.csv'))

############################################################################################################################
# Calibrate SLV
############################################################################################################################

nbr_sim = pow(30, 5) # > nbr_bins
nbr_bins = int(nbr_sim / pow(10, 3)) # define bin size - same nbr of MC path in each 
nbr_bins = 5

kappa = 1
theta = 0.3 #0.25
xi = 0.4 #0.02
rho = 0.05

calls, St, slv_calls, slv_puts, slv_call, slv_put = SLV_IV(spot, vols, F, calib_grid, rho, kappa, theta, xi, nbr_bins, nb_paths=nbr_sim)

SLV = slv_put.copy()
for j in range(len(SLV.columns)):
    for i in range(len(SLV.index)):
        if vols.index[i] >= F.forward[j]:
            SLV.iloc[i, j] = slv_call.iloc[i, j]

SLV.to_csv(os.path.join(outpath, 'SLV_sp500.csv'))
slv_call.to_csv(os.path.join(outpath, 'SLV_call_sp500.csv'))
slv_put.to_csv(os.path.join(outpath, 'SLV_put_sp500.csv'))
slv_calls.to_csv(os.path.join(outpath, 'slv_calls_sp500.csv'))
slv_puts.to_csv(os.path.join(outpath, 'slv_puts_sp500.csv'))

############################################################################################################################
# Plot
############################################################################################################################

if len(tenors) % 3 == 0:
    nbr_rows_plot = int(len(tenors)/3)
else:
    nbr_rows_plot = int(len(tenors)/3) + 1

fig, axs = plt.subplots(nrows=nbr_rows_plot, ncols=3, figsize=(15, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.2)
fig.suptitle('IV SPX 30/06/2025', fontsize=18, y=0.95)

Le = len(axs.flatten()) - len(tenors)
for l in range(Le):
    fig.delaxes(axs[-1, axs.ndim - l])

for i, ax in enumerate(fig.axes):
    # ax.scatter(log_forward_deltas[:, i], vols.iloc[:, i].T, s=5, marker='x', color='red')
    ax.scatter(log_forward_deltas[:, i], SLV.iloc[:, i].T, s=4, marker='<', color='black')
    # ax.plot(log_forward_deltas[:, i], iv_final.iloc[:, i].T, linewidth=0.5)
    ax.set_title('Maturity=' + str(round(tenors[i]))+'d')
    ax.set_xlabel('log(K/F)')
    ax.set_ylabel('Vol')
plt.legend(['BBG', 'IV LV'], loc='upper right')
plt.savefig(os.path.join(outpath, 'lsv_iv_sp500.pdf'))
# plt.savefig(os.path.join(outpath, 'lsv_iv_sp500.jpg'))
plt.show()
plt.close()

