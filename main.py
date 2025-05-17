import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt

from payoffs import *
from diffusions import *
from LV import *


########################################################################################################################################
# Import market IV surface
COB = '2024-06-17'
COB_P = datetime.strptime(COB, '%Y-%m-%d').date()

inpath = r'C:\Users\axelc\OneDrive\Documents\github\SSVI\inputs'
outpath = r'C:\Users\axelc\OneDrive\Documents\github\SSVI\outputs\20240617'

vols = pd.read_csv(os.path.join(inpath, 'vols_sp500.csv'), index_col=0, delimiter=',')
F = pd.read_csv(os.path.join(inpath, 'forward_sp500.csv'), delimiter=',')

eq_expiries = [(datetime.strptime(d, '%Y-%m-%d').date() - COB_P).days for d in vols.columns.values]
e = [eq / 365 for eq in eq_expiries]

# Convert to LV surface
k = sorted(set(vols.index), key=float)
k = [float(s) for s in k]
k_arr = np.array(k)

local_vols = dupire(F, k_arr, vols, e, dt=0.01, dk=0.01)

# plt.plot(vols.iloc[:, 1])
# plt.plot(local_vols)
# plt.plot(local_vols.iloc[:, 1])
# plt.show()

########################################################################################################################################
# Simulation parameters
time_step = 250
nbr_sim = pow(10, 4) # > nbr_bins
nbr_bins = int(nbr_sim / pow(10, 3)) # define bin size - same nbr of MC path in each 
nbr_bins = 5

# Asset parameters
K = 5500
S0 = 5473.23
r = 0.0425
T = 1
v = 0.2
act = np.exp(-r*T)

# SV parameters
V0 = 0.225
kappa = 1
theta = 0.25
xi = 0.02
rho = 0.6
########################################################################################################################################

# Seed
np.random.seed(1234567)

# GBM
# spot_gbm = GBM(r, v, T, S0, time_step, nbr_sim)

# HESTON model
# spot_sv, vol_sv = HESTON(r, v, T, S0, V0, rho, kappa, theta, xi, time_step, nbr_sim)

# LV model
spot_lv = LV(r, vols, T, S0, time_step, nbr_sim)

plt.plot(spot_lv)
plt.show()

# LSV model
# spot_lsv, vol_lsv = LSV(r, v, T, S0, V0, kappa, theta, xi, rho, time_step, nbr_sim, nbr_bins)

# Vanilla price calculation
call_gbm, call_sv, call_lv, call_lsv = 0, 0, 0, 0
for i in range(nbr_sim):
    # call_gbm += act * Call(spot_gbm[-1, i], K) / nbr_sim 
    # call_sv += act * Call(spot_sv[-1, i], K) / nbr_sim
    call_lv += act * Call(spot_lv[-1, i], K) / nbr_sim
    # call_lsv += act * Call(spot_lsv[-1, i], K) / nbr_sim

print('Call BS =', round(Call_BS(S0, K, T, r, v), 3))
print('Call GBM =', round(call_gbm, 3))
print('Call LV =', round(call_lv, 3))
print('Call SV =', round(call_sv, 3))
print('Call LSV =', round(call_lsv, 3))

# Exotic price calculation

# Plots
plt.plot(spot_lv)
plt.show()

x = 0
