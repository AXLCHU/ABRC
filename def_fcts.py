import numpy as np
from interpolations import cubic_interpolation
from svi_utils import *
import os
import matplotlib.pyplot as plt


def get_variables(spot, F, vols, e):

    k = sorted(set(vols.index), key=float)
    k = [float(s) for s in k]

    deltas = [np.log(kk / spot) for kk in k] # spot log moneyness

    log_forward_deltas = np.zeros([len(k), len(e)])
    for strike in range(len(k)):
        for expiry in range(len(e)):
            log_forward_deltas[strike, expiry] = np.log(k[strike] / F.forward[expiry])

    # Get vol ATM forward for each expiry
    vol_atm = np.zeros(len(e))
    for expiry in range(len(e)):
        vol_atm[expiry] = cubic_interpolation([0], deltas, vols.iloc[:, expiry])

    # Get vol ATF
    vol_atf = np.zeros(len(e))
    t_atf = log_forward_deltas.T
    for expiry in range(len(e)):
        vol_atf[expiry] = cubic_interpolation([0], t_atf[expiry], vols.iloc[:, expiry])

    # Get Z-deltas for SSVI
    z_deltas = np.zeros([len(e), len(k)])
    for strike in range(len(k)):
        for expiry in range(len(e)): 
            z_deltas[expiry, strike] = (np.log(k[strike] / F.forward[expiry])) / (vol_atf[expiry]* np.sqrt(e[expiry]))
        
    k_arr = np.array(k)
    # Create total variance matrix
    w = np.zeros([len(k), len(e)])
    for expiry in range(len(e)):
        for strike in range(len(k)):
            w[strike, expiry] = e[expiry] * (vols.iloc[strike, expiry] ** 2)

    return vol_atf, k, k_arr, deltas, z_deltas, log_forward_deltas, w


def butterfly_arbitrage(vols, k, e, F):

    calls = np.zeros([len(k), len(e)])
    for expiry in range(len(e)):
        for strike in range(len(k)):
            calls[strike, expiry] = black_forward_price(F.forward[expiry], k[strike], vols.iloc[strike, expiry], e[expiry], 0, option_type='call')

    calls_butterfly_arb = np.zeros([len(k), len(e)])
    for exp in range(len(e)):
        for s in range(1, len(k)-1):
            calls_butterfly_arb[s, exp] = calls[s-1, exp]*(k[s+1] - k[s]) / (k[s+1] - k[s-1]) + calls[s+1, exp]*(k[s] - k[s-1]) / (k[s+1] - k[s-1]) - calls[s, exp]
    
    nbr_butterfly_arb_call = np.sum((calls_butterfly_arb < 0))
    sum_neg_butterfly_arb_call = calls_butterfly_arb[calls_butterfly_arb < 0].sum()

    return calls_butterfly_arb, nbr_butterfly_arb_call, sum_neg_butterfly_arb_call


def calendar_arbitrage(vols, k, e, log_forward_deltas):

    w = np.zeros([len(k), len(e)])
    for expiry in range(len(e)):
        for strike in range(len(k)):
            w[strike, expiry] = e[expiry] * (vols.iloc[strike, expiry]**2)

    t_vols = vols.T
    calendar_arb = np.zeros([len(k), len(e) - 1])
    for expiry in range(len(e)-1):
        w0 = w[:, expiry]
        v1 = cubic_interpolation(log_forward_deltas[:, expiry], log_forward_deltas[:, expiry+1], vols.iloc[:, expiry+1].tolist())
        w1 = e[expiry+1] * (v1**2)
        calendar_arb[:, expiry] = w1 - w0
    
    nbr_calendar_arb = np.sum((calendar_arb < -0.00001))
    sum_neg_calendar_arb = calendar_arb[calendar_arb < 0].sum()

    return calendar_arb, nbr_calendar_arb, sum_neg_calendar_arb


def plot_vols(vols1, vols2, e, log_forward_deltas, outpath):

    if len(e) % 3 == 0:
        nbr_rows_plot = int(len(e)/3)
    else:
        nbr_rows_plot = int(len(e)/3) + 1

    fig, axs = plt.subplots(nrows=nbr_rows_plot, ncols=3, figsize=(15, 12), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.2)
    fig.suptitle('SPX 17/06/2024', fontsize=18, y=0.95)

    Le = len(axs.flatten()) - len(e)
    for l in range(Le):
        fig.delaxes(axs[-1, axs.ndim - l])

    for i, ax in enumerate(fig.axes):
        ax.scatter(log_forward_deltas[:, i], vols1.iloc[:, i], s=5, marker='+', color='red')
        #index_i = rbf_centers_index_all.iloc[:, i].to_list()
        #ax.scatter(rbf_centers_index_all.iloc[i, :], vols2.iloc[index_i, i], s=5, marker='+', color='black')
        ax.plot(log_forward_deltas[:, i], vols2.iloc[:, i], linewidth=0.5)
        ax.set_title('Maturity=' + vols2.columns[i])
        ax.set_xlabel('log(K/F)')
        ax.set_ylabel('Vol')
    plt.legend(['mdf', 'centers', 'rbf'], loc='upper right')
    plt.savefig(os.path.join(outpath, 'vols_spx.pdf'))
    plt.show()
    plt.close()
