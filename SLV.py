import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import ndtr
from scipy.optimize import brentq
from interpolations import *
from scipy.interpolate import UnivariateSpline


def black_forward_price(w, f, k, t, vol, r):

    d1 = (np.log(f / k) + 0.5 * (vol**2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    price = np.exp(-r * t) * (w * f * ndtr(w * d1) - w * k * ndtr(w * d2))
    # price = np.exp(-r * t) * (w * f * norm.cdf(w * d1) - w * k * norm.cdf(w * d2))
    return price


def iv_black(w, f, k, t, r, p, x0=1):
    # Newton - Raphson algo 1st order difference
    maxIter = 10000 # orig 10000
    dVol = 0.001 # orig 0.0001
    eps = 10**(-8) # orig 10**(-8)
    i = 0
    a = black_forward_price(w, f, k, t, x0, r)
    # sigma = sigma + diff/vega # f(x) / f'(x)
    while abs(black_forward_price(w, f, k, t, x0, r) - p) >= eps and i < maxIter:
        v1 = black_forward_price(w, f, k, t, x0 + dVol, r)
        dx = (v1 - black_forward_price(w, f, k, t, x0, r)) / dVol
        x0 = x0 - (black_forward_price(w, f, k, t, x0, r) - p) / dx
        i = i + 1
    return x0


def iv_from_price(w, f, k, t, r, p):
    low, high = -0.00001, 4
    def objective_function(vols):
        return p - black_forward_price(w, f, k, t, vols, r)
    return brentq(objective_function, a=low, b=high, xtol=1e-12)


def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# def iv_black(w, f, k, t, r, p, x0=1):
#     MAX_ITERATIONS = 150
#     PRECISION = 1.0e-5
#     x0 = 0.5
#     for i in range(0, MAX_ITERATIONS):
#         price = black_forward_price(w, f, k, t, x0, r)
#         vega = bs_vega(f, k, t, r, x0)
#         diff = p - price  # our root
#         if (abs(diff) < PRECISION):
#             return x0
#         x0 = x0 + diff/vega # f(x) / f'(x)
#     return x0


def get_interp_vol2(vol, k, t):
    """
    k: list or float; t float
    Flat extrapolation
    """
    return np.sqrt(linear_interpolation(t, vol.columns.values, [cubic_interpolation(k, vol.index.values, vol[e].values)**2 for e in vol]))


def get_interp_vol(vols, k, t):
    # Linear extrapolation
    tmp = []
    for e in vols:
        extrapolator = UnivariateSpline(vols.index.values, vols[e].values, s=0)
        cc = extrapolator(k) ** 2
        tmp.append(cc)
    res = np.sqrt(linear_interpolation(t, vols.columns.values, tmp))
    return res


def get_local_vols(vols, t, k, f_start, f_end, dt=0.001, dk=0.001):

    t = t / 365
    sigma_bs = get_interp_vol(vols, k, t)
    y = np.log(k / f_end)
    w = (sigma_bs**2)*t
    dy_T = -np.log(f_end / f_start) / (1 / 365)

    d_sigma_T = (get_interp_vol(vols, k, t + dt) - get_interp_vol(vols, k, t - dt)) / (2*dt)
    d_sigma_K = (get_interp_vol(vols, k + dk, t) - get_interp_vol(vols, k - dk, t)) / (2*dk)
    a1 = get_interp_vol(vols, k + dk, t)
    a2 = get_interp_vol(vols, k - dk, t)
    d_sigma_K2 = (get_interp_vol(vols, k + dk, t) - 2*sigma_bs + get_interp_vol(vols, k - dk, t)) / (dk**2)

    dw_T = sigma_bs**2 + 2 * sigma_bs * d_sigma_T * t
    dwK = 2 * sigma_bs * t * d_sigma_K
    dwKK = 2 * t * (sigma_bs * d_sigma_K2 + (d_sigma_K**2))

    dwdy = k * dwK
    d2wdy2 = dwdy + (k**2) * dwKK
    dwdt = dw_T - dwdy * dy_T
    num = dwdt
    den = 1 - (y/w)*dwdy + 0.25*(-0.25 - 1/w + (y/w)**2)*(dwdy**2) + 0.5*d2wdy2

    return vols, num, den, np.sqrt(num / den)


def get_new_strike_grid(tenors, fwd, vol_atf, increment=0.05):

    q = np.zeros([43])

    q[0] = 0.001
    q[1] = 0.01    
    q[-2] = 1.99
    q[-1] = 1.999
    q[21] = 1

    for i in range(2, 21):
        q[i] = q[i-1] + increment
    for i in range(22, 41):
        q[i] = q[i-1] + increment
    
    K_up = np.zeros([len(tenors)])
    K_down = np.zeros([len(tenors)])
    d1_up = np.zeros([len(tenors)])
    d1_down = np.zeros([len(tenors)])
    d2_up = np.zeros([len(tenors)])
    d2_down = np.zeros([len(tenors)])
    call_up = np.zeros([len(tenors)])
    call_down = np.zeros([len(tenors)])
    p_atm = np.zeros([len(tenors)])

    fwd = fwd['forward']

    for e in range(len(tenors)):

        K_up[e] = fwd[e] * (1 + 1 / 10000)
        K_down[e] = fwd[e] * (1 - 1 / 10000)

        d1_up[e] = (np.log(fwd[e]/K_up[e]) + 0.5 * tenors[e] * vol_atf[e]**2) / (vol_atf[e] * np.sqrt(tenors[e]))
        d1_down[e] = (np.log(fwd[e]/K_down[e]) + 0.5 * tenors[e] * vol_atf[e]**2) / (vol_atf[e] * np.sqrt(tenors[e]))

        d2_up[e] = d1_up[e] - vol_atf[e] * np.sqrt(tenors[e])
        d2_down[e] = d1_down[e] - vol_atf[e] * np.sqrt(tenors[e])

        call_up[e] = fwd[e] * norm.cdf(d1_up[e]) - K_up[e] * norm.cdf(d2_up[e])
        call_down[e] = fwd[e] * norm.cdf(d1_down[e]) - K_down[e] * norm.cdf(d2_down[e])

        p_atm[e] = 1 - (call_down[e] - call_up[e]) / (2 * fwd[e] / 10000)

    results = np.zeros([len(tenors), 43])
    results[:, 21] = fwd
    for e in range(len(tenors)):
        for k in range(21):
            results[e, k] = fwd[e] * np.exp(-0.5 * vol_atf[e]**2 + vol_atf[e] * norm.ppf(q[k] * p_atm[e]))
        for k in range(22, 43):
            results[e, k] = fwd[e] * np.exp(-0.5 * vol_atf[e]**2 + vol_atf[e] * norm.ppf(1 - (2 - q[k]) * (1 - p_atm[e])))

    return results


def get_new_strike_grid2(strikes, nbr_strikes):
    strike2 = np.linspace(strikes[0] - strikes[0]*0.15, strikes[-1] + strikes[-1]*0.15, nbr_strikes)
    new_strikes = np.concatenate((strikes, strike2))
    new_strikes = np.sort(new_strikes)
    return new_strikes


def get_new_expiry_grid(expiries):
    t = [i for i in range(1, 21)]
    t += [5*i + 20 for i in range(1, 60)]
    t += [20*i + 20 for i in range(15, int(expiries[-1]))]
    t = [item for item in t if item < int(expiries[-1]) and item > int(expiries[0])]
    t += expiries
    t = list(set(t))
    t = sorted(t)
    t2 = [tt / 365 for tt in t]
    return t, t2


def lv_grid(spot, vols, fwd, exp_vol_grid):

    t = sorted(set(exp_vol_grid['days']))[0]
    f_end = linear_interpolation(t, fwd['days'].values, fwd['forward'].values)
    strikes = list(exp_vol_grid[exp_vol_grid['days'] == t]['strike'].values)
    expiries = list([t] * len(strikes))
    fwds = list([f_end] * len(strikes))
    dens = []
    nums = []
    lvs = []
    ivs = []
    f_start = linear_interpolation(t, fwd['days'].values, fwd['forward'].values)
    f_end = f_start
    k = exp_vol_grid[exp_vol_grid['days'] == t]['strike'].values
    iv, num, den, lv = get_local_vols(vols, t, k, f_start, f_end)
    num_floored = np.maximum(0.000001, num)
    den_mask = (den > 0)
    den2 = den[den_mask]
    den_floored = cubic_interpolation(k, np.sort(k[den_mask]), den2)
    lv = np.sqrt(num_floored / den_floored)
    nums.extend(num_floored)
    dens.extend(den_floored)
    ivs.extend(iv)
    lvs.extend(lv)

    for t in sorted(set(exp_vol_grid['days'])):

        f_start = linear_interpolation(t - 1, fwd['days'].values, fwd['forward'].values)
        f_end = linear_interpolation(t, fwd['days'].values, fwd['forward'].values)
        k = exp_vol_grid[exp_vol_grid['days'] == t]['strike'].values
        
        iv, num, den, lv = get_local_vols(vols, t, k, f_start, f_end)

        num_floored = np.maximum(0.000001, num)
        den_mask = (den > 0)
        den2 = den[den_mask]
        aaa = np.sort(k[den_mask])
        den_floored = cubic_interpolation(k, k[den_mask], den2)
        den_floored = previous_acceptable_den(den_floored)
        lv = np.sqrt(num_floored / den_floored)

        nums.extend(num_floored)
        dens.extend(den_floored)
        ivs.extend(iv)
        expiries.extend([t] * len(k))
        fwds.extend([f_end] * len(k))
        strikes.extend(k)
        lvs.extend(lv)

    return pd.DataFrame(data={'expiry': expiries, 'fwd': fwds, 'strike': strikes, 'num': nums, 'den': dens, 'lv': lvs})


def LV_IV(spot, bs_vols, fwd, step_grid, nb_paths):

    calls = bs_vols.copy()
    for i, e in enumerate(bs_vols):
        calls[e] = black_forward_price(1, fwd['forward'].values[i], bs_vols.index.values, e, bs_vols[e], fwd['rate'].values[i])

    step_grid['expiry'] = step_grid['expiry'].astype('int')
    expiries = sorted(set(step_grid['expiry']))
    dt = 1 / 365

    # simulation per expiry
    sim_e = {}
    for j, exp in enumerate(bs_vols.columns):

        z = np.random.normal(0, 1, [int(np.round(exp*365)), nb_paths])
        r = linear_interpolation(exp * 365, fwd['days'].values, fwd['rate'].values)
        S_t = np.zeros(((int(np.round(exp*365))+1), nb_paths))
        S_t[0] = [spot] * nb_paths

        for t in range(1, int(np.round(exp * 365)) + 1):
            if t / 365 > exp:
                break
            print('Calculate expiry ' + str(exp) + ' for time ' + str(t))
            if t <= expiries[0]:
                step_grid_up = step_grid[step_grid['expiry'] == expiries[0]].copy()
                lv = np.interp(S_t[t-1], step_grid_up['strike'].values, step_grid_up['lv'].values)
            elif t >= expiries[-1]:
                step_grid_up = step_grid[step_grid['expiry'] == expiries[-1]].copy()
                lv = np.interp(S_t[t-1], step_grid_up['strike'].values, step_grid_up['lv'].values)
            else:
                tp = 0
                while t >= expiries[tp]:
                    tp += 1
                d = (expiries[tp] - t) / (expiries[tp] - expiries[tp - 1])
                step_grid_up = step_grid[step_grid['expiry'] == expiries[tp]].copy()
                step_grid_down = step_grid[step_grid['expiry'] == expiries[tp - 1]].copy()
                lv = np.interp(S_t[t - 1], step_grid_down['strike'].values, step_grid_down['lv'].values) * d + np.interp(S_t[t - 1], step_grid_up['strike'].values, step_grid_up['lv'].values) * (1 - d) 

            S_t[t] = S_t[t-1] * np.exp((r - 0.5*(lv**2))*dt + lv*z[t-1]*np.sqrt(dt)) # GBM
            # S_t[t] = S_t[t-1] + r*S_t[t-1]*dt + lv*z[t-1]*np.sqrt(dt)*S_t[t-1] # Euler-Maruyama
            
        sim_e[j] = S_t[t]

    calls_mc = bs_vols.copy()
    puts_mc = bs_vols.copy()
    # calculate MC price and convert price to IV
    for i, e in enumerate(bs_vols.columns):
        for j, k in enumerate(bs_vols.index):
            calls_mc.iat[j, i] = np.exp(-fwd['rate'].values[i] * e) * np.mean(np.maximum(sim_e[i] - k, 0.0))
            puts_mc.iat[j, i] = np.exp(-fwd['rate'].values[i] * e) * np.mean(np.maximum(k - sim_e[i], 0.0))

    calib_vol_call = bs_vols.copy()
    calib_vol_put = bs_vols.copy()    
    for i, e in enumerate(bs_vols):
        for j, k in enumerate(bs_vols.index):
            try:
                calib_vol_call.iat[j, i] = iv_from_price(1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], calls_mc.iat[j, i])  
                calib_vol_put.iat[j, i] = iv_from_price(-1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], puts_mc.iat[j, i])  
            except:
                calib_vol_call.iat[j, i] = iv_black(1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], calls_mc.iat[j, i])  
                calib_vol_put.iat[j, i] = iv_black(-1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], puts_mc.iat[j, i])  

    return calls, S_t, calls_mc, puts_mc, calib_vol_call, calib_vol_put


def previous_acceptable_den(den):
    for i in range(len(den)):
        if den[i] < 0:
            den[i] = den[i-1]
    return den


def SLV_IV(spot, bs_vols, fwd, step_grid, rho, kappa, theta, xi, nbr_bins, nb_paths):

    calls = bs_vols.copy()
    for i, e in enumerate(bs_vols):
        calls[e] = black_forward_price(1, fwd['forward'].values[i], bs_vols.index.values, e, bs_vols[e], fwd['rate'].values[i])

    step_grid['expiry'] = step_grid['expiry'].astype('int')
    expiries = sorted(set(step_grid['expiry']))
    dt = 1 / 365

    # simulation per expiry
    sim_e = {}
    for j, exp in enumerate(bs_vols.columns):

        r = linear_interpolation(exp * 365, fwd['days'].values, fwd['rate'].values)

        W_S = np.random.normal(0, 1, [int(np.round(exp*365)), nb_paths])
        W_V = np.random.normal(0, 1, [int(np.round(exp*365)), nb_paths])
        W_V = rho * W_S + np.sqrt(1 - rho * rho) * W_V
        
        S_t = np.zeros(((int(np.round(exp*365))+1), nb_paths))
        S_t[0] = [spot] * nb_paths

        L = get_interp_vol(bs_vols, spot, j)
        
        V_t = np.zeros(((int(np.round(exp*365))+1), nb_paths))
        V_t[0] = [L.item()] * nb_paths

        for t in range(1, int(np.round(exp * 365)) + 1):
            if t / 365 > exp:
                break
            print('Calculate expiry ' + str(exp) + ' for time ' + str(t))
            if t <= expiries[0]:
                step_grid_up = step_grid[step_grid['expiry'] == expiries[0]].copy()
                lv = np.interp(S_t[t-1], step_grid_up['strike'].values, step_grid_up['lv'].values)
            elif t >= expiries[-1]:
                step_grid_up = step_grid[step_grid['expiry'] == expiries[-1]].copy()
                lv = np.interp(S_t[t-1], step_grid_up['strike'].values, step_grid_up['lv'].values)
            else:
                tp = 0
                while t >= expiries[tp]:
                    tp += 1
                d = (expiries[tp] - t) / (expiries[tp] - expiries[tp - 1])
                step_grid_up = step_grid[step_grid['expiry'] == expiries[tp]].copy()
                step_grid_down = step_grid[step_grid['expiry'] == expiries[tp - 1]].copy()
                lv = np.interp(S_t[t - 1], step_grid_down['strike'].values, step_grid_down['lv'].values) * d + np.interp(S_t[t - 1], step_grid_up['strike'].values, step_grid_up['lv'].values) * (1 - d) 

            vol1 = lv / np.sqrt(L)
            S_t[t] = S_t[t-1] + r*S_t[t-1]*dt + vol1*np.sqrt(V_t[t-1])*W_S[t-1]*np.sqrt(dt)*S_t[t-1] # Euler-Maruyama
            V_t[t] = V_t[t - 1] + kappa * (theta - V_t[t - 1]) * dt + xi * np.sqrt(V_t[t - 1] * dt) * W_V[t-1]
            # V_t[t] = V_t[t - 1] + kappa * (theta - V_t[t - 1]) * dt + xi * np.sqrt(max(V_t[t - 1], 0) * dt) * W_V

            # concat
            spot_vol = pd.concat([pd.DataFrame(S_t[t]), pd.DataFrame(V_t[t])], axis=1)
            spot_vol.columns = ['spot', 'vol']
            # order
            spot_vol_sorted = spot_vol.sort_values('spot') # long
            spot_vol_sorted = spot_vol_sorted.reset_index(drop=True)
            # mean from l bins = conditional expectation used for next time step
            bin = 0
            for l in range(nbr_bins):
                bin_mean = spot_vol_sorted.loc[l*nbr_bins:(l+1)*nbr_bins, 'vol'].mean() 
                bin += bin_mean # sum of means
            L = bin

        sim_e[j] = S_t[t]

    calls_mc = bs_vols.copy()
    puts_mc = bs_vols.copy()
    # calculate MC price and convert price to IV
    for i, e in enumerate(bs_vols.columns):
        for j, k in enumerate(bs_vols.index):
            calls_mc.iat[j, i] = np.exp(-fwd['rate'].values[i] * e) * np.mean(np.maximum(sim_e[i] - k, 0.0))
            puts_mc.iat[j, i] = np.exp(-fwd['rate'].values[i] * e) * np.mean(np.maximum(k - sim_e[i], 0.0))

    calib_vol_call = bs_vols.copy()
    calib_vol_put = bs_vols.copy()    
    for i, e in enumerate(bs_vols):
        for j, k in enumerate(bs_vols.index):
            try:
                calib_vol_call.iat[j, i] = iv_from_price(1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], calls_mc.iat[j, i])  
                calib_vol_put.iat[j, i] = iv_from_price(-1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], puts_mc.iat[j, i])  
            except:
                calib_vol_call.iat[j, i] = iv_black(1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], calls_mc.iat[j, i])  
                calib_vol_put.iat[j, i] = iv_black(-1, fwd['forward'].values[i], k, e, fwd['rate'].values[i], puts_mc.iat[j, i])  

    return calls, S_t, calls_mc, puts_mc, calib_vol_call, calib_vol_put
