import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm #, t
from scipy.optimize import minimize


def black_forward_price(f, k, vol, t, r, option_type):

    d1 = (np.log(f / k) + ((vol**2 / 2)*t) / (vol * np.sqrt(t)))
    d2 = d1 - vol * np.sqrt(t)

    if option_type == 'call':
        price = np.exp(-r*t) * (f*norm.cdf(d1) - k*norm.cdf(d2))
    elif option_type == 'put':
        price = np.exp(-r*t) * (k*norm.cdf(-d2) - f*norm.cdf(-d1))
    
    return price


def vega_grid(k, f, iv, t):
    return norm.cdf((np.log(f / k) + 0.5 *(iv**2) * t) / (iv * np.sqrt(t)))


def delta_grid(k, f, iv, t):
    d1 = (np.log(f / k) + 0.5 * (iv**2) * t)
    return f * np.sqrt(t) * np.exp(-np.square(d1) / 2) / np.square(2 * np.pi)


def ssvi_vol(atm, s2, c2, z):
    return atm * (0.5 * (1 + s2*z) + np.square(0.25 * ((1 + s2*z)**2) + 0.5*c2*(z**2)))


def svi_vol(a, b, rho, m, sigma, t, delta):
    var = a + b * (rho * (delta - m) + np.sqrt((delta - m)**2 + sigma**2))
    return np.sqrt(np.maximum(var, 0.0) / t)


def gsvi_vol(a, b, rho, m, sigma, beta, t, k):
    z = (k - m) / (beta**np.abs(k - m))
    var = a + b * (rho * (z - m) + np.sqrt((z- m)**2 + sigma**2))
    return np.sqrt(np.maximum(var, 0.0) / t)


def ssvi_vol_penalty_term(var, t, next_var, s):
    return s * max((var * t - next_var) / t, 0)


def ssvi_vol_errors(params, iv, z_all, k_all, atm, forward, t, w, target='variance'): #,
#                    penalty='none', tvar_next_slice=None, penalty_s=0.0, w_pen=None):

    s2 = params[0]
    c2 = params[1]
    var_model = [ssvi_vol(atm**2, s2, c2, z) for z in z_all]
    f = 0
 #   p_max = 0
    for i in range(len(z_all)):

        if target == 'variance':
            f += ((var_model[i] - (iv[i]**2))**2) * w[i]

        elif target == 'vol':
            f += ((np.sqrt(var_model[i]) - iv[i])**2) * w[i]

        elif target == 'price':
            option = 'call' if k_all[i] > f else 'put'
            model_price = black_forward_price(forward, k_all[i], np.sqrt(var_model[i]), t, 0, option)
            market_price = black_forward_price(forward, k_all[i], iv[i], t, 0, option)
            f += ((model_price - market_price)**2) * w[i]

    #     if penalty == 'equal' or penalty == 'vega':
    #         f += ssvi_vol_penalty_term(var_model[i], t, tvar_next_slice[i], penalty_s) * w_pen[i]

    #     elif penalty == 'max':
    #         p_max = max(p_max, ssvi_vol_penalty_term(var_model[i], t, tvar_next_slice[i], penalty_s))

    # if penalty == 'max':
    #     f += p_max

        return f


def ssvi_c_constraint(ssvi_parameters, tol):
    c = ssvi_parameters[1]
    return c - tol


def ssvi_butterfly_arb(svi_parameters, v):
    s = svi_parameters[0]
    c = svi_parameters[1]
    s0 = np.sqrt(np.where(c <= 2, 4-v, 4/v))
    c0_temp1 = (np.square(1 - ((1/8) * v)) + v)
    c0_temp2 = (5-((1/8)*v)) / c0_temp1
    c0 = np.where(v <= 4, c0_temp2 + np.sqrt(np.square(c0_temp2) - (1 / (c0_temp1))), 8/v)
    return 1 - (np.abs(s) / s0) - (c / c0)


def ssvi_calibrate(z, k, iv, s, f, t, atm, optimizer1, optimizer2, xn, x0, ssvi_target, ssvi_weights, ssvi_deltacutoff,
                   ssvi_cmin, tol=10**-12):
    """
    z: ndarray
    k: ndarray
    iv: ndarray, target vol
    s: float
    f: float
    t: float, time to expiry in year
    atm: float, ATM forward vol
    x, x0: ndarray
    """

    w = np.ones(len(iv))
#    w_pen = np.ones(len(iv))

    if ssvi_deltacutoff > 0:
        w = np.array(delta_grid(k, f, iv, t))
        w = (w > ssvi_deltacutoff) & (w < (1 - ssvi_deltacutoff))
        w = w.astype(float)
    
    if ssvi_weights == 'vega':
        w *= vega_grid(k, f, iv, t)
    
    w = w / np.sum(w)
    
    cons = [{'type': 'ineq', 'fun': ssvi_c_constraint, 'args': [ssvi_cmin]}]

    # if butterfly_constraint:
    #     cons.append({'type': 'ineq', 'fun': ssvi_butterfly_arb, 'args': [np.square(atm)*t]})

    calibration_result = minimize(ssvi_vol_errors, xn, args=(iv, z, k, atm, f, t, w, ssvi_target),
                                  method=optimizer1, constraints=cons, tol=tol)

    if calibration_result.success:
        method = optimizer1 + '_lastguess'
    else:
        calibration_result = minimize(ssvi_vol_errors, xn, args=(iv, z, k, atm, f, t, w, ssvi_target),
                                  method=optimizer2, constraints=cons, tol=tol)
        if calibration_result.success:
            method = optimizer2 + '_lastguess'
        else:
            method = 'failed'
    
    return calibration_result, method


def ssvi_calib(z_deltas, k, k_arr, vols, spot, F, e, vol_atf, params):

    ssvi_c2 = np.zeros([len(e)])
    ssvi_s2 = np.zeros([len(e)])
    vol_ssvi = np.zeros([len(e), len(k)])
    ssvi_params = np.zeros([len(e), 3])

    for expiry in reversed(range(len(e))):

        if expiry == len(e) - 1:

            ssvi, method = ssvi_calibrate(z_deltas[expiry, :], k_arr, vols.iloc[:, expiry].values, spot,
                                        F.forward[expiry], e[expiry], vol_atf[expiry],
                                        optimizer1 = params['OPTIMIZER_1'], optimizer2 = params['OPTIMIZER_2'],
                                        xn=[float(params['S2_GUESS']), float(params['C2_GUESS'])],
                                        x0=[float(params['S2_GUESS']), float(params['C2_GUESS'])],
                                        ssvi_target=params['TARGET'],
                                        ssvi_weights=params['WEIGHTS'], ssvi_deltacutoff=float(params['DELTA_CUTOFF']), 
                                        ssvi_cmin=float(params['C2_MIN']))

            ssvi_s2[expiry], ssvi_c2[expiry] = ssvi.x

        else:

            guess = [ssvi_s2[expiry+1], ssvi_c2[expiry+1]]

            ssvi, method = ssvi_calibrate(z_deltas[expiry, :], k_arr, vols.iloc[:, expiry].values, spot,
                                        F.forward[expiry], e[expiry], vol_atf[expiry],
                                        optimizer1 = params['OPTIMIZER_1'], optimizer2 = params['OPTIMIZER_2'],
                                        xn=guess, x0=guess, ssvi_target=params['TARGET'],
                                        ssvi_weights=params['WEIGHTS'], ssvi_deltacutoff=float(params['DELTA_CUTOFF']), 
                                        ssvi_cmin=float(params['C2_MIN']))

            ssvi_s2[expiry], ssvi_c2[expiry] = ssvi.x

        vol_ssvi[expiry, :] = np.sqrt(ssvi_vol(vol_atf[expiry]**2, ssvi_s2[expiry], ssvi_c2[expiry], z_deltas[expiry, :]))
        ssvi_params[expiry, :] = [e[expiry], ssvi_s2[expiry], ssvi_c2[expiry]]

    return ssvi_s2, ssvi_c2, ssvi_params, vol_ssvi


###################################################################################################################################
# WITH PENALTY
###################################################################################################################################

def ssvi_vol_errors_pen(params, iv, z_all, k_all, atm, forward, t, w, target, penalty, tvar_next_slice, penalty_s, w_pen):
    """
    w = weight for error
    w_pen: weight for penalty cross
    """

    s2 = params[0]
    c2 = params[1]
    var_model = [ssvi_vol(atm**2, s2, c2, z) for z in z_all]
    f = 0
    p_max = 0
    for i in range(len(z_all)):

        if target == 'variance':
            f += ((var_model[i] - (iv[i]**2))**2) * w[i]

        elif target == 'vol':
            f += ((np.sqrt(var_model[i]) - iv[i])**2) * w[i]

        elif target == 'price':
            option = 'call' if k_all[i] > f else 'put'
            model_price = black_forward_price(forward, k_all[i], np.sqrt(var_model[i]), t, 0, option)
            market_price = black_forward_price(forward, k_all[i], iv[i], t, 0, option)
            f += ((model_price - market_price)**2) * w[i]

        if penalty == 'equal' or penalty == 'vega':
            f += ssvi_vol_penalty_term(var_model[i], t, tvar_next_slice[i], penalty_s) * w_pen[i]

        elif penalty == 'max':
            p_max = max(p_max, ssvi_vol_penalty_term(var_model[i], t, tvar_next_slice[i], penalty_s))

    if penalty == 'max':
        f += p_max

    return f


def ssvi_calibrate_pen(z, k, iv, s, f, t, atm, optimizer1, optimizer2, xn, x0, ssvi_target, ssvi_weights, ssvi_deltacutoff,
                   ssvi_cmin, penalty, tvar_next_slice, p_scaler, butterfly_constraint, tol=10**-12):
    """
    z: ndarray
    k: ndarray
    iv: ndarray, target vol
    s: float
    f: float
    t: float, time to expiry in year
    atm: float, ATM forward vol
    x, x0: ndarray
    """

    w = np.ones(len(iv))
    w_pen = np.ones(len(iv))

    if ssvi_deltacutoff > 0:
        w = np.array(delta_grid(k, f, iv, t))
        w = (w > ssvi_deltacutoff) & (w < (1 - ssvi_deltacutoff))
        w = w.astype(float)
    if ssvi_weights == 'vega':
        w *= vega_grid(k, f, iv, t)
    if penalty == 'vega':
        w *= vega_grid(k, f, iv, t)
    w = w / np.sum(w)
    w_pen = w_pen / np.sum(w_pen)

    cons = [{'type': 'ineq', 'fun': ssvi_c_constraint, 'args': [ssvi_cmin]}]

    if butterfly_constraint:
        cons.append({'type': 'ineq', 'fun': ssvi_butterfly_arb, 'args': [np.square(atm)*t]})

    calibration_result = minimize(ssvi_vol_errors_pen, xn, args=(iv, z, k, atm, f, t, w, ssvi_target, penalty, tvar_next_slice, p_scaler, w_pen),
                                  method=optimizer1, constraints=cons, tol=tol)

    if calibration_result.success:
        method = optimizer1 + '_lastguess'
    else:
        calibration_result = minimize(ssvi_vol_errors_pen, xn, args=(iv, z, k, atm, f, t, w, ssvi_target, penalty, tvar_next_slice, p_scaler, w_pen),
                                  method=optimizer2, constraints=cons, tol=tol)
        if calibration_result.success:
            method = optimizer2 + '_lastguess'
        else:
            calibration_result = minimize(ssvi_vol_errors_pen, x0, args=(iv, z, k, atm, f, t, w, ssvi_target, penalty, tvar_next_slice, p_scaler, w_pen),
                                  method=optimizer1, constraints=cons, tol=tol)
            if calibration_result.success:
                method = optimizer1 + '_initialguess'
            else:
                calibration_result = minimize(ssvi_vol_errors_pen, x0, args=(iv, z, k, atm, f, t, w, ssvi_target, penalty, tvar_next_slice, p_scaler, w_pen),
                                  method=optimizer2, constraints=cons, tol=tol)
                if calibration_result.success:
                    method = optimizer2 + '_initialguess'
                else:
                    method = 'failed'

    return calibration_result, method


def ssvi_calib_pen(z_deltas, k, k_arr, vols, spot, F, e, vol_atf, params):

    ssvi_c2 = np.zeros([len(e)])
    ssvi_s2 = np.zeros([len(e)])
    vol_ssvi = np.zeros([len(e), len(k)])
    ssvi_params = np.zeros([len(e), 3])
    tvar_previous = np.zeros([len(k)])

    for expiry in reversed(range(len(e))):

        if expiry == len(e) - 1:

            penalty = 'none'

            ssvi, method = ssvi_calibrate_pen(z_deltas[expiry, :], k_arr, vols.iloc[:, expiry].values, spot,
                                        F.forward[expiry], e[expiry], vol_atf[expiry],
                                        params['OPTIMIZER_1'], params['OPTIMIZER_2'],
                                        [float(params['S2_GUESS']), float(params['C2_GUESS'])],
                                        [float(params['S2_GUESS']), float(params['C2_GUESS'])],
                                        params['TARGET'],
                                        params['WEIGHTS'], float(params['DELTA_CUTOFF']), 
                                        float(params['C2_MIN']),
                                        penalty, tvar_previous, float(params['PENALTY_SCALER']),
                                        params['BUTTERFLY_CONSTRAINT'])

            ssvi_s2[expiry], ssvi_c2[expiry] = ssvi.x

        else:

            penalty = params['PENALTY_MODE']

            guess = [ssvi_s2[expiry+1], ssvi_c2[expiry+1]]

            ssvi, method = ssvi_calibrate_pen(z_deltas[expiry, :], k_arr, vols.iloc[:, expiry].values, spot,
                                        F.forward[expiry], e[expiry], vol_atf[expiry],
                                        params['OPTIMIZER_1'], params['OPTIMIZER_2'],
                                        guess, guess, 
                                        params['TARGET'],
                                        params['WEIGHTS'], float(params['DELTA_CUTOFF']), 
                                        float(params['C2_MIN']),
                                        penalty, tvar_previous, float(params['PENALTY_SCALER']),
                                        params['BUTTERFLY_CONSTRAINT'])

            ssvi_s2[expiry], ssvi_c2[expiry] = ssvi.x

        vol_ssvi[expiry, :] = np.sqrt(ssvi_vol(vol_atf[expiry]**2, ssvi_s2[expiry], ssvi_c2[expiry], z_deltas[expiry, :]))
        ssvi_params[expiry, :] = [e[expiry], ssvi_s2[expiry], ssvi_c2[expiry]]

        previous_slice = interpolate.CubicSpline(k_arr, vol_ssvi[expiry, :], bc_type='natural')
        if expiry < len(e) - 1:
            tvar_previous = e[expiry] * np.square(previous_slice(k_arr * F.forward[expiry] / F.forward[expiry + 1]))
        else:
            tvar_previous = e[expiry+1] * (vol_ssvi[expiry, :]**2)

    return ssvi_s2, ssvi_c2, ssvi_params, vol_ssvi


def ssvi_calib_pen_v2(z_deltas, k, k_arr, vols, spot, F, e, vol_atf, params):

    ssvi_c2 = np.zeros([len(e)])
    ssvi_s2 = np.zeros([len(e)])
    vol_ssvi = np.zeros([len(e), len(k)])
    ssvi_params = np.zeros([len(e), 3])
    tvar_previous = np.zeros([len(k)])
    last_success = [float(params['S2_GUESS']), float(params['C2_GUESS'])]

    for i, t in enumerate(reversed(e)):

        if i == 0 or params['PENALTY_WEIGHTS'] == 'none':
            penalty = 'none'
        else:
            penalty = params['PENALTY_WEIGHTS']

        i = len(e) - i - 1
        ssvi, method = ssvi_calibrate_pen(z_deltas[i, :], k_arr, vols.iloc[:, i].values, spot, F.forward[i], e[i], vol_atf[i],
                                    params['OPTIMIZER_1'], params['OPTIMIZER_2'],
                                    [float(params['S2_GUESS']), float(params['C2_GUESS'])], last_success,
                                    params['TARGET'],
                                    params['WEIGHTS'], float(params['DELTA_CUTOFF']), 
                                    float(params['C2_MIN']),
                                    penalty, tvar_previous, float(params['PENALTY_SCALER']),
                                    params['BUTTERFLY_CONSTRAINT'])

        ssvi_s2[i], ssvi_c2[i] = ssvi.x
        last_success = ssvi.x

        vol_last_slice = np.sqrt(ssvi_vol(vol_atf[i]**2, ssvi_s2[i], ssvi_c2[i], z_deltas[i, :]))
        vol_last_slice = interpolate.CubicSpline(k_arr, vol_last_slice, bc_type='natural')
        if i > 0:
            tvar_previous = t * np.square(vol_last_slice(k_arr * F.forward[i] / F.forward[i - 1]))
    
        vol_ssvi[i, :] = np.sqrt(ssvi_vol(vol_atf[i]**2, ssvi_s2[i], ssvi_c2[i], z_deltas[i, :]))
        ssvi_params[i, :] = [e[i], ssvi_s2[i], ssvi_c2[i]]

    return ssvi_s2, ssvi_c2, ssvi_params, vol_ssvi.T

