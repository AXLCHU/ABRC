import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
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
