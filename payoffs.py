import numpy as np
import pandas as pd
from scipy.stats import norm


def Call(S, K):
    return max(S - K, 0)


def Put(S, K):
    return max(K - S, 0)


def Call_BS(S, K, T, r, v):
	d1 = (1 / v * np.sqrt(T)) * (np.log(S / K) + (r + v * v / 2) * T)
	d2 = d1 - v * np.sqrt(T)
	call = S * norm.cdf(d1) - K * norm.cdf(d2) * np.exp(-r * T)
	return call


def Put_BS(S, K, T, r, v):
	d1 = (1/v*np.sqrt(T)) * (np.log(S/K) + (r + v*v/2)*T)
	d2 = d1 - v*np.sqrt(T)
	put = K*norm.cdf(-d2)*np.exp(-r*T) - S*norm.cdf(-d1)
	return put


def ABRC(K, H, H_AC, Y, T_E, T_O, N, v, S, minS):
	"""
	face_value (float): 1
	Y: coupon (float)
	"""
	if (S >= H) and (minS <= H_AC):
		payoff = 1 + Y - (1/K)*Put(S, K)
	elif (S < H) and (minS <= H_AC):
		payoff = 1 - (1/K)*Put(S, K)
	elif (S >= H) and (minS > H_AC):
		payoff = 1 + Y
	else:
		payoff = 0

	return payoff
