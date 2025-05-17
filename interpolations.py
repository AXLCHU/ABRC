import numpy as np
from scipy import interpolate


def cubic_interpolation(x_new, x, y):
    """
    x_new: list
    x: list
    y: list
    """
    f = interpolate.CubicSpline(x, y, bc_type='natural')
    new_y = f(x_new)
    for k in range(len(x_new)):
        if x_new[k] <= x[0]:
            new_y[k] = y[0]
        elif x_new[k] >= x[len(x)-1]:
            new_y[k] = y[len(x)-1]
    return new_y


def linear_interpolation(x_new, x, y):

    if x_new <= x[0]:
        new_y = y[0]
    elif x_new >= x[-1]:
        new_y = y[-1]
    else:
        n = 0
        while x_new > x[n]:
            n += 1
            new_y = y[n] * (x_new - x[n-1]) / (x[n] - x[n-1]) + y[n-1] * (x[n] - x_new) / (x[n] - x[n-1])

    return new_y