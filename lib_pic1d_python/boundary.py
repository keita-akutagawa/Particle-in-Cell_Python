import numpy as np


def periodic_condition_x(x_max, x):
    over_xmax_index = np.where(x[0, :] >= x_max)[0]
    x[0, over_xmax_index] = 1e-10
    under_x0_index = np.where(x[0, :] <= 0.0)[0]
    x[0, under_x0_index] = x_max - 1e-10

    return x 

