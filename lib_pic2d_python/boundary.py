import numpy as np


def periodic_condition_x(x, x_max):

    over_xmax_index = np.where(x[0, :] >= x_max)[0]
    x[0, over_xmax_index] = 1e-10

    under_x0_index = np.where(x[0, :] <= 0.0)[0]
    x[0, under_x0_index] = x_max - 1e-10

    return x 


def periodic_condition_y(x, y_max):

    over_ymax_index = np.where(x[1, :] >= y_max)[0]
    x[1, over_ymax_index] = 1e-10

    under_y0_index = np.where(x[1, :] <= 0.0)[0]
    x[1, under_y0_index] = y_max - 1e-10

    return x



def refrective_condition_x(v, x, x_max):

    over_xmax_index = np.where(x[0, :] >= x_max)[0]
    x[0, over_xmax_index] = x_max - 1e-10
    v[0, over_xmax_index] = -v[0, over_xmax_index]

    under_x0_index = np.where(x[0, :] <= 0.0)[0]
    x[0, under_x0_index] = 1e-10
    v[0, under_x0_index] = -v[0, under_x0_index]

    return v, x


def refrective_condition_y(v, x, y_max):

    over_ymax_index = np.where(x[1, :] >= y_max)[0]
    x[1, over_ymax_index] = y_max - 1e-10
    v[1, over_ymax_index] = -v[1, over_ymax_index]
    
    under_y0_index = np.where(x[1, :] <= 0.0)[0]
    x[1, under_y0_index] = 1e-10
    v[1, under_y0_index] = -v[1, under_y0_index]

    return v, x
