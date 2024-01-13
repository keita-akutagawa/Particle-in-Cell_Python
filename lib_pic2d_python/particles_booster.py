import numpy as np


def buneman_boris_v(c, dt, q_list, m_list, E, B, v):

    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0)/c)**2)

    #TとSの設定
    T = (q_list/m_list) * dt * B / 2.0 / gamma
    S = 2.0 * T / (1.0 + np.linalg.norm(T, axis=0)**2)

    #時間発展
    v_minus = v + (q_list/m_list) * E * (dt/2)
    v_0 = v_minus + np.cross(v_minus, T, axis=0)
    v_plus = v_minus + np.cross(v_0, S, axis=0)
    v = v_plus + (q_list/m_list) * E * (dt/2.0)

    return v 


def buneman_boris_x(c, dt, v, x):

    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0)/c)**2)

    x = x + v * dt / gamma

    return x


def time_evolution_v(c, E, B, x, q_list, m_list, n_x, n_y, dx, dy, dt, v):
    
    E_tmp = E.copy()
    B_tmp = B.copy()

    #整数格子点上に再定義。特に磁場は平均の取り方に注意。
    E_tmp[0, :, :] = (E[0, :, :] + np.roll(E[0, :, :], 1, axis=0)) / 2.0
    E_tmp[1, :, :] = (E[1, :, :] + np.roll(E[1, :, :], 1, axis=1)) / 2.0
    B_tmp[0, :, :] = (B[0, :, :] + np.roll(B[0, :, :], 1, axis=1)) / 2.0
    B_tmp[1, :, :] = (B[1, :, :] + np.roll(B[1, :, :], 1, axis=0)) / 2.0
    B_tmp[2, :, :] = (B[2, :, :] + np.roll(B[2, :, :], 1, axis=0) + np.roll(B[2, :, :], 1, axis=1) + np.roll(B[2, :, :], [1, 1], axis=[0, 1])) / 4.0

    x_index = np.floor(x[0, :] / dx).astype(int)
    y_index = np.floor(x[1, :] / dy).astype(int)

    E_particle = np.zeros(x.shape)
    B_particle = np.zeros(x.shape)

    cx1 = x[0, :] / dx - x_index 
    cx2 = 1.0 - cx1
    cy1 = x[1, :] / dy - y_index  
    cy2 = 1.0 - cy1
    cx1 = cx1.reshape(-1, 1)
    cx2 = cx2.reshape(-1, 1)
    cy1 = cy1.reshape(-1, 1)
    cy2 = cy2.reshape(-1, 1)

    #電場
    E_particle[:, :] = (E_tmp[:, x_index, y_index].T * (cx2 * cy2) \
                     + E_tmp[:, (x_index+1)%n_x, y_index].T * (cx1 * cy2) \
                     + E_tmp[:, x_index, (y_index+1)%n_y].T * (cx2 * cy1) \
                     + E_tmp[:, (x_index+1)%n_x, (y_index+1)%n_y].T * (cx1 * cy1)
                    ).T
    
    #磁場
    B_particle[:, :] = (B_tmp[:, x_index, y_index].T * (cx2 * cy2) \
                     + B_tmp[:, (x_index+1)%n_x, y_index].T * (cx1 * cy2) \
                     + B_tmp[:, x_index, (y_index+1)%n_y].T * (cx2 * cy1) \
                     + B_tmp[:, (x_index+1)%n_x, (y_index+1)%n_y].T * (cx1 * cy1)
                    ).T
  
    v = buneman_boris_v(c, dt, q_list, m_list, E_particle, B_particle, v)

    return v


def time_evolution_x(c, dt, v, x):
    
    x = buneman_boris_x(c, dt, v, x)

    return x 


