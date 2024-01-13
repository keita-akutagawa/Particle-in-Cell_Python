import numpy as np

def buneman_boris_v(c, dt, q_list, m_list, E, B, v):
    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0)/c)**2)
    #TとSの設定
    T = (q_list/m_list) * dt * B / 2.0 / gamma
    S = 2.0 * T / (1.0 + np.linalg.norm(T, axis=0)**2)
    #時間発展
    v_minus = v + (q_list/m_list) * E * (dt/2.0)
    v_0 = v_minus + np.cross(v_minus, T, axis=0)
    v_plus = v_minus + np.cross(v_0, S, axis=0)
    v = v_plus + (q_list/m_list) * E * (dt/2.0)
    return v 


def buneman_boris_x(c, dt, v, x):
    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0)/c)**2)
    x = x + v * dt / gamma
    return x



def time_evolution_v(c, E, B, x, q_list, m_list, n_x, dx, dt, v):
    E_tmp = E.copy()
    B_tmp = B.copy()
    E_tmp[0, :] = (E[0, :] + np.roll(E[0, :], 1, axis=0)) / 2.0
    B_tmp[0, :] = (B[0, :] + np.roll(B[0, :], -1, axis=0)) / 2.0
    x_index = np.floor(x[0, :] / dx).astype(int)
    x_index_half = np.floor((x[0, :] - 1/2*dx) / dx).astype(int)
    x_index_half_minus = np.where(x_index_half == -1)
    x_index_half[x_index_half == -1] = n_x-1
    E_particle = np.zeros(x.shape)
    B_particle = np.zeros(x.shape)

    #電場
    cx1 = (x[0, :] - x_index*dx)/dx  
    cx2 = ((x_index+1)*dx - x[0, :])/dx 
    cx1 = cx1.reshape(-1, 1)
    cx2 = cx2.reshape(-1, 1)
    E_particle[:, :] = (E_tmp[:, x_index].T * cx2 \
                     + E_tmp[:, (x_index+1)%n_x].T * cx1 \
                    ).T
    
    #磁場
    cx1 = (x[0, :] - (x_index_half + 1/2)*dx)/dx  
    cx2 = ((x_index_half + 3/2)*dx - x[0, :])/dx 
    cx1[x_index_half_minus] = (x[0, x_index_half_minus] - (-1/2)*dx)/dx  
    cx2[x_index_half_minus] = ((1/2)*dx - x[0, x_index_half_minus])/dx 
    cx1 = cx1.reshape(-1, 1)
    cx2 = cx2.reshape(-1, 1)
    B_particle[:, :] = (B_tmp[:, x_index_half].T * cx2 \
                     + B_tmp[:, (x_index_half+1)%n_x].T * cx1 \
                    ).T
  
    v = buneman_boris_v(c, dt, q_list, m_list, E_particle, B_particle, v)

    return v


def time_evolution_x(c, dt, v, x):
    x = buneman_boris_x(c, dt, v, x)

    return x 

