import numpy as np


def get_rho(q_list, x, n_x, dx):
    x_index = np.floor(x[0, :] / dx).astype(np.int64)
    rho = np.zeros([n_x])

    cx1 = (x[0, :] - x_index*dx)/dx  
    cx2 = ((x_index+1)*dx - x[0, :])/dx 
    index_one_array = x_index

    rho += np.bincount(index_one_array, 
                       weights=q_list * cx2,
                       minlength=n_x
                      )
    rho += np.roll(np.bincount(index_one_array, 
                               weights=q_list * cx1,
                               minlength=n_x
                              ), 1, axis=0)
    
    return rho


def solve_poisson_not_periodic(rho, n_x, epsilon0, dx, E):
    phi = np.zeros(n_x)
    for k in range(10000):
        phi = (rho/epsilon0 * dx**2 + np.roll(phi, -1) + np.roll(phi, 1)) / 2.0
    E[0, :] = -(np.roll(phi, -1) - phi) / dx
    return E


def E_modification(q_list, x, n_x, dx, epsilon0, E):
    rho = get_rho(q_list, x, n_x, dx)
    div_E = (E[0, :] - np.roll(E[0, :], 1)) / dx
    delta_rho = rho - div_E

    delta_E = np.zeros(E.shape)
    delta_E = solve_poisson_not_periodic(delta_rho, n_x, epsilon0, dx, delta_E)
    E += delta_E
    
    return E 


def get_current_density(c, q_list, v, x, n_x, dx, dt):
    x_index = np.floor(x[0, :] / dx).astype(int)
    x_index_half = np.floor((x[0, :] - 1/2*dx) / dx).astype(int)
    x_index_half_minus = np.where(x_index_half == -1)
    x_index_half[x_index_half == -1] = n_x-1

    current = np.zeros([3, n_x])
    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0)/c)**2)


    cx1 = (x[0, :] - (x_index_half + 1/2)*dx)/dx  
    cx2 = ((x_index_half + 3/2)*dx - x[0, :])/dx 
    cx1[x_index_half_minus] = (x[0, x_index_half_minus] - (-1/2)*dx)/dx  
    cx2[x_index_half_minus] = ((1/2)*dx - x[0, x_index_half_minus])/dx 
    index_one_array = x_index_half

    current[0, :] += np.bincount(index_one_array, 
                                 weights=q_list * v[0, :]/gamma * cx2, 
                                 minlength=n_x
                                )
    current[0, :] += np.roll(np.bincount(index_one_array, 
                                         weights=q_list * v[0, :]/gamma * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)


    cx1 = (x[0, :] - x_index*dx)/dx  
    cx2 = ((x_index+1)*dx - x[0, :])/dx 
    index_one_array = x_index
    
    current[1, :] += np.bincount(index_one_array, 
                                    weights=q_list * v[1, :]/gamma * cx2, 
                                    minlength=n_x
                                    )
    current[1, :] += np.roll(np.bincount(index_one_array, 
                                            weights=q_list * v[1, :]/gamma * cx1, 
                                            minlength=n_x
                                            ), 1, axis=0)

    current[2, :] += np.bincount(index_one_array, 
                                 weights=q_list * v[2, :]/gamma * cx2, 
                                 minlength=n_x
                                )
    current[2, :] += np.roll(np.bincount(index_one_array, 
                                         weights=q_list * v[2, :]/gamma * cx1, 
                                         minlength=n_x
                                        ), 1, axis=0)
    
    return current


