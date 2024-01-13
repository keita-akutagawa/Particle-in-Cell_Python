import numpy as np


def get_rho(q_list, x, n_x, n_y, dx, dy):

    x_index = np.floor(x[0, :] / dx).astype(np.int64)
    y_index = np.floor(x[1, :] / dy).astype(np.int64)

    rho = np.zeros([n_x, n_y])

    cx1 = x[0, :] / dx - x_index 
    cx2 = 1.0 - cx1
    cy1 = x[1, :] / dy - y_index  
    cy2 = 1.0 - cy1

    index_one_array = x_index * n_y + y_index

    rho[:, :] += np.bincount(index_one_array, 
                            weights=q_list * cx2 * cy2, 
                            minlength=n_x*n_y
                            ).reshape(n_x, n_y)
    rho[:, :] += np.roll(np.bincount(index_one_array, 
                                    weights=q_list * cx1 * cy2, 
                                    minlength=n_x*n_y
                                    ).reshape(n_x, n_y), 1, axis=0)
    rho[:, :] += np.roll(np.bincount(index_one_array, 
                                    weights=q_list * cx2 * cy1, 
                                    minlength=n_x*n_y
                                    ).reshape(n_x, n_y), 1, axis=1)
    rho[:, :] += np.roll(np.bincount(index_one_array, 
                                    weights=q_list * cx1 * cy1, 
                                    minlength=n_x*n_y
                                    ).reshape(n_x, n_y), [1, 1], axis=[0, 1])
    
    return rho


def solve_poisson_not_periodic(rho, n_x, n_y, dx, dy, epsilon0, E):

    phi = np.zeros([n_x, n_y])

    for k in range(10000):
        phi += (((np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0))/dx**2
            +(np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1))/dy**2
            + rho/epsilon0) / (2 * (1/dx**2 + 1/dy**2)) - phi) * 1.0

    E[0] = -(np.roll(phi, -1, axis=0) - phi) / dx
    E[1] = -(np.roll(phi, -1, axis=1) - phi) / dy

    return E


def solve_poisson_refrective_wall(rho, n_x, n_y, dx, dy, epsilon0, E):

    phi = np.zeros([n_x, n_y])

    for k in range(10000):
        phi += (((np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0))/dx**2
            +(np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1))/dy**2
            + rho/epsilon0) / (2 * (1/dx**2 + 1/dy**2)) - phi) * 1.0
        phi[0, :] = 0.0
        phi[-1, :] = 0.0
        phi[:, 0] = 0.0
        phi[:, -1] = 0.0

    E[0] = -(np.roll(phi, -1, axis=0) - phi) / dx
    E[1] = -(np.roll(phi, -1, axis=1) - phi) / dy

    return E



def E_modification(q_list, x, n_x, n_y, dx, dy, epsilon0, E):

    rho = get_rho(q_list, x, n_x, n_y, dx, dy)
    div_E = (E[0, :, :] - np.roll(E[0, :, :], 1, axis=0)) / dx \
          + (E[1, :, :] - np.roll(E[1, :, :], 1, axis=1)) / dy 
    delta_rho = rho - div_E

    delta_E = np.zeros(E.shape)
    delta_E = solve_poisson_not_periodic(delta_rho, n_x, n_y, dx, dy, epsilon0, delta_E)

    E += delta_E
    
    return E 



def current_component(current_comp, cx1, cx2, cy1, cy2, 
                      gamma, index_one_array, n_x, n_y, 
                      q_list, v_comp):

    current_comp += np.bincount(index_one_array, 
                                weights=q_list * v_comp/gamma * cx2 * cy2, 
                                minlength=n_x*n_y
                                ).reshape(n_x, n_y)
    current_comp += np.roll(np.bincount(index_one_array, 
                                        weights=q_list * v_comp/gamma * cx1 * cy2, 
                                        minlength=n_x*n_y
                                        ).reshape(n_x, n_y), 1, axis=0)
    current_comp += np.roll(np.bincount(index_one_array, 
                                        weights=q_list * v_comp/gamma * cx2 * cy1, 
                                        minlength=n_x*n_y
                                        ).reshape(n_x, n_y), 1, axis=1)
    current_comp += np.roll(np.bincount(index_one_array, 
                                        weights=q_list * v_comp/gamma * cx1 * cy1, 
                                        minlength=n_x*n_y
                                        ).reshape(n_x, n_y), [1, 1], axis=[0, 1])
    
    return current_comp


def get_current_density(c, q_list, v, x, n_x, n_y, dx, dy, current):

    x_index = np.floor(x[0, :] / dx).astype(int)
    y_index = np.floor(x[1, :] / dy).astype(int)

    gamma = np.sqrt(1.0 + (np.linalg.norm(v, axis=0)/c)**2)

    cx1 = x[0, :] / dx - x_index 
    cx2 = 1.0 - cx1
    cy1 = x[1, :] / dy - y_index  
    cy2 = 1.0 - cy1

    current = np.zeros(current.shape)

    index_one_array = x_index * n_y + y_index

    current[0, :, :] = current_component(current[0, :, :], cx1, cx2, cy1, cy2, 
                                         gamma, index_one_array, n_x, n_y, 
                                         q_list, v[0, :])
    current[1, :, :] = current_component(current[1, :, :], cx1, cx2, cy1, cy2, 
                                         gamma, index_one_array, n_x, n_y, 
                                         q_list, v[1, :])
    current[2, :, :] = current_component(current[2, :, :], cx1, cx2, cy1, cy2, 
                                         gamma, index_one_array, n_x, n_y, 
                                         q_list, v[2, :])
    
    current[0, :, :] = (current[0, :, :] + np.roll(current[0, :, :], -1, axis=0)) / 2.0
    current[1, :, :] = (current[1, :, :] + np.roll(current[1, :, :], -1, axis=1)) / 2.0
    
    return current




