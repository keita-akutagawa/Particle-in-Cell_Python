import os 
import sys
sys.path.append(os.pardir)
import numpy as np

from lib_pic2d.boundary import *
from lib_pic2d.particles_booster import *
from lib_pic2d.maxwell import *
from lib_pic2d.others import *
from const_plasmoid import *



current_path = os.getcwd()
if not os.path.exists(current_path + '/results_plasmoid'):
    os.mkdir('results_plasmoid')

filename = 'progress.txt'
f = open(filename, 'w')



#STEP1
rho = get_rho(q_list, x, n_x, n_y, dx, dy)
E = solve_poisson_not_periodic(rho, n_x, n_y, dx, dy, epsilon0, E)  

for k in range(step+1):
    if k == 0:
        B += delta_B
    #STEP2
    B = time_evolution_B(E, dx, dy, dt/2, B) 
    B[1, :, [0, 1]] = 0.0
    B[1, :, -1] = 0.0
    B[0, :, 0] = -mu_0 * current[2, :, 0]
    B[2, :, 0] = mu_0 * current[0, :, 0]
    B[0, :, -1] = mu_0 * current[2, :, -1]
    B[2, :, -1] = -mu_0 * current[0, :, -1]
    #STEP3
    v = time_evolution_v(c, E, B, x, q_list, m_list, n_x, n_y, dx, dy, dt, v)
    #STEP4
    x = time_evolution_x(c, dt/2, v, x)
    x = periodic_condition_x_x(x_max, x)
    v, x = refrective_condition_x_y(y_max, v, x)
    #STEP5
    current = get_current_density(c, q_list, v, x, n_x, n_y, dx, dy, current)
    #STEP6
    B = time_evolution_B(E, dx, dy, dt/2, B) 
    B[1, :, [0, 1]] = 0.0
    B[1, :, -1] = 0.0
    B[0, :, 0] = -mu_0 * current[2, :, 0]
    B[2, :, 0] = mu_0 * current[0, :, 0]
    B[0, :, -1] = mu_0 * current[2, :, -1]
    B[2, :, -1] = -mu_0 * current[0, :, -1]
    #STEP7
    if k % 10 == 0:
        with open(filename, 'a') as f:
            f.write(f'{int(k*dt)} step done...\n')
        E = E_modification(q_list, x, n_x, n_y, dx, dy, epsilon0, E)
    E = time_evolution_E(B, current, c, epsilon0, dx, dy, dt, E)
    rho = get_rho(q_list, x, n_x, n_y, dx, dy)
    E[1, :, 0] = rho[:, 0] / epsilon0
    E[1, :, -1] = -rho[:, -1] / epsilon0
    E[0, :, [0, 1]] = 0.0
    E[2, :, [0, 1]] = 0.0
    E[0, :, -1] = 0.0
    E[2, :, -1] = 0.0
    #STEP8
    x = time_evolution_x(c, dt/2, v, x)
    x = periodic_condition_x_x(x_max, x)
    v, x = refrective_condition_x_y(y_max, v, x)

    if k % 500 == 0:
        k1 = k // 500
        KE = np.sum(1/2 * m_list * np.linalg.norm(v, axis=0)**2)
        np.save(f'results_plasmoid/results_plasmoid_xv_{k1}.npy', np.concatenate([x, v]))
        np.save(f'results_plasmoid/results_plasmoid_E_{k1}.npy', E)
        np.save(f'results_plasmoid/results_plasmoid_B_{k1}.npy', B)
        np.save(f'results_plasmoid/results_plasmoid_current_{k1}.npy', current)
        np.save(f'results_plasmoid/results_plasmoid_KE_{k1}.npy', KE)



sys.exit()

