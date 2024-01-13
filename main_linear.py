import os 
import sys
sys.path.append(os.pardir)
import numpy as np

from lib_pic2d.boundary import *
from lib_pic2d.particles_booster import *
from lib_pic2d.maxwell import *
from lib_pic2d.others import *
from const_linear import *



current_path = os.getcwd()
if not os.path.exists(current_path + '/results_linear'):
    os.mkdir('results_linear')

filename = 'progress.txt'
f = open(filename, 'w')

if c * dt > dx:
    print(f"You had better change some parameters! \nCFL condition is not satisfied \n c * dt = {c * dt} > dx = {dx} \n")
else:
    print(f'c * dt = {c * dt} < dx = {dx} \n')

if omega_pe * dt > 0.2:
    print(f"You had better change some parameters! \n$\omega$_pe * dt = {omega_pe * dt} > 0.1 \n")
if dx != 1.0:
    print(f"You had better change some parameters! \ndebye length = {debye_length} should be equal to grid size = {dx} \n")

print(f"total number of particles is {n_plus + n_minus}.")


#STEP1
rho = get_rho(q_list, x, n_x, n_y, dx, dy)
E = solve_poisson_not_periodic(rho, n_x, n_y, dx, dy, epsilon0, E)   

for k in range(step+1):
    #STEP2
    B = time_evolution_B(E, dx, dy, dt/2, B)   
    #STEP3
    v = time_evolution_v(c, E, B, x, q_list, m_list, n_x, n_y, dx, dy, dt, v)
    #STEP4
    x = time_evolution_x(c, dt/2, v, x)
    x = periodic_condition_x(x_max, y_max, x)
    #STEP5
    current = get_current_density(c, q_list, v, x, n_x, n_y, dx, dy, current)
    #STEP6
    B = time_evolution_B(E, dx, dy, dt/2, B) 
    #STEP7
    if k % 100 == 0:
        with open(filename, 'a') as f:
            f.write(f'{int(k*dt)} step done...\n')
        E = E_modification(q_list, x, n_x, n_y, dx, dy, epsilon0, E)
    E = time_evolution_E(B, current, c, epsilon0, dx, dy, dt, E)
    #STEP8
    x = time_evolution_x(c, dt/2, v, x)
    x = periodic_condition_x(x_max, y_max, x)

    if k % 1000 == 0:
        k1 = k // 1000
        KE = np.sum(1/2 * m_list * np.linalg.norm(v, axis=0)**2)
        np.save(f'results_linear/results_2pic_linear_xv_{k1}.npy', np.concatenate([x, v]))
        np.save(f'results_linear/results_2pic_linear_E_{k1}.npy', E)
        np.save(f'results_linear/results_2pic_linear_B_{k1}.npy', B)
        np.save(f'results_linear/results_2pic_linear_current_{k1}.npy', current)
        np.save(f'results_linear/results_2pic_linear_KE_{k1}.npy', KE)



sys.exit()
