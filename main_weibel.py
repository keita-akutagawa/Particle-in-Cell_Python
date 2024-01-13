import os 
import sys
sys.path.append(os.pardir)
import numpy as np
import time
from memory_profiler import profile

from lib_pic2d.boundary import *
from lib_pic2d.particles_booster import *
from lib_pic2d.maxwell import *
from lib_pic2d.others import *
from const_weibel import *

current_path = os.getcwd()
if not os.path.exists(current_path + '/results_weibel'):
    os.mkdir('results_weibel')

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

print(f"total number of particles is {n_particle}.")


#print(x.__sizeof__()/1024**3 * 34 + B.__sizeof__()/1024**3 * 20)

#STEP1
rho = get_rho(rho, cx1, cx2, cy1, cy2, dx, dy, 
            index_one_array, n_x, n_y, q_list, 
            x, x_index, y, y_index)
E = solve_poisson_not_periodic(E, dx, dy, epsilon0, n_x, n_y, phi, rho)


for k in range(step+1):
    #STEP2
    B = time_evolution_B(B, dt/2, dx, dy, E)
    #STEP3
    B_tmp = B.copy()
    E_tmp = E.copy()
    B_particle, E_particle = get_partice_field(B_particle, B_tmp, 
                                               cx1, cx2, cy1, cy2,
                                               dx, dy, E_particle, E_tmp, 
                                               n_x, n_y, x, y)
    vx, vy, vz = buneman_boris_v(vx, vy, vz, B_particle, c, dt, E_particle, gamma,
                                 m_list, q_list, S, T, v_minus, v_plus, v_0)
    #STEP4
    x, y, z = buneman_boris_x(x, y, z, c, dt/2, gamma, vx, vy, vz)
    x = periodic_condition_x(x, x_max)
    y = periodic_condition_y(y, y_max)
    #STEP5
    current = get_current_density(current, c, cx1, cx2, cy1, cy2, 
                                  dx, dy, gamma, q_list, n_x, n_y, 
                                  vx, vy, vz, x, y, z, x_index, y_index)
    #STEP6
    B = time_evolution_B(B, dt/2, dx, dy, E)
    #STEP7
    if k % 10 == 0:
        with open(filename, 'a') as f:
            f.write(f'{int(k*dt)} step done...\n')
        print(f'{int(k*dt)} step done...\n')
        rho = get_rho(rho, cx1, cx2, cy1, cy2, dx, dy, 
                      index_one_array, n_x, n_y, q_list, 
                      x, x_index, y, y_index)
        E = E_modification(E, delta_E, dx, dy, epsilon0, n_x, n_y, phi, rho)
    E = time_evolution_E(E, B, c, current, dt, dx, dy, epsilon0)
    #STEP8
    x, y, z = buneman_boris_x(x, y, z, c, dt/2, gamma, vx, vy, vz)
    x = periodic_condition_x(x, x_max)
    y = periodic_condition_y(y, y_max)

    if k % 100 == 0:
        k1 = k // 100
        KE = np.sum(1/2 * m_list * (vx**2 + vy**2 + vz**2))
        np.save(f'results_weibel/results_2pic_weibel_xv_{k1}.npy', np.vstack([x, y, z, vx, vy, vz]))
        np.save(f'results_weibel/results_2pic_weibel_E_{k1}.npy', E)
        np.save(f'results_weibel/results_2pic_weibel_B_{k1}.npy', B)
        np.save(f'results_weibel/results_2pic_weibel_current_{k1}.npy', current)
        np.save(f'results_weibel/results_2pic_weibel_KE_{k1}.npy', KE)


sys.exit()

