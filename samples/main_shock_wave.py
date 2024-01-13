import os 
import sys
sys.path.append(os.pardir)
import numpy as np

from lib_pic1d.boundary import *
from lib_pic1d.particles_booster import *
from lib_pic1d.maxwell import *
from lib_pic1d.others import *
from const_shock_wave import *



current_path = os.getcwd()
if not os.path.exists(current_path + '/results_1pic_shock_wave'):
    os.mkdir('results_1pic_shock_wave')

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

print(f"total number of particles is {n_plus_upstream + n_plus_downstream + n_minus_upstream + n_minus_downstream}.")



#STEP1
rho = get_rho(q_list, x, n_x, dx)
E = solve_poisson_not_periodic(rho, n_x, epsilon0, dx, E)  

for k in range(step+1):
    #STEP2
    B = time_evolution_B(E, dx, dt/2, B)   
    B[:, -1] = B[:, -2]
    B[:, 0] = B[:, 1]
    #STEP3
    v = time_evolution_v(c, E, B, x, q_list, m_list, n_x, dx, dt, v)
    #STEP4
    x = time_evolution_x(c, dt/2, v, x)
    #v, x = refrective_condition_x(x_max, v, x)
    x = periodic_condition_x(x_max, x)
    #STEP5
    current = get_current_density(c, q_list, v, x, n_x, dx, dt)
    #STEP6
    B = time_evolution_B(E, dx, dt/2, B) 
    B[:, -1] = B[:, -2]
    B[:, 0] = B[:, 1]
    #STEP7
    if k % 100 == 0:
        with open(filename, 'a') as f:
            f.write(f'{int(k*dt)} step done...\n')
        E = E_modification(q_list, x, n_x, dx, epsilon0, E)
    E = time_evolution_E(B, current, c, epsilon0, dx, dt, E)
    E[:, -1] = E[:, -2]
    E[:, 0] = E[:, 1]
    #STEP8
    x = time_evolution_x(c, dt/2, v, x)
    #v, x = refrective_condition_x(x_max, v, x)
    x = periodic_condition_x(x_max, x)

    if k % 100 == 0:
        k1 = k // 100
        KE = np.sum(1/2 * m_list * np.linalg.norm(v, axis=0)**2)
        np.save(f'shock_wave/results_1pic_shock_wave_xv_{k1}.npy', np.concatenate([x, v]))
        np.save(f'shock_wave/results_1pic_shock_wave_E_{k1}.npy', E)
        np.save(f'shock_wave/results_1pic_shock_wave_B_{k1}.npy', B)
        np.save(f'shock_wave/results_1pic_shock_wave_current_{k1}.npy', current)
        np.save(f'shock_wave/results_1pic_shock_wave_KE_{k1}.npy', KE)


sys.exit()
