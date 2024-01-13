import os 
import sys
sys.path.append(os.pardir)
import time
import numpy as np

import os

from lib_pic1d.boundary import *
from lib_pic1d.particles_booster import *
from lib_pic1d.maxwell import *
from lib_pic1d.others import *
from const_two_stream_electron import *



current_path = os.getcwd()
if not os.path.exists(current_path + '/results_1pic_two_stream_electron'):
    os.mkdir('results_1pic_two_stream_electron')

filename = 'progress.txt'
f = open(filename, 'w')


start = time.time()

#STEP1
rho = get_rho(q_list, x, n_x, dx)
E = solve_poisson_not_periodic(rho, n_x, epsilon0, dx, E)  

for k in range(step+1):
    #STEP2
    B = time_evolution_B(E, dx, dt/2, B)   
    #STEP3
    v = time_evolution_v(c, E, B, x, q_list, m_list, n_x, dx, dt, v)
    #STEP4
    x = time_evolution_x(c, dt/2, v, x)
    x = periodic_condition_x(x_max, x)
    #STEP5
    current = get_current_density(c, q_list, v, x, n_x, dx, dt)
    #STEP6
    B = time_evolution_B(E, dx, dt/2, B) 
    #STEP7
    if k % 100 == 0:
        with open(filename, 'a') as f:
            f.write(f'{int(k*dt)} step done...\n')
            print(f'{int(k*dt)} step done...\n')
        E = E_modification(q_list, x, n_x, dx, epsilon0, E)
    E = time_evolution_E(B, current, c, epsilon0, dx, dt, E)
    #STEP8
    x = time_evolution_x(c, dt/2, v, x)
    x = periodic_condition_x(x_max, x)

    #if k % 100 == 0:
    #    k1 = k // 100
    #    KE = np.sum(1/2 * m_list * np.linalg.norm(v, axis=0)**2)
    #    np.save(f'results_1pic_two_stream_electron/results_1pic_two_stream_electron_xv_{k1}.npy', np.concatenate([x, v]))
    #    np.save(f'results_1pic_two_stream_electron/results_1pic_two_stream_electron_E_{k1}.npy', E)
    #    np.save(f'results_1pic_two_stream_electron/results_1pic_two_stream_electron_B_{k1}.npy', B)
    #    np.save(f'results_1pic_two_stream_electron/results_1pic_two_stream_electron_current_{k1}.npy', current)
    #    np.save(f'results_1pic_two_stream_electron/results_1pic_two_stream_electron_KE_{k1}.npy', KE)


finish = time.time()

filename = 'time.txt'
f = open(filename, 'w')

with open(filename, 'a') as f:
    f.write(f"Time is {round(finish - start, 3)}s")

sys.exit()
