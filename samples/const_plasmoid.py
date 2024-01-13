import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1/25
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 0.2
r_q = 1.0
n_e = 10 #ここは手動で調整すること
B0 = np.sqrt(n_e) / 1.5
n_i = int(n_e / r_q)
T_i  = (B0**2 / 2.0 / mu_0) / (n_i + n_e * t_r)
T_e = T_i * t_r
q_unit = np.sqrt(epsilon0 * T_e / n_e)
q_electron = -1 * q_unit
q_ion = r_q * q_unit
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
ion_inertial_length = c / omega_pi
sheat_thickness = 0.5 * ion_inertial_length
v_electron = np.array([0.0, 0.0, c * debye_length / sheat_thickness * np.sqrt(2 / (1.0 + 1/t_r))])
v_ion = -v_electron / t_r
v_thermal_electron = np.sqrt(T_e / m_electron)
v_thermal_ion = np.sqrt(T_i / m_ion)

dx = debye_length
dy = debye_length
n_x = 2048
n_y = 256
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
dt = 1.0
step = 20000
t_max = step * dt


E = np.zeros([3, n_x, n_y])
B = np.zeros([3, n_x, n_y])
current = np.zeros([3, n_x, n_y])
for j in range(n_y):
    B[0, :, j] = B0 * np.tanh((y_coordinate[j] - y_max/2) / sheat_thickness)

reconnection_ratio = 0.1
num_of_plasmoid = 8
delta_B = np.zeros([3, n_x, n_y])
X, Y = np.meshgrid(x_coordinate, y_coordinate)
for i in range(num_of_plasmoid):
    delta_B[0, :, :] += -np.array(reconnection_ratio * B0 * (Y - y_max/2) / sheat_thickness \
                    * np.exp(-((X - x_max/num_of_plasmoid*(i+1/2))**2 + (Y - y_max/2)**2) / ((2.0 * sheat_thickness)**2))).T
    delta_B[1, :, :] += np.array(reconnection_ratio * B0 * (X - x_max/num_of_plasmoid*(i+1/2)) / sheat_thickness \
                    * np.exp(-((X - x_max/num_of_plasmoid*(i+1/2))**2 + (Y - y_max/2)**2) / ((2.0 * sheat_thickness)**2))).T

n_plus = int(n_x * n_i * 2.0 * sheat_thickness)
n_minus = int(n_plus * abs(q_ion / q_electron))
n_plus_background = int(n_x * 0.2 * n_i * (y_max - 2.0 * sheat_thickness))
n_minus_background = int(n_x * 0.2 * n_e * (y_max - 2.0 * sheat_thickness))
x = np.zeros([3, n_plus + n_plus_background + n_minus + n_minus_background])
v = np.zeros([3, n_plus + n_plus_background + n_minus + n_minus_background])
print(f"total number of particles is {n_plus + n_plus_background + n_minus + n_minus_background}.")

np.random.RandomState(1)
x_start_plus = np.random.rand(n_plus) * x_max
x_start_plus_background = np.random.rand(n_plus_background) * x_max
x_start_minus = np.random.rand(n_minus) * x_max
x_start_minus_background = np.random.rand(n_minus_background) * x_max
y_start_plus = np.array(y_max/2 + sheat_thickness * np.arctanh(2.0 * np.random.rand(n_plus) - 1.0))
y_start_plus[y_start_plus > y_max] = y_max/2
y_start_plus[y_start_plus < 0.0] = y_max/2
y_start_plus_background = np.zeros(n_plus_background)
for i in range(n_plus_background):
    while True:
        rand = np.random.rand(1) * y_max 
        rand_pn = np.random.rand(1)
        if rand_pn < (1.0 - 1.0/np.cosh((rand - y_max/2)/sheat_thickness)):
            y_start_plus_background[i] = rand
            break
y_start_minus = np.array(y_max/2 + sheat_thickness * np.arctanh(2.0 * np.random.rand(n_minus) - 1.0))
y_start_minus[y_start_minus > y_max] = y_max/2
y_start_minus[y_start_minus < 0.0] = y_max/2
y_start_minus_background = np.zeros(n_minus_background)
for i in range(n_minus_background):
    while True:
        rand = np.random.rand(1) * y_max 
        rand_pn = np.random.rand(1)
        if rand_pn < (1.0 - 1.0/np.cosh((rand - y_max/2)/sheat_thickness)):
            y_start_minus_background[i] = rand
            break
x[0, :] = np.concatenate([x_start_plus, x_start_plus_background, x_start_minus, x_start_minus_background])
x[1, :] = np.concatenate([y_start_plus, y_start_plus_background, y_start_minus, y_start_minus_background])
v[0, :n_plus] = np.array(stats.norm.rvs(v_ion[0], v_thermal_ion, size=n_plus))
v[0, n_plus:n_plus+n_plus_background] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_plus_background))
v[0, n_plus+n_plus_background:n_plus+n_plus_background+n_minus] = np.array(stats.norm.rvs(v_electron[0], v_thermal_electron, size=n_minus))
v[0, n_plus+n_plus_background+n_minus:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_minus_background))
v[1, :n_plus] = np.array(stats.norm.rvs(v_ion[1], v_thermal_ion, size=n_plus))
v[1, n_plus:n_plus+n_plus_background] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_plus_background))
v[1, n_plus+n_plus_background:n_plus+n_plus_background+n_minus] = np.array(stats.norm.rvs(v_electron[1], v_thermal_electron, size=n_minus))
v[1, n_plus+n_plus_background+n_minus:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_minus_background))
v[2, :n_plus] = np.array(stats.norm.rvs(v_ion[2], v_thermal_ion, size=n_plus))
v[2, n_plus:n_plus+n_plus_background] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_plus_background))
v[2, n_plus+n_plus_background:n_plus+n_plus_background+n_minus] = np.array(stats.norm.rvs(v_electron[2], v_thermal_electron, size=n_minus))
v[2, n_plus+n_plus_background+n_minus:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_minus_background))

q_list = np.zeros(n_plus + n_plus_background + n_minus + n_minus_background)
q_list[:n_plus+n_plus_background] = q_ion
q_list[n_plus+n_plus_background:] = q_electron
m_list = np.zeros(n_plus + n_plus_background + n_minus + n_minus_background)
m_list[:n_plus+n_plus_background] = m_ion
m_list[n_plus+n_plus_background:] = m_electron