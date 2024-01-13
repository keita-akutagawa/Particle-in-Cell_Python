import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1/100
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 1.0
r_q = 1.0
V_A = c / 16.7
n_e = 8 #ここは手動で調整すること
n_i = int(n_e / r_q)
B0 = V_A * np.sqrt(mu_0 * (n_i * m_ion + n_e * m_electron))
T_i = (B0**2 / 2.0 / mu_0) / (n_i + n_e * t_r)
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

dx = 1.0
dy = 1.0
n_x = int(50 * ion_inertial_length)
n_y = int(10 * ion_inertial_length)
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
dt = 1.0
step = 50000
t_max = step * dt

if c * dt > dx:
    print(f"You had better change some parameters! \nCFL condition is not satisfied \n c * dt = {c * dt} > dx = {dx} \n")
else:
    print(f'c * dt = {c * dt:.5f} < dx = {dx:.5f} \n')

if omega_pe * dt > 0.2:
    print(f"You had better change some parameters! \n")
    print(f"omega_pe * dt = {omega_pe * dt:.5f} > 0.1 \n")
if round(dx, 5) != 1.0:
    print(f"You had better change some parameters! \n")
    print(f"debye length = {debye_length:.5f} should be equal to grid size = {dx:.5f} \n")


E = np.zeros([3, n_x, n_y])
B = np.zeros([3, n_x, n_y])
current = np.zeros([3, n_x, n_y])
for j in range(n_y):
    B[0, :, j] = B0 * np.tanh((y_coordinate[j] - y_max/2) / sheat_thickness)

reconnection_ratio = 0.1
delta_B = np.zeros([3, n_x, n_y])
X, Y = np.meshgrid(x_coordinate, y_coordinate)
delta_B[0, :, :] = -np.array(reconnection_ratio * B0 * (Y - y_max/2) / sheat_thickness \
                 * np.exp(-((X - x_max/2)**2 + (Y - y_max/2)**2) / ((2.0 * sheat_thickness)**2))).T
delta_B[1, :, :] = np.array(reconnection_ratio * B0 * (X - x_max/2) / sheat_thickness \
                 * np.exp(-((X - x_max/2)**2 + (Y - y_max/2)**2) / ((2.0 * sheat_thickness)**2))).T

n_plus = int(n_x * n_i * 2.0 * sheat_thickness)
n_minus = int(n_plus * abs(q_ion / q_electron))
x = np.zeros([3, n_plus + n_minus])
v = np.zeros([3, n_plus + n_minus])
print(f"total number of particles is {n_plus + n_minus}.")

np.random.RandomState(1)
x_start_plus = np.random.rand(n_plus) * x_max
x_start_minus = np.random.rand(n_minus) * x_max
y_start_plus = np.array(y_max/2 + sheat_thickness * np.arctanh(2.0 * np.random.rand(n_plus) - 1.0))
y_start_plus[y_start_plus > y_max] = y_max/2
y_start_plus[y_start_plus < 0.0] = y_max/2
y_start_minus = np.array(y_max/2 + sheat_thickness * np.arctanh(2.0 * np.random.rand(n_minus) - 1.0))
y_start_minus[y_start_minus > y_max] = y_max/2
y_start_minus[y_start_minus < 0.0] = y_max/2
x[0, :] = np.concatenate([x_start_plus, x_start_minus])
x[1, :] = np.concatenate([y_start_plus, y_start_minus])
v[0, :n_plus] = np.array(stats.norm.rvs(v_ion[0], v_thermal_ion, size=n_plus))
v[0, n_plus:] = np.array(stats.norm.rvs(v_electron[0], v_thermal_electron, size=n_minus))
v[1, :n_plus] = np.array(stats.norm.rvs(v_ion[1], v_thermal_ion, size=n_plus))
v[1, n_plus:] = np.array(stats.norm.rvs(v_electron[1], v_thermal_electron, size=n_minus))
v[2, :n_plus] = np.array(stats.norm.rvs(v_ion[2], v_thermal_ion, size=n_plus))
v[2, n_plus:] = np.array(stats.norm.rvs(v_electron[2], v_thermal_electron, size=n_minus))

q_list = np.zeros(n_plus + n_minus)
q_list[:n_plus] = q_ion
q_list[n_plus:] = q_electron
m_list = np.zeros(n_plus + n_minus)
m_list[:n_plus] = m_ion
m_list[n_plus:] = m_electron


