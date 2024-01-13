import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
r_q = 1.0
n_e = 50 #ここは手動で調整すること
B0 = np.sqrt(n_e) / 2.0
n_i = int(n_e / r_q)
T_i = (B0**2 / 2.0 / mu_0) / n_i * 1.0
T_e = (B0**2 / 2.0 / mu_0) / n_e * 1.0
q_unit = np.sqrt(epsilon0 * T_e / n_e)
q_electron = -1 * q_unit
q_ion = r_q * q_unit
r_m = 1/1
m_electron = 1.0
m_ion = m_electron / r_m
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
V_A = B0 / np.sqrt(mu_0 * n_i * m_ion)
C_S = np.sqrt(5/3 * n_i * T_i / (n_i * m_ion))
V_f = np.sqrt(V_A**2 + C_S**2)
v_thermal_electron = np.sqrt(2.0 * T_e / m_electron)
v_thermal_ion = np.sqrt(2.0 * T_i / m_ion)
shear_thickness = V_f / 2.0 / omega_ci * 4.0

dx = 1.0
dy = 1.0
n_x = 512
n_y = 256
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
dt = 1.0
step = 3000
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
B[2, :, :] = B0

n_plus = int(n_x * n_y * n_i)
n_minus = int(n_plus * abs(q_ion / q_electron))
n_minus_append = int(epsilon0 * V_f * B0 / q_unit * n_x)
x = np.zeros([3, n_plus + n_minus + n_minus_append])
v = np.zeros([3, n_plus + n_minus + n_minus_append])
print(f"total number of particles is {n_plus + n_minus + n_minus_append}.")

np.random.RandomState(1)
x_start_plus = np.random.uniform(low=0.0, high=x_max, size=n_plus)
x_start_minus = np.random.uniform(low=0.0, high=x_max, size=n_minus)
x_start_minus_append = np.random.uniform(low=0.0, high=x_max, size=n_minus_append)
y_start_plus = np.random.uniform(low=0.0, high=y_max, size=n_plus)
y_start_minus = np.random.uniform(low=0.0, high=y_max, size=n_minus)
y_start_minus_append = np.array(y_max/2 + shear_thickness * np.arctanh(2.0 * np.random.rand(n_minus_append) - 1.0))
x[0, :] = np.concatenate([x_start_plus, x_start_minus, x_start_minus_append])
x[1, :] = np.concatenate([y_start_plus, y_start_minus, y_start_minus_append])
v[0, :n_plus] = np.array(stats.norm.rvs(-V_f / 2 * np.tanh((y_start_plus - y_max/2)/shear_thickness), v_thermal_ion))
v[0, n_plus:n_plus+n_minus] = np.array(stats.norm.rvs(-V_f / 2 * np.tanh((y_start_minus - y_max/2)/shear_thickness), v_thermal_electron))
v[0, n_plus+n_minus:] = np.array(stats.norm.rvs(-V_f / 2 * np.tanh((y_start_minus_append - y_max/2)/shear_thickness), v_thermal_electron))
v[1, :n_plus] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_plus))
v[1, n_plus:n_plus+n_minus] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_minus))
v[1, n_plus+n_minus:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_minus_append))
v[2, :n_plus] = np.array(stats.norm.rvs(0.0, v_thermal_ion, size=n_plus))
v[2, n_plus:n_plus+n_minus] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_minus))
v[2, n_plus+n_minus:] = np.array(stats.norm.rvs(0.0, v_thermal_electron, size=n_minus_append))

q_list = np.zeros(n_plus + n_minus + n_minus_append)
q_list[:n_plus] = q_ion
q_list[n_plus:] = q_electron
m_list = np.zeros(n_plus + n_minus + n_minus_append)
m_list[:n_plus] = m_ion
m_list[n_plus:] = m_electron