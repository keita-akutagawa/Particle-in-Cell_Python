import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
B0 = 1.0
r_m = 1/100
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 1.0
T_e = 1/2 * m_electron * (0.1 * c)**2
T_i = T_e / t_r
C_S = np.sqrt(r_m * T_e)
n_e = 10 #ここは手動で調整すること
q_unit = np.sqrt(T_e / n_e)
r_q = 1.0
q_electron = -1 * q_unit
q_ion = r_q * q_unit
n_i = int(n_e * np.abs(q_electron) / q_ion)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * r_m) #直したほうがいい
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
V_A = np.sqrt(B0**2 / mu_0 / (n_e * m_electron + n_i * m_ion))
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)

dx = debye_length
dy = debye_length
n_x = 128
n_y = 128
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
dt = 1.0
step = 10000
t_max = step * dt
v_ion = 0.0
v_electron = 0.0
v_thermal_ion = np.sqrt(2.0 * T_i / m_ion)
v_thermal_electron = np.sqrt(2.0 * T_e / m_electron)


E = np.zeros([3, n_x, n_y])
B = np.zeros([3, n_x, n_y])
current = np.zeros([3, n_x, n_y])

n_plus = int(n_x * n_y * n_i)
n_minus = int(n_x * n_y * n_e)
np.random.RandomState(1)
x_start_plus = np.random.rand(n_plus) * x_max
y_start_plus = np.random.rand(n_plus) * y_max
x_start_minus = np.random.rand(n_minus) * x_max
y_start_minus = np.random.rand(n_minus) * y_max

x = np.zeros([3, n_plus + n_minus])
v = np.zeros([3, n_plus + n_minus])
x[0, :] = np.concatenate([x_start_plus, x_start_minus])
x[1, :] = np.concatenate([y_start_plus, y_start_minus])
v[0, :n_plus] = np.array(stats.norm.rvs(v_ion, v_thermal_ion, size=n_plus))
v[0, n_plus:] = np.array(stats.norm.rvs(v_electron, v_thermal_electron, size=n_minus))
v[1, :n_plus] = np.array(stats.norm.rvs(v_ion, v_thermal_ion, size=n_plus))
v[1, n_plus:] = np.array(stats.norm.rvs(v_electron, v_thermal_electron, size=n_minus))
v[2, :n_plus] = np.array(stats.norm.rvs(v_ion, v_thermal_ion, size=n_plus))
v[2, n_plus:] = np.array(stats.norm.rvs(v_electron, v_thermal_electron, size=n_minus))

q_list = np.zeros(n_plus + n_minus)
q_list[:n_plus] = q_ion
q_list[n_plus:] = q_electron
m_list = np.zeros(n_plus + n_minus)
m_list[:n_plus] = m_ion
m_list[n_plus:] = m_electron