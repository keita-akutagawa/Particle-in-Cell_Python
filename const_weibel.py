import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
B0 = 1.0
r_m = 1/1
m_electron = 1 * m_unit
m_ion = m_electron / r_m
T_e = 1/2 * m_electron * (0.1 * c)**2
T_i = 1/2 * m_ion * (0.1 * c)**2
C_S = np.sqrt(r_m * T_e)
n_e = 20 #ここは手動で調整すること
q_unit = np.sqrt(T_e / n_e)
q_electron = -1.0 * q_unit
q_ion = 1.0 * q_unit
n_i = int(n_e * np.abs(q_electron) / q_ion)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * r_m) #直したほうがいい
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
V_A = B0 / np.sqrt(mu_0 * (n_e * m_electron + n_i * m_ion))
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)
electron_inertial_length = c / omega_pe

dx = debye_length
dy = debye_length
n_x = 256
n_y = 256
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
dt = 1.0
step = 3000
t_max = step * dt
v_ion = 0.0
v_electron = 0.0
v_thermal_ion = np.sqrt(T_i / m_ion)
v_thermal_electron = np.sqrt(T_e / m_electron)

if c * dt > dx:
    print(f"You had better change some parameters! \nCFL condition is not satisfied \n c * dt = {c * dt} > dx = {dx} \n")
else:
    print(f'c * dt = {c * dt} < dx = {dx} \n')

if omega_pe * dt > 0.2:
    print(f"You had better change some parameters! \n$\omega$_pe * dt = {omega_pe * dt} > 0.1 \n")
if dx != 1.0:
    print(f"You had better change some parameters! \ndebye length = {debye_length} should be equal to grid size = {dx} \n")


n_ion = int(n_x * n_y * n_i)
n_electron = int(n_x * n_y * n_e)
n_particle = n_ion + n_electron
print(f"total particle number is {n_particle}")
np.random.RandomState(1)
x_start_plus = np.random.rand(n_ion) * x_max
y_start_plus = np.random.rand(n_ion) * y_max
x_start_minus = np.random.rand(n_electron) * x_max
y_start_minus = np.random.rand(n_electron) * y_max

x = np.zeros((3, n_ion + n_electron))
v = np.zeros((3, n_ion + n_electron))
x[0, :] = np.concatenate([x_start_plus, x_start_minus])
x[1, :] = np.concatenate([y_start_plus, y_start_minus])
v[0, :n_ion] = np.array(stats.norm.rvs(v_ion, v_thermal_ion, size=n_ion))
v[0, n_ion:] = np.array(stats.norm.rvs(v_electron, v_thermal_electron, size=n_electron))
v[1, :n_ion] = np.array(stats.norm.rvs(v_ion, v_thermal_ion, size=n_ion))
v[1, n_ion:] = np.array(stats.norm.rvs(v_electron, v_thermal_electron, size=n_electron))
v[2, :n_ion] = np.array(stats.norm.rvs(v_ion, v_thermal_ion*5, size=n_ion))
v[2, n_ion:] = np.array(stats.norm.rvs(v_electron, v_thermal_electron*5, size=n_electron))

del x_start_plus, x_start_minus, y_start_plus, y_start_minus

q_list = np.zeros(n_particle)
q_list[:n_ion] = q_ion
q_list[n_ion:] = q_electron
m_list = np.zeros(n_particle)
m_list[:n_ion] = m_ion
m_list[n_ion:] = m_electron


B = np.zeros([3, n_x, n_y])
E = np.zeros([3, n_x, n_y])
current = np.zeros([3, n_x, n_y])

