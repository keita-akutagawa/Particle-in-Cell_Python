import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1/100
t_r = 1/100
m_electron = 1 * m_unit
m_ion = m_electron / r_m
r_q = 1.0
T_e = 1/2 * m_electron * (0.01*c)**2
T_i = T_e / t_r
n_e = 10 #ここは手動で調整すること
B0 = np.sqrt(n_e) / 10.0
q_unit = np.sqrt(epsilon0 * T_e / n_e)
q_electron = -1 * q_unit
q_ion = r_q * q_unit
n_i = int(n_e * np.abs(q_electron) / q_ion)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
V_A = c * np.sqrt(B0**2 / (n_e*m_electron + n_i*m_ion))
C_S = np.sqrt(r_m * T_e)
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)

dx = debye_length
n_x = 512
x_max = n_x * dx
x_coordinate = np.arange(0.0, x_max, dx)
dt = 1.0
step = 80000
t_max = step * dt
v_thermal_ion = np.sqrt(T_i / m_ion)
v_thermal_electron = np.sqrt(T_e / m_electron)
v_ion = np.array([0.0, 0.0, 0.0])
v_electron = np.array([-10.0*v_thermal_ion, 0.0, 0.0])
v_beam = np.array([10.0*v_thermal_ion, 0.0, 0.0])
if c * dt > dx:
    print(f"You had better change some parameters! \nCFL condition is not satisfied \n c * dt = {c * dt} > dx = {dx} \n")
else:
    print(f'c * dt = {c * dt} < dx = {dx} \n')

if omega_pe * dt > 0.2:
    print(f"You had better change some parameters! \n$\omega$_pe * dt = {omega_pe * dt} > 0.1 \n")
if round(dx, 5) != 1.0:
    print(f"You had better change some parameters! \ndebye length = {debye_length} should be equal to grid size = {dx} \n")


E = np.zeros([3, len(x_coordinate)])
B = np.zeros([3, len(x_coordinate)])
current = np.zeros([3, len(x_coordinate)])

n_plus = int(n_x * n_i)
n_minus = int(n_x * n_e / 2)
n_beam = int(n_x * n_e / 2)
np.random.RandomState(1)
x_start_plus = np.random.rand(n_plus) * x_max
x_start_minus = np.random.rand(n_minus) * x_max
x_start_beam = np.random.rand(n_beam) * x_max
print(f"total number of particle is {n_plus + n_minus + n_beam}")

x = np.zeros([3, n_plus + n_minus + n_beam])
v = np.zeros([3, n_plus + n_minus + n_beam])
x[0, :] = np.concatenate([x_start_plus, x_start_minus, x_start_beam])
v[0, :n_plus] = np.array(stats.norm.rvs(v_ion[0], v_thermal_ion, size=n_plus))
v[0, n_plus:n_plus + n_minus] = np.array(stats.norm.rvs(v_electron[0], v_thermal_electron, size=n_minus))
v[0, n_plus + n_minus:] = np.array(stats.norm.rvs(v_beam[0], v_thermal_electron, size=n_beam))
v[1, :n_plus] = np.array(stats.norm.rvs(v_ion[1], v_thermal_ion, size=n_plus))
v[1, n_plus:n_plus + n_minus] = np.array(stats.norm.rvs(v_electron[1], v_thermal_electron, size=n_minus))
v[1, n_plus + n_minus:] = np.array(stats.norm.rvs(v_beam[1], v_thermal_electron, size=n_beam))
v[2, :n_plus] = np.array(stats.norm.rvs(v_ion[2], v_thermal_ion, size=n_plus))
v[2, n_plus:n_plus + n_minus] = np.array(stats.norm.rvs(v_electron[2], v_thermal_electron, size=n_minus))
v[2, n_plus + n_minus:] = np.array(stats.norm.rvs(v_beam[2], v_thermal_electron, size=n_beam))

q_list = np.zeros(n_plus + n_minus + n_beam)
q_list[:n_plus] = q_ion
q_list[n_plus:] = q_electron
m_list = np.zeros(n_plus + n_minus + n_beam)
m_list[:n_plus] = m_ion
m_list[n_plus:] = m_electron