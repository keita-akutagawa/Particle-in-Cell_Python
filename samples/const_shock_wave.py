import numpy as np
from scipy import stats


c = 0.5
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1/25
m_electron = 1 * m_unit
m_ion = m_electron / r_m
n_e_upstream = 50 #ここは手動で調整すること
n_e_downstream = 20 #ここは手動で調整すること
B0 = np.sqrt(n_e_upstream) / 5.0
r_q = 1.0
n_i_upstream = n_e_upstream
n_i_downstream = n_e_downstream
T_e = (B0**2 / 2 / mu_0) / n_e_upstream * 0.5
T_i = (B0**2 / 2 / mu_0) / n_i_upstream * 0.125
q_unit = np.sqrt(epsilon0 * T_e / n_e_upstream)
q_electron = -1 * q_unit
q_ion = r_q * q_unit
omega_pe = np.sqrt(n_e_upstream * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i_upstream * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
V_A = B0 / np.sqrt(mu_0 * (n_e_upstream * m_electron + n_i_upstream * m_ion)) #上流でのAlfven速度
C_S = np.sqrt(r_m * T_e)
debye_length = np.sqrt(epsilon0 * T_e / n_e_upstream / q_electron**2)
ion_inertial_length = c / omega_pi

dx = debye_length
n_x = 8092
x_max = n_x * dx
x_coordinate = np.arange(0.0, x_max, dx)
dt = 1.0
step = 30000
t_max = step * dt
v_ion_upstream = V_A * 3.0
v_ion_downstream = 0.0
v_electron_upstream = V_A * 3.0
v_electron_downstream = 0.0
#v_thermal_ion_upstream = np.sqrt(T_i / m_ion)
#v_thermal_electron_upstream = np.sqrt(T_e / m_electron)
#v_thermal_ion_downstream = np.sqrt(T_i / m_ion) / 10.0
#v_thermal_electron_downstream = np.sqrt(T_e / m_electron) / 10.0
v_thermal_ion = np.sqrt(T_i / m_ion)
v_thermal_electron = np.sqrt(T_e / m_electron)

if c * dt > dx:
    print(f"You had better change some parameters! \nCFL condition is not satisfied \n c * dt = {c * dt} > dx = {dx} \n")
else:
    print(f'c * dt = {c * dt} <= dx = {dx} \n')

if omega_pe * dt > 0.2:
    print(f"You had better change some parameters! \nomega_pe * dt = {omega_pe * dt} > 0.1 \n")
if dx != 1.0:
    print(f"You had better change some parameters! \ndebye length = {debye_length} should be equal to grid size = {dx} \n")


E = np.zeros([3, len(x_coordinate)])
B = np.zeros([3, len(x_coordinate)])
B[2, :] = B0
current = np.zeros([3, len(x_coordinate)])

dense_plasma_left = 0.1
dense_plasma_right = 0.5
rare_plasma_right = 0.9
n_plus_upstream = int(n_x * (dense_plasma_right - dense_plasma_left) * n_i_upstream)
n_minus_upstream = int(n_x * (dense_plasma_right - dense_plasma_left) * n_e_upstream)
n_plus_downstream = int(n_x * (rare_plasma_right - dense_plasma_right) * n_i_downstream)
n_minus_downstream = int(n_x * (rare_plasma_right - dense_plasma_right) * n_e_downstream)
n_total = n_plus_upstream + n_plus_downstream + n_minus_upstream + n_minus_downstream
print(f"total number of particles is {n_total}")

x_start_plus_upstream = np.random.rand(n_plus_upstream) * x_max * (dense_plasma_right - dense_plasma_left) + x_max * dense_plasma_left
x_start_minus_upstream = np.random.rand(n_minus_upstream) * x_max * (dense_plasma_right - dense_plasma_left) + x_max * dense_plasma_left
x_start_plus_downstream = np.random.rand(n_plus_downstream) * x_max * (rare_plasma_right - dense_plasma_right) + x_max * dense_plasma_right
x_start_minus_downstream = np.random.rand(n_minus_downstream) * x_max * (rare_plasma_right - dense_plasma_right) + x_max * dense_plasma_right
x = np.zeros([3, n_total])
v = np.zeros([3, n_total])
#upstream plus, downstream plus, upstream minus, downstream minnus の順
x[0, :] = np.concatenate([x_start_plus_upstream, x_start_plus_downstream, x_start_minus_upstream, x_start_minus_downstream])
v[0, :n_plus_upstream] = stats.norm.rvs(v_ion_upstream, v_thermal_ion, size=n_plus_upstream)
v[0, n_plus_upstream : n_plus_upstream + n_plus_downstream] = stats.norm.rvs(v_ion_downstream, v_thermal_ion, size=n_plus_downstream)
v[0, n_plus_upstream + n_plus_downstream : n_plus_upstream + n_plus_downstream + n_minus_upstream] = stats.norm.rvs(v_electron_upstream, v_thermal_electron, size=n_minus_upstream)
v[0, n_plus_upstream + n_plus_downstream + n_minus_upstream:] = stats.norm.rvs(v_electron_downstream, v_thermal_electron, size=n_minus_downstream)
q_list = np.zeros(n_total)
q_list[:n_plus_upstream + n_plus_downstream] = q_ion
q_list[n_plus_upstream + n_plus_downstream:] = q_electron 
m_list = np.zeros(n_total)
m_list[:n_plus_upstream + n_plus_downstream] = m_ion
m_list[n_plus_upstream + n_plus_downstream:] = m_electron