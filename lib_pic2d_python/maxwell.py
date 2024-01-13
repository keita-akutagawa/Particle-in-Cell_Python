import numpy as np


def time_evolution_E(B, current, c, epsilon0, dx, dy, dt, E):

    E[0, :, :] += (-current[0, :, :]/epsilon0 \
               + c**2 * (B[2, :, :] - np.roll(B[2, :, :], 1, axis=1))/dy) * dt
    E[1, :, :] += (-current[1, :, :]/epsilon0 \
               - c**2 * (B[2, :, :] - np.roll(B[2, :, :], 1, axis=0))/dx) * dt
    E[2, :, :] += (-current[2, :, :]/epsilon0 \
               + c**2 * ((B[1, :, :] - np.roll(B[1, :, :], 1, axis=0))/dx \
               - (B[0, :, :] - np.roll(B[0, :, :], 1, axis=1))/dy)) * dt
    
    return E


def time_evolution_B(E, dx, dy, dt, B):

    B[0, :, :] += -(np.roll(E[2, :, :], -1, axis=1) - E[2, : , :])/dy * dt
    B[1, :, :] += (np.roll(E[2, :, :], -1, axis=0) - E[2, :, :])/dx * dt
    B[2, :, :] += (-(np.roll(E[1, :, :], -1, axis=0) - E[1, :, :])/dx \
               + (np.roll(E[0, :, :], -1, axis=1) - E[0, :, :])/dy) * dt
    
    return B