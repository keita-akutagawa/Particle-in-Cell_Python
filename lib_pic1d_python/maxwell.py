import numpy as np

def time_evolution_E(B, current, c, epsilon0, dx, dt, E):
    E[0, :] = -current[0, :]/epsilon0 * dt + E[0, :]
    E[1, :] = (-current[1, :]/epsilon0 \
            - c**2 * (B[2, :] - np.roll(B[2, :], 1))/dx) * dt \
            + E[1, :]
    E[2, :] = (-current[2, :]/epsilon0 \
            + c**2 * (B[1, :] - np.roll(B[1, :], 1)) / dx) * dt \
            + E[2, :]
    return E


def time_evolution_B(E, dx, dt, B):
    B[0, :] = B[0, :]
    B[1, :] = -(-(np.roll(E[2, :], -1) - E[2, :]) / dx) * dt \
            + B[1, :]
    B[2, :] = -((np.roll(E[1, :], -1) - E[1, :]) / dx) * dt \
            + B[2, :]
    return B