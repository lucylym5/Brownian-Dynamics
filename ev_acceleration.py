"""三维无流 Hookean + WCA 的 Numba 核；保留原显式 Euler-Maruyama 离散。"""

import numpy as np
from numba import njit


@njit(cache=True)
def wca_force(r, epsilon, sigma):
    force = np.zeros_like(r)
    if epsilon == 0.0:
        return force
    cutoff2 = (2.0**(1.0 / 6.0) * sigma)**2
    sigma2 = sigma * sigma
    for m in range(r.shape[0]):
        for gap in range(1, r.shape[1]):
            for left in range(r.shape[1] - gap):
                right = left + gap
                dx = r[m, right, 0] - r[m, left, 0]
                dy = r[m, right, 1] - r[m, left, 1]
                dz = r[m, right, 2] - r[m, left, 2]
                distance2 = dx * dx + dy * dy + dz * dz
                if distance2 == 0.0:
                    raise ValueError("Completely overlapping beads in WCA force.")
                if distance2 < cutoff2:
                    s6 = (sigma2 / distance2)**3
                    factor = 24.0 * epsilon / distance2 * (2.0 * s6 * s6 - s6)
                    fx, fy, fz = factor * dx, factor * dy, factor * dz
                    force[m, right, 0] += fx
                    force[m, right, 1] += fy
                    force[m, right, 2] += fz
                    force[m, left, 0] -= fx
                    force[m, left, 1] -= fy
                    force[m, left, 2] -= fz
    return force


@njit(cache=True)
def hookean_wca_step(r, standard_noise, kBT, gamma, H, dt, epsilon, sigma):
    ev_force = wca_force(r, epsilon, sigma)
    updated = np.empty_like(r)
    noise_scale = np.sqrt(2.0 * (kBT / gamma) * dt)
    for m in range(r.shape[0]):
        for bead in range(r.shape[1]):
            for axis in range(3):
                spring = 0.0
                if bead < r.shape[1] - 1:
                    spring += H * (r[m, bead + 1, axis] - r[m, bead, axis])
                if bead > 0:
                    spring -= H * (r[m, bead, axis] - r[m, bead - 1, axis])
                updated[m, bead, axis] = (
                    r[m, bead, axis]
                    + (spring + ev_force[m, bead, axis]) / gamma * dt
                    + noise_scale * standard_noise[m, bead, axis]
                )
    return updated


def warm_up():
    """在主进程准备缓存，避免每个工作进程重复完整编译。"""
    r = np.zeros((1, 2, 3))
    r[0, 1, 0] = 1.5
    hookean_wca_step(r, np.zeros_like(r), 1.0, 1.0, 1.0, 1e-4, 1.0, 1.0)
