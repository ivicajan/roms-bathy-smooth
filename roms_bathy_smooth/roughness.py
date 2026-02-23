"""Roughness factor computation for ROMS grids.

Vectorized computation of rx0 (Haney number) and rx1 (Beckmann-Haidvogel
number) roughness factors used to assess pressure gradient errors in
terrain-following ocean models.

Replaces: GRID_RoughnessMatrix.m, GRID_ComputeMatrixRx1_V2.m
"""

import numpy as np


def compute_rx0(h, mask):
    """Compute the rx0 roughness factor (Haney number).

    rx0 = max over 4-connected neighbors of |h1 - h2| / (h1 + h2)

    Args:
        h: Bathymetry, shape (eta_rho, xi_rho). Positive depth.
        mask: Land/sea mask, shape (eta_rho, xi_rho). 1=sea, 0=land.

    Returns:
        rx0: Roughness matrix, shape (eta_rho, xi_rho).
    """
    rx0 = np.zeros_like(h)

    # Eta direction: pairs (i, i+1)
    m_eta = mask[:-1, :] * mask[1:, :]
    h1, h2 = h[:-1, :], h[1:, :]
    s = h1 + h2
    safe = (m_eta > 0) & (s > 0)
    r_eta = np.where(safe, np.abs(h1 - h2) / np.where(s > 0, s, 1.0), 0.0)
    rx0[:-1, :] = np.maximum(rx0[:-1, :], r_eta)
    rx0[1:, :] = np.maximum(rx0[1:, :], r_eta)

    # Xi direction: pairs (j, j+1)
    m_xi = mask[:, :-1] * mask[:, 1:]
    h1, h2 = h[:, :-1], h[:, 1:]
    s = h1 + h2
    safe = (m_xi > 0) & (s > 0)
    r_xi = np.where(safe, np.abs(h1 - h2) / np.where(s > 0, s, 1.0), 0.0)
    rx0[:, :-1] = np.maximum(rx0[:, :-1], r_xi)
    rx0[:, 1:] = np.maximum(rx0[:, 1:], r_xi)

    # Zero out land points
    rx0[mask == 0] = 0.0
    return rx0


def compute_rx1(z_w, mask):
    """Compute the rx1 roughness factor (Beckmann-Haidvogel number).

    rx1 = max over 4 neighbors and N levels of:
        |z_w(k,i,j) - z_w(k,i',j') + z_w(k-1,i,j) - z_w(k-1,i',j')|
        / |z_w(k,i,j) + z_w(k,i',j') - z_w(k-1,i,j) - z_w(k-1,i',j')|

    Args:
        z_w: Vertical Z at W-points, shape (N+1, eta_rho, xi_rho).
        mask: Land/sea mask, shape (eta_rho, xi_rho). 1=sea, 0=land.

    Returns:
        rx1: Roughness matrix, shape (eta_rho, xi_rho).
    """
    N = z_w.shape[0] - 1
    rx1 = np.zeros_like(mask, dtype=float)

    # Vectorize over all k-levels at once
    zk = z_w[1:, :, :]     # k=1..N, shape (N, eta, xi)
    zkm1 = z_w[:-1, :, :]  # k=0..N-1

    # Eta direction: pairs (i, i+1)
    m_eta = mask[:-1, :] * mask[1:, :]
    if m_eta.any():
        a1 = np.abs(zk[:, :-1, :] - zk[:, 1:, :] + zkm1[:, :-1, :] - zkm1[:, 1:, :])
        b1 = np.abs(zk[:, :-1, :] + zk[:, 1:, :] - zkm1[:, :-1, :] - zkm1[:, 1:, :])
        safe = (m_eta[np.newaxis, :, :] > 0) & (b1 > 0)
        r = np.where(safe, a1 / b1, 0.0)
        r_max = r.max(axis=0)  # max over k
        rx1[:-1, :] = np.maximum(rx1[:-1, :], r_max)
        rx1[1:, :] = np.maximum(rx1[1:, :], r_max)

    # Xi direction: pairs (j, j+1)
    m_xi = mask[:, :-1] * mask[:, 1:]
    if m_xi.any():
        a1 = np.abs(zk[:, :, :-1] - zk[:, :, 1:] + zkm1[:, :, :-1] - zkm1[:, :, 1:])
        b1 = np.abs(zk[:, :, :-1] + zk[:, :, 1:] - zkm1[:, :, :-1] - zkm1[:, :, 1:])
        safe = (m_xi[np.newaxis, :, :] > 0) & (b1 > 0)
        r = np.where(safe, a1 / b1, 0.0)
        r_max = r.max(axis=0)
        rx1[:, :-1] = np.maximum(rx1[:, :-1], r_max)
        rx1[:, 1:] = np.maximum(rx1[:, 1:], r_max)

    rx1[mask == 0] = 0.0
    return rx1
