"""Iterative bathymetry smoothing methods.

Implements Laplacian selective smoothing, positive-only and negative-only
rx0 smoothing, and positive rx1 smoothing for ROMS sigma coordinates.

Replaces: GRID_SmoothPositive_rx0.m, GRID_SmoothNegative_rx0_ivica.m,
          GRID_LaplacianSelectSmooth_rx0_ivica.m,
          GRID_SmoothPositive_ROMS_rx1_ivica.m,
          GRID_SmoothPositive_Vtrans2_rx1_ivica.m
"""

import numpy as np
from .roughness import compute_rx0, compute_rx1
from .vertical import VerticalCoords

NEIGHBORS_4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
NEIGHBORS_8 = [(1, 0), (1, 1), (0, 1), (-1, 1),
               (-1, 0), (-1, -1), (0, -1), (1, -1)]


def smooth_positive_rx0(mask, h, rx0max):
    """Smooth bathymetry by only increasing depth to satisfy rx0.

    Iteratively raises shallow points that violate the rx0 constraint
    relative to deeper neighbors. Vectorized using shifted arrays.

    Args:
        mask: Land/sea mask, shape (eta, xi). 1=sea, 0=land.
        h: Bathymetry, shape (eta, xi). Positive depth.
        rx0max: Target rx0, scalar or array (eta, xi).

    Returns:
        h_new: Smoothed bathymetry (depths only increase or stay same).
    """
    eta, xi = mask.shape
    rx0max = np.broadcast_to(np.atleast_1d(rx0max), (eta, xi)).copy()
    h_new = h.copy().astype(float)
    sea = mask == 1
    tol = 1e-6
    nb_modif = 0

    while True:
        # Compute lower bound from each neighbor direction:
        #   lb = h_neighbor * (1 - r_neighbor) / (1 + r_neighbor)
        # Take max lower bound across all 4 directions
        lb = np.zeros_like(h_new)

        # South neighbor (i+1, j) constrains point (i, j)
        r = rx0max[1:, :]
        m = sea[:-1, :] & sea[1:, :]
        bound = np.where(m, h_new[1:, :] * (1 - r) / (1 + r), 0.0)
        lb[:-1, :] = np.maximum(lb[:-1, :], bound)

        # North neighbor (i-1, j) constrains point (i, j)
        r = rx0max[:-1, :]
        m = sea[1:, :] & sea[:-1, :]
        bound = np.where(m, h_new[:-1, :] * (1 - r) / (1 + r), 0.0)
        lb[1:, :] = np.maximum(lb[1:, :], bound)

        # East neighbor (i, j+1) constrains point (i, j)
        r = rx0max[:, 1:]
        m = sea[:, :-1] & sea[:, 1:]
        bound = np.where(m, h_new[:, 1:] * (1 - r) / (1 + r), 0.0)
        lb[:, :-1] = np.maximum(lb[:, :-1], bound)

        # West neighbor (i, j-1) constrains point (i, j)
        r = rx0max[:, :-1]
        m = sea[:, 1:] & sea[:, :-1]
        bound = np.where(m, h_new[:, :-1] * (1 - r) / (1 + r), 0.0)
        lb[:, 1:] = np.maximum(lb[:, 1:], bound)

        # Update points where h < lower_bound
        violations = sea & (h_new - lb < -tol)
        n_viol = violations.sum()
        if n_viol == 0:
            break
        nb_modif += n_viol
        h_new[violations] = lb[violations]

    print(f"  smooth_positive_rx0: {nb_modif} modifications")
    return h_new


def smooth_negative_rx0(mask, h, rx0max):
    """Smooth bathymetry by only decreasing depth to satisfy rx0.

    Uses 8-connected neighbors (including diagonals).

    Args:
        mask: Land/sea mask, shape (eta, xi). 1=sea, 0=land.
        h: Bathymetry, shape (eta, xi). Positive depth.
        rx0max: Target rx0, scalar or array (eta, xi).

    Returns:
        h_new: Smoothed bathymetry (depths only decrease or stay same).
    """
    eta, xi = mask.shape
    rx0max = np.broadcast_to(np.atleast_1d(rx0max), (eta, xi))
    h_new = h.copy()
    tol = 1e-4
    nb_modif = 0

    while True:
        finished = True
        for i in range(eta):
            for j in range(xi):
                if mask[i, j] != 1:
                    continue
                for di, dj in NEIGHBORS_8:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < eta and 0 <= nj < xi and mask[ni, nj] == 1):
                        r = rx0max[ni, nj]
                        upper_bound = h_new[ni, nj] * (1 + r) / (1 - r)
                        if h_new[i, j] > upper_bound + tol:
                            finished = False
                            h_new[i, j] = upper_bound
                            nb_modif += 1
        if finished:
            break

    print(f"  smooth_negative_rx0: {nb_modif} modifications")
    return h_new


def laplacian_smooth_rx0(mask, h, rx0max):
    """Selective Laplacian diffusion smoothing for rx0.

    Applies Laplacian smoothing only to points that exceed the target rx0
    and their neighbors, iterating until convergence.

    Args:
        mask: Land/sea mask, shape (eta, xi). 1=sea, 0=land.
        h: Bathymetry, shape (eta, xi). Positive depth.
        rx0max: Target rx0, scalar or array (eta, xi).

    Returns:
        h_new: Smoothed bathymetry.
    """
    eta, xi = mask.shape
    rx0max = np.broadcast_to(np.atleast_1d(rx0max), (eta, xi))
    h_ret = h.copy()
    tol = 1e-5

    # Precompute neighbor count (weight) for each point
    weight_matrix = np.zeros((eta, xi))
    for i in range(eta):
        for j in range(xi):
            w = 0
            for di, dj in NEIGHBORS_4:
                ni, nj = i + di, j + dj
                if 0 <= ni < eta and 0 <= nj < xi and mask[ni, nj] == 1:
                    w += 1
            weight_matrix[i, j] = w

    number_dones = np.zeros((eta, xi))

    while True:
        rough = compute_rx0(h_ret, mask)
        k_before = np.sum((rough > rx0max) & (mask == 1))

        correction = np.zeros((eta, xi))
        finished = True
        nb_mod = 0
        additional = np.zeros((eta, xi))

        for i in range(eta):
            for j in range(xi):
                neigh_sum = 0.0
                for di, dj in NEIGHBORS_4:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < eta and 0 <= nj < xi and mask[ni, nj] == 1:
                        neigh_sum += h_ret[ni, nj]
                        additional[ni, nj] += number_dones[i, j]

                w = weight_matrix[i, j]
                do_smooth = False
                if w > tol:
                    if rough[i, j] > rx0max[i, j]:
                        do_smooth = True
                    if number_dones[i, j] > 0:
                        do_smooth = True

                if do_smooth:
                    finished = False
                    delta = (neigh_sum - w * h_ret[i, j]) / (2 * w)
                    correction[i, j] += delta
                    nb_mod += 1
                    number_dones[i, j] = 1

        number_dones += additional
        h_ret += correction

        new_rough = compute_rx0(h_ret, mask)
        k_after = np.sum((new_rough > rx0max) & (mask == 1))

        overlap = np.sum((rough > rx0max) & (new_rough > rx0max) & (mask == 1))
        if overlap == k_after and k_before == k_after:
            erase_str = " no erase"
        else:
            erase_str = ""
            number_dones = np.zeros((eta, xi))

        real_r = rough[mask == 1].max() if mask.any() else 0
        print(f"  laplacian: rx0={real_r:.6f}  nbPointMod={nb_mod}{erase_str}")

        if finished:
            break

    return h_ret


def smooth_positive_rx1(mask, h, rx1max, vertical):
    """Smooth bathymetry by increasing depth to satisfy rx1 constraint.

    Uses ROMS sigma coordinate vertical levels. Solves a quadratic equation
    at each cell/level/neighbor to find the minimum depth increase needed.

    Only supports Vtransform=2 (the standard modern ROMS transform).

    Args:
        mask: Land/sea mask, shape (eta, xi). 1=sea, 0=land.
        h: Bathymetry, shape (eta, xi). Positive depth.
        rx1max: Target rx1, scalar or float.
        vertical: VerticalCoords instance.

    Returns:
        h_new: Smoothed bathymetry (depths only increase).
    """
    eta, xi = mask.shape
    if isinstance(rx1max, (int, float)):
        rx1max_arr = np.full((eta, xi), float(rx1max))
    else:
        rx1max_arr = np.broadcast_to(rx1max, (eta, xi)).copy()

    Sc_w, Cs_w, Sc_r, Cs_r = vertical.get_sc_cs()
    N = vertical.N
    hc = vertical.hc

    # Compute initial rx1
    Z_r, Z_w = vertical.get_z_levels(h, mask)
    rx1_mat = compute_rx1(Z_w, mask)
    print(f"  Original rx1={rx1_mat.max():.4f}")

    if vertical.vtransform != 2:
        raise NotImplementedError("smooth_positive_rx1 only supports Vtransform=2")

    h_ret = h.copy()
    tol = 1e-6

    # Track which points need work
    msk_bad = np.zeros((eta, xi), dtype=int)
    msk_bad[rx1_mat > rx1max_arr] = 1

    nb_modif = 0
    total_diff = 0.0

    while True:
        finished = True
        for i in range(eta):
            for j in range(xi):
                if mask[i, j] != 1 or msk_bad[i, j] != 1:
                    continue

                dep_h = h_ret[i, j]
                did_something = False

                for di, dj in NEIGHBORS_4:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < eta and 0 <= nj < xi and mask[ni, nj] == 1):
                        continue

                    dep_hp = h_ret[ni, nj]
                    r = rx1max_arr[ni, nj]

                    for k in range(1, N + 1):
                        a = hc * Sc_w[k] * (r - 1) - hc * Sc_w[k - 1] * (r + 1)
                        b = Cs_w[k] * (r - 1) - Cs_w[k - 1] * (r + 1)
                        ap = hc * Sc_w[k] * (r + 1) + hc * Sc_w[k - 1] * (-r + 1)
                        bp = Cs_w[k] * (r + 1) + Cs_w[k - 1] * (-r + 1)

                        a_pol = b * (hc + dep_hp)
                        b_pol = a * (hc + dep_hp) + dep_hp * (ap + bp * dep_hp)
                        c_pol = hc * (ap + bp * dep_hp) * dep_hp

                        disc = b_pol * b_pol - 4 * a_pol * c_pol
                        the_sum = dep_h * dep_h * a_pol + dep_h * b_pol + c_pol

                        if disc < 0:
                            if a_pol < 0:
                                raise RuntimeError(
                                    f"Cannot smooth rx1 at ({i},{j}), k={k}: "
                                    f"negative discriminant with a_pol<0")
                            continue

                        if the_sum >= -tol:
                            continue

                        # Need to increase depth
                        sqrt_disc = np.sqrt(disc)
                        sol1 = (-b_pol + sqrt_disc) / (2 * a_pol)
                        sol2 = (-b_pol - sqrt_disc) / (2 * a_pol)

                        # Pick the closest solution that increases depth
                        d1 = abs(dep_h - sol1)
                        d2 = abs(dep_h - sol2)

                        if d1 < d2 and sol1 > dep_h:
                            new_dep = sol1
                        elif d2 < d1 and sol2 > dep_h:
                            new_dep = sol2
                        elif sol1 > dep_h:
                            new_dep = sol1
                        elif sol2 > dep_h:
                            new_dep = sol2
                        else:
                            continue

                        finished = False
                        nb_modif += 1
                        total_diff += abs(new_dep - dep_h)
                        h_ret[i, j] = new_dep
                        dep_h = new_dep
                        did_something = True

                if not did_something:
                    msk_bad[i, j] = 0
                else:
                    for di, dj in NEIGHBORS_4:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < eta and 0 <= nj < xi and mask[ni, nj] == 1:
                            msk_bad[ni, nj] = 1

        if finished:
            break

    # Compute final rx1
    Z_r_new, Z_w_new = vertical.get_z_levels(h_ret, mask)
    rx1_new = compute_rx1(Z_w_new, mask)
    print(f"  smooth_positive_rx1: {nb_modif} modifications, "
          f"total_diff={total_diff:.2f}, new rx1={rx1_new.max():.4f}")

    return h_ret
