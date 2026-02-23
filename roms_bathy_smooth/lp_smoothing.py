"""Linear Programming bathymetry smoothing for rx0.

Formulates the rx0 smoothing problem as a Linear Program:
    minimize sum |h_new - h_obs|
    subject to:
        rx0 constraints: |h_i - h_j| / (h_i + h_j) <= r  for all adjacent pairs
        amplitude constraints: |h_new - h_obs| <= alpha * h_obs
        sign constraints: force increase-only or decrease-only at selected points

Uses scipy.optimize.linprog with the HiGHS backend (sparse matrices,
no file I/O, much faster than external lp_solve).

Replaces: GRID_LinearProgrammingSmoothing_rx0.m,
          GRID_LinearProgrammingSmoothing_rx0_simple_v2.m,
          GRID_LinProgGetIJS_rx0.m, GRID_LinProgGetIJS_maxamp.m,
          GRID_LinProgGetIJS_signs.m, LP_MergeIJS_listings.m,
          LP_WriteLinearProgram.m, LP_SolveLinearProgram.m,
          LP_ReadLinearProgram.m
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .roughness import compute_rx0


def _build_vertex_map(mask):
    """Build mapping from 2D grid indices to sequential vertex indices.

    Returns:
        vertex_map: Array of shape (eta, xi), -1 for land, 0..n-1 for sea.
        n_verts: Total number of sea vertices.
    """
    vertex_map = np.full(mask.shape, -1, dtype=int)
    sea = mask == 1
    vertex_map[sea] = np.arange(sea.sum())
    return vertex_map, int(sea.sum())


def _build_rx0_constraints(mask, h, rx0max, vertex_map, n_verts):
    """Build inequality constraints for the rx0 roughness condition.

    For each pair of adjacent sea points (i,j) and (i',j'):
        (1-r)*d_i - (1+r)*d_j <= (1+r)*h_j + (r-1)*h_i
        -(1+r)*d_i + (1-r)*d_j <= (1+r)*h_i + (r-1)*h_j

    where d_i = h_new_i - h_obs_i is the correction variable.

    Args:
        mask: Land/sea mask.
        h: Observed bathymetry.
        rx0max: Target rx0, scalar or array of shape (eta, xi).
        vertex_map: Vertex index map.
        n_verts: Number of sea vertices.

    Returns:
        rows, cols, vals, rhs: COO sparse data for A_ub and b_ub.
    """
    eta, xi = mask.shape
    rx0max = np.broadcast_to(rx0max, (eta, xi))

    rows, cols, vals, rhs = [], [], [], []
    row_idx = 0

    # Eta direction: pairs (iEta, iEta+1)
    for i in range(eta - 1):
        for j in range(xi):
            if mask[i, j] == 1 and mask[i + 1, j] == 1:
                idx1 = vertex_map[i, j]
                idx2 = vertex_map[i + 1, j]
                r1 = rx0max[i, j]
                r2 = rx0max[i + 1, j]

                # Constraint 1: (1-r1)*d1 - (1+r2)*d2 <= (1+r2)*h2 + (r1-1)*h1
                cst = (1 + r2) * h[i + 1, j] + (r1 - 1) * h[i, j]
                rows.extend([row_idx, row_idx])
                cols.extend([idx1, idx2])
                vals.extend([1 - r1, -(1 + r2)])
                rhs.append(cst)
                row_idx += 1

                # Constraint 2: -(1+r1)*d1 + (1-r2)*d2 <= (1+r1)*h1 + (r2-1)*h2
                cst = (1 + r1) * h[i, j] + (r2 - 1) * h[i + 1, j]
                rows.extend([row_idx, row_idx])
                cols.extend([idx1, idx2])
                vals.extend([-(1 + r1), 1 - r2])
                rhs.append(cst)
                row_idx += 1

    # Xi direction: pairs (iXi, iXi+1)
    for i in range(eta):
        for j in range(xi - 1):
            if mask[i, j] == 1 and mask[i, j + 1] == 1:
                idx1 = vertex_map[i, j]
                idx2 = vertex_map[i, j + 1]
                r1 = rx0max[i, j]
                r2 = rx0max[i, j + 1]

                # Constraint 1
                cst = (1 + r2) * h[i, j + 1] + (r1 - 1) * h[i, j]
                rows.extend([row_idx, row_idx])
                cols.extend([idx1, idx2])
                vals.extend([1 - r1, -(1 + r2)])
                rhs.append(cst)
                row_idx += 1

                # Constraint 2
                cst = (1 + r1) * h[i, j] + (r2 - 1) * h[i, j + 1]
                rows.extend([row_idx, row_idx])
                cols.extend([idx1, idx2])
                vals.extend([-(1 + r1), 1 - r2])
                rhs.append(cst)
                row_idx += 1

    return rows, cols, vals, rhs, row_idx


def _build_amplitude_constraints(mask, h, amp_const, vertex_map, n_verts, row_offset):
    """Build constraints: |h_new - h_obs| <= alpha * h_obs.

    For each sea point with amp_const < 9999:
        d_i <= alpha * h_i
        -d_i <= alpha * h_i

    Returns:
        rows, cols, vals, rhs, new_row_offset
    """
    eta, xi = mask.shape
    rows, cols, vals, rhs = [], [], [], []
    row_idx = row_offset

    for i in range(eta):
        for j in range(xi):
            if mask[i, j] == 1 and amp_const[i, j] < 9999:
                idx = vertex_map[i, j]
                alpha_h = amp_const[i, j] * h[i, j]

                # d_i <= alpha*h
                rows.append(row_idx)
                cols.append(idx)
                vals.append(1.0)
                rhs.append(alpha_h)
                row_idx += 1

                # -d_i <= alpha*h
                rows.append(row_idx)
                cols.append(idx)
                vals.append(-1.0)
                rhs.append(alpha_h)
                row_idx += 1

    return rows, cols, vals, rhs, row_idx


def _build_sign_constraints(mask, sign_const, vertex_map, n_verts, row_offset):
    """Build sign constraints on corrections.

    sign_const == +1: only increases allowed (d_i >= 0 => -d_i <= 0)
    sign_const == -1: only decreases allowed (d_i <= 0)

    Returns:
        rows, cols, vals, rhs, new_row_offset
    """
    eta, xi = mask.shape
    rows, cols, vals, rhs = [], [], [], []
    row_idx = row_offset

    for i in range(eta):
        for j in range(xi):
            if mask[i, j] == 1 and sign_const[i, j] != 0:
                idx = vertex_map[i, j]
                if sign_const[i, j] == 1:
                    # d_i >= 0 => -d_i <= 0
                    rows.append(row_idx)
                    cols.append(idx)
                    vals.append(-1.0)
                elif sign_const[i, j] == -1:
                    # d_i <= 0
                    rows.append(row_idx)
                    cols.append(idx)
                    vals.append(1.0)
                rhs.append(0.0)
                row_idx += 1

    return rows, cols, vals, rhs, row_idx


def _build_absval_constraints(n_verts, row_offset):
    """Build absolute value linearization: d_i <= t_i and -d_i <= t_i.

    Variables are [d_0..d_{n-1}, t_0..t_{n-1}] where t_i >= |d_i|.

    Returns:
        rows, cols, vals, rhs, new_row_offset
    """
    rows, cols, vals, rhs = [], [], [], []
    row_idx = row_offset

    for idx in range(n_verts):
        # d_i - t_i <= 0
        rows.extend([row_idx, row_idx])
        cols.extend([idx, n_verts + idx])
        vals.extend([1.0, -1.0])
        rhs.append(0.0)
        row_idx += 1

        # -d_i - t_i <= 0
        rows.extend([row_idx, row_idx])
        cols.extend([idx, n_verts + idx])
        vals.extend([-1.0, -1.0])
        rhs.append(0.0)
        row_idx += 1

    return rows, cols, vals, rhs, row_idx


def lp_smooth_rx0(mask, h, rx0max, amp_const=None, sign_const=None):
    """Smooth bathymetry using Linear Programming to satisfy rx0 constraints.

    Minimizes sum of |h_new - h_obs| subject to rx0 roughness constraints,
    amplitude constraints, and sign constraints.

    Args:
        mask: Land/sea mask, shape (eta, xi). 1=sea, 0=land.
        h: Observed bathymetry, shape (eta, xi). Positive depth.
        rx0max: Target rx0 roughness, scalar or array (eta, xi).
        amp_const: Maximum relative change |dh|/h, shape (eta, xi).
                   Use large value (e.g. 100) for unconstrained points.
                   Default: 100 everywhere.
        sign_const: Sign constraint, shape (eta, xi).
                    +1 = only deepen, -1 = only shallow, 0 = both.
                    Default: 0 everywhere.

    Returns:
        h_new: Smoothed bathymetry, shape (eta, xi).
    """
    eta, xi = mask.shape

    if amp_const is None:
        amp_const = np.full((eta, xi), 100.0)
    if sign_const is None:
        sign_const = np.zeros((eta, xi))

    amp_const = np.broadcast_to(amp_const, (eta, xi)).copy()
    sign_const = np.broadcast_to(sign_const, (eta, xi)).copy()
    rx0max = np.broadcast_to(np.atleast_1d(rx0max), (eta, xi)).copy()

    vertex_map, n_verts = _build_vertex_map(mask)

    if n_verts == 0:
        return h.copy()

    # Build all constraints
    all_rows, all_cols, all_vals, all_rhs = [], [], [], []

    # rx0 constraints
    r, c, v, rhs, row_off = _build_rx0_constraints(mask, h, rx0max, vertex_map, n_verts)
    all_rows.extend(r); all_cols.extend(c); all_vals.extend(v); all_rhs.extend(rhs)

    # Amplitude constraints
    r, c, v, rhs, row_off = _build_amplitude_constraints(
        mask, h, amp_const, vertex_map, n_verts, row_off)
    all_rows.extend(r); all_cols.extend(c); all_vals.extend(v); all_rhs.extend(rhs)

    # Sign constraints
    r, c, v, rhs, row_off = _build_sign_constraints(
        mask, sign_const, vertex_map, n_verts, row_off)
    all_rows.extend(r); all_cols.extend(c); all_vals.extend(v); all_rhs.extend(rhs)

    # Absolute value linearization: t_i >= |d_i|
    r, c, v, rhs, row_off = _build_absval_constraints(n_verts, row_off)
    all_rows.extend(r); all_cols.extend(c); all_vals.extend(v); all_rhs.extend(rhs)

    n_constraints = row_off
    n_vars = 2 * n_verts  # [d_0..d_{n-1}, t_0..t_{n-1}]

    # Build sparse constraint matrix
    A_ub = coo_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_constraints, n_vars)
    ).tocsc()
    b_ub = np.array(all_rhs)

    # Objective: minimize sum(t_i)
    c_obj = np.zeros(n_vars)
    c_obj[n_verts:] = 1.0

    # Variable bounds: d_i is free (unbounded), t_i >= 0
    bounds = [(None, None)] * n_verts + [(0, None)] * n_verts

    # Solve
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options={'disp': False, 'presolve': True})

    if not result.success:
        print(f"LP solver warning: {result.message}")
        return h.copy()

    # Extract corrections and build new bathymetry
    corrections = result.x[:n_verts]
    h_new = h.copy()
    sea = mask == 1
    h_new[sea] = h[sea] + corrections

    rx0_new = compute_rx0(h_new, mask)
    print(f"  LP result: objective={result.fun:.4f}, "
          f"rx0_max={rx0_new.max():.6f} (target={rx0max.max():.4f})")

    return h_new
