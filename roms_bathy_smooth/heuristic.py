"""Heuristic LP decomposition for large grids.

Decomposes the bathymetry smoothing problem into connected components
of "bad" points, expands each component by a neighborhood radius,
and solves a separate (smaller) LP problem for each component.

Supports parallel execution of component LP solves via multiprocessing.

Replaces: GRID_LinProgHeuristic_v2.m, GRID_GetBadPoints.m
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import binary_dilation, label

from .iterative import smooth_positive_rx0
from .lp_smoothing import lp_smooth_rx0
from .roughness import compute_rx0


def _get_bad_points(mask, h, rx0max):
    """Identify points that need modification to satisfy rx0."""
    h_smooth = smooth_positive_rx0(mask, h, rx0max)
    msk_bad = np.zeros_like(mask, dtype=int)
    msk_bad[h_smooth != h] = 1
    return msk_bad


def _diamond_kernel(radius):
    """Create a diamond-shaped structuring element for dilation.

    This approximates the step-neighborhood on a grid (L1 distance).
    """
    size = 2 * radius + 1
    k = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if abs(i - radius) + abs(j - radius) <= radius:
                k[i, j] = True
    return k


def _solve_component(sub):
    """Solve LP for a single component subproblem (runs in worker process)."""
    h_local = np.zeros_like(sub['h_sub'])
    m = sub['mask_sub'] > 0
    h_local[m] = sub['h_sub'][m]

    h_solved = lp_smooth_rx0(
        sub['mask_sub'], h_local, sub['rx0_sub'],
        amp_const=sub['amp_sub'], sign_const=sub['sign_sub']
    )

    return {
        'ic': sub['ic'],
        'slice': sub['slice'],
        'mask_sub': sub['mask_sub'],
        'h_solved': h_solved,
        'n_local': sub['n_local'],
    }


def lp_heuristic(mask, h, rx0max, amp=None, sign=None, kdist=5, n_workers=None):
    """LP bathymetry smoothing with connected-component decomposition.

    Steps:
    1. Identify bad points (where rx0 is violated).
    2. Dilate bad-point mask by 2*kdist+1 to connect nearby clusters.
    3. Find connected components via scipy.ndimage.label.
    4. For each component: expand by kdist, extract bounding box, solve LP.
    5. Merge solutions back into the full grid.

    Component LP solves run in parallel using multiprocessing.

    Args:
        mask: Land/sea mask, shape (eta, xi). 1=sea, 0=land.
        h: Bathymetry, shape (eta, xi). Positive depth.
        rx0max: Target rx0, scalar or array (eta, xi).
        amp: Amplitude constraint, shape (eta, xi). Default: 100 everywhere.
        sign: Sign constraint, shape (eta, xi). Default: 0 everywhere.
        kdist: Neighborhood radius for component expansion (default 5).
        n_workers: Number of parallel workers. Default: number of CPU cores.
                   Use 1 for serial execution.

    Returns:
        h_new: Smoothed bathymetry.
    """
    eta, xi = mask.shape
    rx0max = np.broadcast_to(np.atleast_1d(rx0max), (eta, xi)).copy()

    if amp is None:
        amp = np.full((eta, xi), 100.0)
    if sign is None:
        sign = np.zeros((eta, xi))

    amp = np.broadcast_to(amp, (eta, xi)).copy()
    sign = np.broadcast_to(sign, (eta, xi)).copy()

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # Step 1: Find bad points
    msk_bad = _get_bad_points(mask, h, rx0max)
    n_bad = msk_bad.sum()

    if n_bad == 0:
        print("  No bad points found — bathymetry already satisfies rx0.")
        return h.copy()

    print(f"  Found {n_bad} bad points")

    # Step 2: Connect nearby bad points by dilating with radius 2*kdist+1,
    # then find connected components. Much faster than per-point BFS.
    connect_kernel = _diamond_kernel(2 * kdist + 1)
    bad_dilated = binary_dilation(msk_bad, structure=connect_kernel) & (mask == 1)
    comp_labels, n_components = label(bad_dilated)
    print(f"  {n_components} connected component(s), using {n_workers} worker(s)")

    # Print component sizes (in bad points, not dilated)
    for ic in range(1, n_components + 1):
        n_in = (msk_bad & (comp_labels == ic)).sum()
        print(f"    component {ic}: {n_in} bad points")

    # Step 3: Build subproblems — expand each component by kdist and
    # extract bounding-box sub-arrays
    expand_kernel = _diamond_kernel(kdist)
    subproblems = []

    for ic in range(1, n_components + 1):
        # Component mask (just the bad points in this component)
        comp_mask = (comp_labels == ic).astype(int)

        # Expand by kdist to include surrounding context for LP
        expanded = binary_dilation(comp_mask, structure=expand_kernel) & (mask == 1)
        msk_local = expanded.astype(int)

        # Bounding box
        rows = np.any(msk_local, axis=1)
        cols = np.any(msk_local, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        r1 += 1
        c1 += 1

        n_local = int(msk_local[r0:r1, c0:c1].sum())

        subproblems.append({
            'ic': ic,
            'slice': (r0, r1, c0, c1),
            'mask_sub': msk_local[r0:r1, c0:c1].astype(float),
            'h_sub': h[r0:r1, c0:c1].copy(),
            'rx0_sub': rx0max[r0:r1, c0:c1].copy(),
            'amp_sub': amp[r0:r1, c0:c1].copy(),
            'sign_sub': sign[r0:r1, c0:c1].copy(),
            'n_local': n_local,
        })

    # Step 4: Solve LP per component
    h_new = h.copy()

    if n_workers == 1:
        for sub in subproblems:
            print(f"  --- Component {sub['ic']}/{n_components} "
                  f"({sub['n_local']} pts) ---")
            result = _solve_component(sub)
            r0, r1, c0, c1 = result['slice']
            m = result['mask_sub'] > 0
            h_new[r0:r1, c0:c1][m] = result['h_solved'][m]
    else:
        results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_solve_component, sub): sub['ic']
                       for sub in subproblems}
            for future in as_completed(futures):
                ic = futures[future]
                result = future.result()
                results[ic] = result

        for ic in range(1, n_components + 1):
            result = results[ic]
            r0, r1, c0, c1 = result['slice']
            m = result['mask_sub'] > 0
            h_new[r0:r1, c0:c1][m] = result['h_solved'][m]
            print(f"  Component {ic}/{n_components}: {result['n_local']} pts — done")

    rx0_final = compute_rx0(h_new, mask)
    print(f"  Final: rx0_max={rx0_final.max():.6f} (target={rx0max.max():.4f})")

    return h_new
