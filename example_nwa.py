#!/usr/bin/env python3
"""NWA bathymetry smoothing example.

Reproduces the workflow from smooth_example_NWA.m using the Python
roms_bathy_smooth library. Reads the ROMS grid from north_grid.nc,
applies depth-dependent rx0 smoothing via LP + rx1 smoothing, and
optionally writes the result back.

Usage:
    python example_nwa.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from roms_bathy_smooth import (
    ROMSGrid,
    VerticalCoords,
    compute_rx0,
    compute_rx1,
    lp_heuristic,
    smooth_positive_rx1,
    laplacian_smooth_rx0,
)


def main():
    # ----------------------------------------------------------------
    # 1. Load grid
    # ----------------------------------------------------------------
    grid = ROMSGrid.from_netcdf('north_grid.nc')
    print(f"Grid loaded: {grid.shape} (eta x xi)")
    print(f"  Sea points: {int(grid.mask.sum())}")

    # Use hraw if it has real data, otherwise fall back to h
    hraw = grid.hraw.copy()
    if hraw.max() - hraw.min() < 1.0:
        print("  hraw is uniform — using h as input bathymetry")
        hraw = grid.h.copy()

    # ----------------------------------------------------------------
    # 2. Clip depth range
    # ----------------------------------------------------------------
    hmin = 10.0
    hmax = hraw.max()
    hraw = np.clip(hraw, hmin, hmax)
    print(f"  Depth range: {hraw.min():.1f} to {hraw.max():.1f} m")

    # ----------------------------------------------------------------
    # 3. Depth-dependent rx0 target
    #    Deeper areas get stricter (lower) rx0 — smooth more in deep,
    #    less in shallow. Same breakpoints as the MATLAB example.
    # ----------------------------------------------------------------
    depths = [8000, 4000, 3000, 2000, 1000, 0]
    values = [0.02, 0.05, 0.07, 0.10, 0.13, 0.15]

    # Use grid method (handles sorting for np.interp)
    grid.hraw = hraw  # ensure grid has the clipped version
    rx0_target = grid.depth_dependent_rx0(depths, values)
    sea = grid.mask == 1
    print(f"  rx0 target range: {rx0_target[sea].min():.3f} to {rx0_target[sea].max():.3f}")

    # Initial roughness
    rx0_before = compute_rx0(hraw, grid.mask)
    print(f"  Initial rx0 = {rx0_before.max():.4f}")

    # ----------------------------------------------------------------
    # 4. Vertical coordinate parameters
    # ----------------------------------------------------------------
    vertical = VerticalCoords(
        N=20,
        theta_s=7.0,
        theta_b=0.5,
        vtransform=2,
        vstretching=2,
        hc=50.0,
    )

    # ----------------------------------------------------------------
    # 5. First sweep: LP heuristic rx0 smoothing
    # ----------------------------------------------------------------
    print("\n=== First sweep: LP heuristic rx0 smoothing ===")
    h_lp = lp_heuristic(grid.mask, hraw, rx0_target,
                        amp=100 * np.ones_like(hraw),
                        sign=np.zeros_like(hraw))

    rx0_after_lp = compute_rx0(h_lp, grid.mask)
    print(f"  After LP: rx0 = {rx0_after_lp.max():.4f}")

    # ----------------------------------------------------------------
    # 6. First sweep: rx1 positive smoothing
    # ----------------------------------------------------------------
    print("\n=== First sweep: rx1 smoothing (target=6.8) ===")
    h_rx1 = smooth_positive_rx1(grid.mask, h_lp, rx1max=6.8,
                                vertical=vertical)

    # ----------------------------------------------------------------
    # 7. Boundary blending (if external depth data available)
    # ----------------------------------------------------------------
    # In the original MATLAB example, the bathymetry is blended with
    # Mercator model depth near boundaries. Here we skip this step
    # since we don't have the Mercator data, but the infrastructure
    # is in grid.boundary_taper().
    h = h_rx1.copy()

    # ----------------------------------------------------------------
    # 8. Light 3x3 convolution smoothing
    # ----------------------------------------------------------------
    kernel = np.ones((3, 3)) * 0.1
    kernel[1, 1] = 0.2
    h_conv = convolve(h, kernel, mode='constant', cval=0)
    # Only apply to interior (keep boundary row/col from original)
    h_smooth = h.copy()
    h_smooth[1:-1, 1:-1] = h_conv[1:-1, 1:-1]

    # ----------------------------------------------------------------
    # 9. Second sweep with stricter boundary targets
    # ----------------------------------------------------------------
    # Smooth 50% more at boundaries (lower rx0 target near edges),
    # matching the MATLAB example: w goes from 2 at edge to 1 interior,
    # rx0_target2 = rx0_target / w  (halved at boundary)
    width = 30
    eta, xi = grid.shape
    w = np.ones_like(rx0_target)
    ramp = np.linspace(2, 1, width, endpoint=False)
    for k in range(width):
        v = ramp[k]
        w[k:eta - k, k] = np.minimum(w[k:eta - k, k], v)             # west
        w[k, k:xi - k] = np.minimum(w[k, k:xi - k], v)               # south
        w[eta - 1 - k, k:xi - k] = np.minimum(w[eta - 1 - k, k:xi - k], v)  # north
        w[k:eta - k, xi - 1 - k] = np.minimum(w[k:eta - k, xi - 1 - k], v)  # east
    rx0_target2 = rx0_target / w
    rx0_target2 = np.maximum(rx0_target2, min(values))

    print("\n=== Second sweep: LP heuristic rx0 smoothing ===")
    h_lp2 = lp_heuristic(grid.mask, h_smooth, rx0_target2,
                         amp=100 * np.ones_like(hraw),
                         sign=np.zeros_like(hraw))

    print("\n=== Second sweep: rx1 smoothing (target=6.0) ===")
    h_final = smooth_positive_rx1(grid.mask, h_lp2, rx1max=6.0,
                                  vertical=vertical)

    # Enforce minimum depth
    h_final = np.maximum(h_final, hmin)

    # ----------------------------------------------------------------
    # 10. Final diagnostics
    # ----------------------------------------------------------------
    rx0_final = compute_rx0(h_final, grid.mask)
    Z_r, Z_w = vertical.get_z_levels(h_final, grid.mask)
    rx1_final = compute_rx1(Z_w, grid.mask)

    print(f"\n{'='*60}")
    print(f" SMOOTHING COMPLETE")
    print(f"{'='*60}")
    print(f"  rx0: {rx0_before.max():.4f} -> {rx0_final.max():.4f}")
    print(f"  rx1: {rx1_final.max():.4f}")
    print(f"  Depth range: {h_final[sea].min():.1f} to {h_final[sea].max():.1f} m")
    diff = h_final - hraw
    print(f"  Mean depth change: {diff[sea].mean():.3f} m")
    print(f"  Max depth change:  {np.abs(diff[sea]).max():.2f} m")

    # ----------------------------------------------------------------
    # 11. Comparison plots
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('ROMS Bathymetry Smoothing — NWA Grid', fontsize=14)

    # (a) Original bathymetry
    ax = axes[0, 0]
    im = ax.pcolormesh(hraw, cmap='terrain_r')
    ax.set_title(f'(a) Original bathymetry\nmax={hraw[sea].max():.0f} m')
    plt.colorbar(im, ax=ax, label='Depth (m)')

    # (b) Smoothed bathymetry
    ax = axes[0, 1]
    im = ax.pcolormesh(h_final, cmap='terrain_r')
    ax.set_title(f'(b) Smoothed bathymetry\nmax={h_final[sea].max():.0f} m')
    plt.colorbar(im, ax=ax, label='Depth (m)')

    # (c) Depth change
    ax = axes[0, 2]
    vmax = np.abs(diff[sea]).max()
    im = ax.pcolormesh(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'(c) Depth change (new-old)\nmax |dh|={vmax:.1f} m')
    plt.colorbar(im, ax=ax, label='dh (m)')

    # (d) rx0 before
    ax = axes[1, 0]
    im = ax.pcolormesh(rx0_before, cmap='hot_r', vmin=0, vmax=0.3)
    ax.set_title(f'(d) rx0 before\nmax={rx0_before.max():.4f}')
    plt.colorbar(im, ax=ax, label='rx0')

    # (e) rx0 after
    ax = axes[1, 1]
    im = ax.pcolormesh(rx0_final, cmap='hot_r', vmin=0, vmax=0.3)
    ax.set_title(f'(e) rx0 after\nmax={rx0_final.max():.4f}')
    plt.colorbar(im, ax=ax, label='rx0')

    # (f) rx0 target
    ax = axes[1, 2]
    im = ax.pcolormesh(rx0_target, cmap='hot_r', vmin=0, vmax=0.3)
    ax.set_title(f'(f) rx0 target\nrange={rx0_target[sea].min():.3f}-{rx0_target[sea].max():.3f}')
    plt.colorbar(im, ax=ax, label='rx0 target')

    for ax in axes.flat:
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('smoothing_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved: smoothing_result.png")

    # Optionally write back
    # grid.h = h_final
    # grid.write_h('north_grid_smoothed.nc')


if __name__ == '__main__':
    main()
