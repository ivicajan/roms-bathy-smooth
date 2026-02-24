#!/usr/bin/env python3
"""Shark Bay bathymetry smoothing example -- sea-only vs fill-over-land.

Demonstrates the difference between:
  (a) Sea-only smoothing:  mask filters out land, rx0 computed on sea-sea pairs only.
  (b) Fill-over-land:      land depths filled via Laplacian extrapolation,
                           mask=ones so rx0 includes land-sea boundaries.

Option (b) matches how Rutgers ROMS internally computes rx0 and rx1.

Usage:
    python example_shark_bay.py                 # sea-only (default)
    python example_shark_bay.py --fill-land     # fill-over-land (ROMS-compatible)
    python example_shark_bay.py --compare       # run both and compare
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from roms_bathy_smooth import (
    ROMSGrid,
    VerticalCoords,
    compute_rx0,
    compute_rx1,
    lp_heuristic,
    smooth_positive_rx1,
)

# ----------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------
GRID_FILE = 'shark_bay_grid.nc'
HMIN = 4.0
RX0_TARGET_VAL = 0.15
RX1_TARGET = 6.0


def make_vertical():
    return VerticalCoords(
        N=20, theta_s=7.0, theta_b=0.5,
        vtransform=2, vstretching=2, hc=50.0,
    )


def smooth_sea_only(grid, vertical):
    """Standard sea-only smoothing (rx0/rx1 computed on sea-sea pairs)."""
    print("\n" + "=" * 60)
    print(" SEA-ONLY WORKFLOW")
    print("=" * 60)

    hraw = np.maximum(grid.hraw.copy(), HMIN)
    mask = grid.mask

    rx0_before = compute_rx0(hraw, mask)
    print(f"  Initial rx0 (sea-only) = {rx0_before.max():.6f}")

    # LP rx0 smoothing
    print("\n--- LP heuristic rx0 smoothing ---")
    h_lp = lp_heuristic(mask, hraw, RX0_TARGET_VAL, n_workers=1)
    rx0_after_lp = compute_rx0(h_lp, mask)
    print(f"  After LP: rx0 = {rx0_after_lp.max():.6f}")

    # rx1 smoothing
    print(f"\n--- rx1 smoothing (target={RX1_TARGET}) ---")
    h_final = smooth_positive_rx1(mask, h_lp, rx1max=RX1_TARGET,
                                  vertical=vertical)
    h_final = np.maximum(h_final, HMIN)

    # Final diagnostics
    rx0_final = compute_rx0(h_final, mask)
    Z_r, Z_w = vertical.get_z_levels(h_final, mask)
    rx1_final = compute_rx1(Z_w, mask)

    sea = mask == 1
    print(f"\n  Final rx0 (sea-only) = {rx0_final.max():.6f}")
    print(f"  Final rx1 (sea-only) = {rx1_final.max():.4f}")
    print(f"  Depth range: {h_final[sea].min():.1f} to {h_final[sea].max():.1f} m")

    return h_final, rx0_before, rx0_final, rx1_final


def smooth_fill_land(grid, vertical):
    """Fill-over-land smoothing (ROMS-compatible rx0/rx1)."""
    print("\n" + "=" * 60)
    print(" FILL-OVER-LAND WORKFLOW (ROMS-compatible)")
    print("=" * 60)

    # Step 1: Fill land depths via Laplacian extrapolation
    h_filled = grid.fill_land_depths(hmin=HMIN)
    mask_all = np.ones_like(grid.mask)

    rx0_before = compute_rx0(h_filled, mask_all)
    print(f"  Initial rx0 (all cells) = {rx0_before.max():.6f}")

    # Step 2: Depth-dependent rx0 target over ALL points
    rx0_target = grid.depth_dependent_rx0(
        depths=[0], values=[RX0_TARGET_VAL],
        h=h_filled, mask=mask_all,
    )

    # Step 3: LP rx0 smoothing with mask=ones
    print("\n--- LP heuristic rx0 smoothing (all cells) ---")
    h_lp = lp_heuristic(mask_all, h_filled, rx0_target, n_workers=1)
    rx0_after_lp = compute_rx0(h_lp, mask_all)
    print(f"  After LP: rx0 = {rx0_after_lp.max():.6f}")

    # Step 4: rx1 smoothing with mask=ones
    print(f"\n--- rx1 smoothing (target={RX1_TARGET}, all cells) ---")
    h_final = smooth_positive_rx1(mask_all, h_lp, rx1max=RX1_TARGET,
                                  vertical=vertical)
    h_final = np.maximum(h_final, HMIN)

    # Final diagnostics (report both sea-only and all-cells)
    rx0_final_all = compute_rx0(h_final, mask_all)
    rx0_final_sea = compute_rx0(h_final, grid.mask)
    Z_r, Z_w = vertical.get_z_levels(h_final, mask_all)
    rx1_final = compute_rx1(Z_w, mask_all)

    sea = grid.mask == 1
    print(f"\n  Final rx0 (all cells) = {rx0_final_all.max():.6f}")
    print(f"  Final rx0 (sea-only)  = {rx0_final_sea.max():.6f}")
    print(f"  Final rx1 (all cells) = {rx1_final.max():.4f}")
    print(f"  Depth range (sea): {h_final[sea].min():.1f} to {h_final[sea].max():.1f} m")

    return h_final, rx0_before, rx0_final_all, rx1_final


def run_compare(grid, vertical):
    """Run both workflows and produce comparison plots."""
    h_sea, rx0_bef_sea, rx0_aft_sea, rx1_sea = smooth_sea_only(grid, vertical)
    h_land, rx0_bef_land, rx0_aft_land, rx1_land = smooth_fill_land(grid, vertical)

    # Also compute all-cells rx0 for the sea-only result (to show the gap)
    h_filled_raw = grid.fill_land_depths(hmin=HMIN)
    mask_all = np.ones_like(grid.mask)

    # For sea-only result: what does ROMS actually see?
    # Restore original land depths around the sea-only smoothed result
    h_sea_with_land = h_filled_raw.copy()
    sea = grid.mask == 1
    h_sea_with_land[sea] = h_sea[sea]
    rx0_sea_allcells = compute_rx0(h_sea_with_land, mask_all)

    # ----------------------------------------------------------------
    # Comparison figure
    # ----------------------------------------------------------------
    vmax_rx0 = 0.4
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Shark Bay: Sea-Only vs Fill-Over-Land Smoothing', fontsize=14)

    # Row 0 col 0: Original bathymetry
    ax = axes[0, 0]
    im = ax.pcolormesh(np.maximum(grid.hraw, HMIN), cmap='terrain_r')
    ax.set_title(f'(a) Original bathymetry\nhmin={HMIN} m')
    plt.colorbar(im, ax=ax, label='Depth (m)')

    # Row 0 col 1: rx0 before (sea-only)
    ax = axes[0, 1]
    im = ax.pcolormesh(rx0_bef_sea, cmap='hot_r', vmin=0, vmax=vmax_rx0)
    ax.set_title(f'(b) rx0 before (sea-only)\nmax={rx0_bef_sea.max():.4f}')
    plt.colorbar(im, ax=ax, label='rx0')

    # Row 0 col 2: rx0 before (all cells)
    ax = axes[0, 2]
    im = ax.pcolormesh(rx0_bef_land, cmap='hot_r', vmin=0, vmax=vmax_rx0)
    ax.set_title(f'(c) rx0 before (all cells)\nmax={rx0_bef_land.max():.4f}')
    plt.colorbar(im, ax=ax, label='rx0')

    # Row 1 col 0: Land/sea mask
    ax = axes[1, 0]
    im = ax.pcolormesh(grid.mask, cmap='gray')
    ax.set_title(f'(d) Land/sea mask\nsea={int(grid.mask.sum())}, '
                 f'land={int((grid.mask == 0).sum())}')
    plt.colorbar(im, ax=ax, label='mask')

    # Row 1 col 1: rx0 after (sea-only smoothing, but measured over all cells)
    ax = axes[1, 1]
    im = ax.pcolormesh(rx0_sea_allcells, cmap='hot_r', vmin=0, vmax=vmax_rx0)
    ax.set_title(f'(e) rx0 after sea-only\n(ROMS sees: {rx0_sea_allcells.max():.4f})')
    plt.colorbar(im, ax=ax, label='rx0')

    # Row 1 col 2: rx0 after (fill-over-land)
    ax = axes[1, 2]
    im = ax.pcolormesh(rx0_aft_land, cmap='hot_r', vmin=0, vmax=vmax_rx0)
    ax.set_title(f'(f) rx0 after fill-over-land\n(ROMS sees: {rx0_aft_land.max():.4f})')
    plt.colorbar(im, ax=ax, label='rx0')

    for ax in axes.flat:
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('shark_bay_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved: shark_bay_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description='Shark Bay bathymetry smoothing: sea-only vs fill-over-land')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fill-land', action='store_true',
                       help='Fill-over-land workflow (ROMS-compatible)')
    group.add_argument('--compare', action='store_true',
                       help='Run both workflows and produce comparison plots')
    args = parser.parse_args()

    grid = ROMSGrid.from_netcdf(GRID_FILE)
    print(f"Grid loaded: {grid.shape} (eta x xi)")
    print(f"  Sea points: {int(grid.mask.sum())}")
    print(f"  Land points: {int((grid.mask == 0).sum())}")

    vertical = make_vertical()

    if args.compare:
        run_compare(grid, vertical)
    elif args.fill_land:
        smooth_fill_land(grid, vertical)
    else:
        smooth_sea_only(grid, vertical)


if __name__ == '__main__':
    main()
