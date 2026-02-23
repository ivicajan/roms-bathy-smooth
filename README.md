# ROMS Bathymetry Smoothing

Python library for smoothing ocean model (ROMS) bathymetry to satisfy rx0 and rx1 roughness constraints using Linear Programming and iterative methods.

Converted from the MATLAB LP_SMOOTH toolbox (Mathieu Dutour Sikiric), with modifications by Ivica Janekovic.

## Features

- **LP smoothing (rx0)** — minimise bathymetry change subject to rx0 constraints using scipy/HiGHS
- **Heuristic decomposition** — connected-component decomposition for large grids, with parallel multiprocessing
- **Depth-dependent rx0** — spatially variable targets (stricter in deep water, relaxed in shallow)
- **Iterative smoothers** — Laplacian, positive-only, negative-only (rx0), positive rx1
- **ROMS vertical coordinates** — Vtransform 1/2, Vstretching 1/2/3
- **Vectorised NumPy** — no external LP solver binary needed

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy
- scipy (HiGHS LP solver)
- netCDF4
- matplotlib (optional, for plotting)

## Quick Start

```python
from roms_bathy_smooth import (
    ROMSGrid, VerticalCoords, compute_rx0,
    lp_heuristic, smooth_positive_rx1,
)

# Load grid
grid = ROMSGrid.from_netcdf('my_grid.nc')

# Depth-dependent rx0 target (stricter in deep water)
rx0_target = grid.depth_dependent_rx0(
    depths=[8000, 4000, 3000, 2000, 1000, 0],
    values=[0.02, 0.05, 0.07, 0.10, 0.13, 0.15],
)

# LP smoothing with parallel component solves
h_smooth = lp_heuristic(grid.mask, grid.hraw, rx0_target, n_workers=8)

# rx1 smoothing
vertical = VerticalCoords(N=30, theta_s=7.0, theta_b=1.0,
                          vtransform=2, vstretching=2, hc=200)
h_final = smooth_positive_rx1(grid.mask, h_smooth, rx1max=6.0, vertical=vertical)

# Write result
grid.h = h_final
grid.write_h()
```

## Example

See `example_nwa.py` for a complete workflow matching the original MATLAB `smooth_example_NWA.m`.

```bash
python example_nwa.py
```

## MATLAB Reference

The original MATLAB `.m` files are included in the `matlab/` directory for reference.

## Module Reference

| Module | Description |
|---|---|
| `grid.py` | `ROMSGrid` class — mask, bathymetry, NetCDF I/O |
| `vertical.py` | `VerticalCoords` — Sc, Cs, Z-level computation |
| `roughness.py` | Vectorised rx0 and rx1 computation |
| `lp_smoothing.py` | LP rx0 smoothing via scipy.optimize.linprog (HiGHS) |
| `heuristic.py` | Connected-component decomposition + parallel LP |
| `iterative.py` | Laplacian, positive/negative rx0, positive rx1 |
| `graph.py` | Connected components, BFS neighbourhood |
