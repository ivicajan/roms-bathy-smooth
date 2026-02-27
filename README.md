# ROMS Bathymetry Smoothing

Python library for smoothing ocean model (ROMS) bathymetry to satisfy rx0 and rx1 roughness constraints using Linear Programming and iterative methods.

## Important: ROMS rx0/rx1 Computation

> **Rutgers ROMS computes rx0 and rx1 over ALL adjacent cell pairs, including
> land-sea boundaries.** This library's `compute_rx0` / `compute_rx1` only
> consider pairs where both cells are sea (controlled by `mask`). If you
> smooth with the sea-only mask, the roughness values ROMS actually sees at
> coastlines can be much larger than what the library reports.
>
> To match ROMS behaviour, use the **fill-over-land workflow** described below.
> In case you are using WET_DRY then values over land play important role
> and you have to take special care.

## Features

- **LP smoothing (rx0)** -- minimise bathymetry change subject to rx0 constraints using scipy/HiGHS
- **Heuristic decomposition** -- connected-component decomposition for large grids, with parallel multiprocessing
- **Depth-dependent rx0** -- spatially variable targets (stricter in deep water, relaxed in shallow)
- **Fill-over-land** -- Laplacian extrapolation of sea depths over land for ROMS-compatible roughness
- **Iterative smoothers** -- Laplacian, positive-only, negative-only (rx0), positive rx1
- **ROMS vertical coordinates** -- Vtransform 1/2, Vstretching 1/2/3
- **Vectorised NumPy** -- no external LP solver binary needed

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

## Fill-Over-Land Workflow (ROMS-Compatible)

Use this when you need rx0/rx1 to match what ROMS actually computes at runtime:

```python
import numpy as np
from roms_bathy_smooth import (
    ROMSGrid, VerticalCoords, compute_rx0, compute_rx1,
    lp_heuristic, smooth_positive_rx1,
)

grid = ROMSGrid.from_netcdf('my_grid.nc')
vertical = VerticalCoords(N=20, theta_s=7.0, theta_b=0.5,
                          vtransform=2, vstretching=2, hc=50.0)

# 1. Fill land depths via Laplacian extrapolation
h_filled = grid.fill_land_depths(hmin=4.0)

# 2. Use mask=ones so smoothing sees ALL cells
mask_all = np.ones_like(grid.mask)

# 3. Depth-dependent rx0 target over all points
rx0_target = grid.depth_dependent_rx0(
    depths=[8000, 4000, 3000, 2000, 1000, 0],
    values=[0.02, 0.05, 0.07, 0.10, 0.13, 0.15],
    h=h_filled, mask=mask_all,
)

# 4. LP smoothing with mask=ones
h_smooth = lp_heuristic(mask_all, h_filled, rx0_target)

# 5. rx1 smoothing with mask=ones
h_final = smooth_positive_rx1(mask_all, h_smooth, rx1max=6.0, vertical=vertical)

# 6. Verify -- rx0/rx1 now match what ROMS sees
rx0 = compute_rx0(h_final, mask_all)
Z_r, Z_w = vertical.get_z_levels(h_final, mask_all)
rx1 = compute_rx1(Z_w, mask_all)
```

See `example_shark_bay.py` for a complete working example with `--fill-land` and `--compare` flags.

## Examples

| Script | Description |
|---|---|
| `example_nwa.py` | NWA grid: depth-dependent rx0, boundary blending, two-pass LP + rx1 |
| `example_shark_bay.py` | Shark Bay: sea-only vs fill-over-land comparison |

```bash
python example_nwa.py
python example_shark_bay.py --fill-land     # ROMS-compatible workflow
python example_shark_bay.py --compare       # side-by-side comparison
```

## Module Reference

| Module | Description |
|---|---|
| `grid.py` | `ROMSGrid` class -- mask, bathymetry, NetCDF I/O, `fill_land_depths()`, `depth_dependent_rx0()` |
| `vertical.py` | `VerticalCoords` -- Sc, Cs, Z-level computation |
| `roughness.py` | Vectorised rx0 and rx1 computation |
| `lp_smoothing.py` | LP rx0 smoothing via scipy.optimize.linprog (HiGHS) |
| `heuristic.py` | Connected-component decomposition + parallel LP |
| `iterative.py` | Laplacian, positive/negative rx0, positive rx1 |
| `graph.py` | Connected components, BFS neighbourhood |

## License

This project is licensed under the GNU General Public License v3.0 -- see the [LICENSE](LICENSE) file for details.

## Credits

Reference: Mathieu Dutour Sikirić, Ivica Janeković, Milivoj Kuzmić. 2009. 
A new approach to bathymetry smoothing in sigma-coordinate ocean models, Ocean Modelling,
Volume 29, Issue 2, Pages 128-136, ISSN 1463-5003,
https://doi.org/10.1016/j.ocemod.2009.03.009.
(https://www.sciencedirect.com/science/article/pii/S1463500309000742)
