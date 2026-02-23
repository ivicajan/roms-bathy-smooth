"""ROMS Bathymetry Smoothing Library.

A Python library for smoothing ocean model (ROMS) bathymetry to satisfy
rx0 and rx1 roughness constraints using Linear Programming and iterative methods.

Converted from the MATLAB LP_SMOOTH toolbox by Mathieu Dutour Sikiric,
with modifications by Ivica Janekovic.

Main classes:
    ROMSGrid        - Grid container with NetCDF I/O
    VerticalCoords  - ROMS vertical coordinate parameters

Main functions:
    compute_rx0()          - Haney roughness number
    compute_rx1()          - Beckmann-Haidvogel roughness number
    lp_smooth_rx0()        - LP-based rx0 smoothing (scipy/HiGHS)
    lp_heuristic()         - LP with connected-component decomposition
    smooth_positive_rx0()  - Increase-only rx0 smoothing
    smooth_negative_rx0()  - Decrease-only rx0 smoothing
    laplacian_smooth_rx0() - Selective Laplacian diffusion
    smooth_positive_rx1()  - Increase-only rx1 smoothing (sigma coords)
"""

from .grid import ROMSGrid
from .vertical import VerticalCoords
from .roughness import compute_rx0, compute_rx1
from .lp_smoothing import lp_smooth_rx0
from .heuristic import lp_heuristic
from .iterative import (
    smooth_positive_rx0,
    smooth_negative_rx0,
    laplacian_smooth_rx0,
    smooth_positive_rx1,
)

__all__ = [
    'ROMSGrid',
    'VerticalCoords',
    'compute_rx0',
    'compute_rx1',
    'lp_smooth_rx0',
    'lp_heuristic',
    'smooth_positive_rx0',
    'smooth_negative_rx0',
    'laplacian_smooth_rx0',
    'smooth_positive_rx1',
]
