"""ROMS grid container with NetCDF I/O.

Provides the ROMSGrid class for loading, storing, and writing back
ROMS bathymetry grids from/to NetCDF files.

Replaces: NetCDF I/O in smooth_example_NWA.m
"""

import numpy as np
import netCDF4 as nc


class ROMSGrid:
    """Container for ROMS grid data.

    Attributes:
        mask: Land/sea mask, shape (eta_rho, xi_rho). 1=sea, 0=land.
        h: Current bathymetry (positive depth), shape (eta_rho, xi_rho).
        hraw: Original/raw bathymetry, shape (eta_rho, xi_rho).
        lon_rho: Longitude at rho-points (optional).
        lat_rho: Latitude at rho-points (optional).
        filepath: Source file path.
    """

    def __init__(self, mask, h, hraw=None, lon_rho=None, lat_rho=None,
                 filepath=None):
        self.mask = np.asarray(mask, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.hraw = np.asarray(hraw, dtype=float) if hraw is not None else self.h.copy()
        self.lon_rho = lon_rho
        self.lat_rho = lat_rho
        self.filepath = filepath

    @property
    def shape(self):
        return self.mask.shape

    @property
    def eta_rho(self):
        return self.mask.shape[0]

    @property
    def xi_rho(self):
        return self.mask.shape[1]

    @classmethod
    def from_netcdf(cls, filepath):
        """Load a ROMS grid from a NetCDF file.

        Reads mask_rho, h, and optionally hraw, lon_rho, lat_rho.

        Args:
            filepath: Path to the NetCDF grid file.

        Returns:
            ROMSGrid instance.
        """
        ds = nc.Dataset(filepath, 'r')

        mask = np.squeeze(ds.variables['mask_rho'][:])
        h = np.squeeze(ds.variables['h'][:])

        hraw = None
        if 'hraw' in ds.variables:
            hraw = np.abs(np.squeeze(ds.variables['hraw'][:]))

        lon_rho = None
        if 'lon_rho' in ds.variables:
            lon_rho = ds.variables['lon_rho'][:]

        lat_rho = None
        if 'lat_rho' in ds.variables:
            lat_rho = ds.variables['lat_rho'][:]

        ds.close()

        return cls(mask=mask, h=h, hraw=hraw, lon_rho=lon_rho,
                   lat_rho=lat_rho, filepath=filepath)

    def write_h(self, filepath=None, varname='h'):
        """Write the current bathymetry back to a NetCDF file.

        Args:
            filepath: Target file. Defaults to the source file.
            varname: Variable name to write (default 'h').
        """
        path = filepath or self.filepath
        if path is None:
            raise ValueError("No filepath specified and no source file set.")

        ds = nc.Dataset(path, 'r+')
        ds.variables[varname][:] = self.h
        ds.close()

    def depth_dependent_rx0(self, depths, values, h=None, mask=None):
        """Create a spatially-variable rx0 target based on depth.

        Linearly interpolates rx0 values as a function of bathymetry depth.
        Deeper areas get smaller (stricter) rx0 targets.

        Args:
            depths: Depth breakpoints in any order, e.g. [8000, 4000, 3000, 2000, 1000, 0].
            values: Corresponding rx0 values, e.g. [0.02, 0.05, 0.07, 0.10, 0.13, 0.15].
            h: Bathymetry to use for interpolation. Default: ``self.hraw``.
                When using ``fill_land_depths()``, pass the filled array so
                former-land points get depth-appropriate rx0 targets.
            mask: Mask selecting points to interpolate. Default: ``self.mask``.
                Pass ``np.ones_like(self.mask)`` for the ROMS-compatible
                fill-over-land workflow.

        Returns:
            rx0_target: Array of shape (eta_rho, xi_rho) with spatially-variable rx0.
        """
        if h is None:
            h = self.hraw
        if mask is None:
            mask = self.mask

        # np.interp requires increasing x — sort the pairs
        depths = np.asarray(depths, dtype=float)
        values = np.asarray(values, dtype=float)
        order = np.argsort(depths)
        depths_sorted = depths[order]
        values_sorted = values[order]

        rx0_target = np.full_like(self.h, values_sorted[-1])
        active = mask == 1
        rx0_target[active] = np.interp(h[active], depths_sorted, values_sorted)
        return rx0_target

    def fill_land_depths(self, hmin=None):
        """Extrapolate sea depths over land by solving Laplace's equation.

        Replaces land-point depths with a smooth extrapolation of neighbouring
        sea depths.  Sea-point values are kept unchanged (except for optional
        hmin clipping).  This removes artificial gradients at land-sea
        boundaries so that subsequent smoothing with ``mask=np.ones(...)``
        sees a smooth starting field everywhere.

        The method iteratively averages land-point values from their
        4-connected neighbours until convergence, which is equivalent to
        solving nabla^2 h = 0 on the land domain with Dirichlet BCs from
        the sea.

        Args:
            hmin: Optional minimum depth applied to ALL points before
                  filling.  This ensures shallow fill values (e.g. hraw=1)
                  are clipped to a physically reasonable floor that matches
                  the ROMS model configuration.

        Returns:
            h_filled: Array of shape (eta_rho, xi_rho) with land depths
                      replaced by smoothly extrapolated values.
        """
        h_filled = self.hraw.copy()

        # Clip to hmin everywhere BEFORE filling, so the sea BCs that
        # propagate into land already respect the minimum depth.
        if hmin is not None:
            h_filled = np.maximum(h_filled, hmin)

        land = self.mask == 0
        sea = ~land

        if not land.any():
            return h_filled

        tol = 1e-4
        max_iter = 10000

        # Wipe land values so only sea BCs propagate inward
        h_filled[land] = np.nan

        for it in range(max_iter):
            h_old = h_filled.copy()

            # 4-connected neighbour sum and count, ignoring NaN
            nsum = np.zeros_like(h_filled)
            ncnt = np.zeros_like(h_filled)

            for src, dst in [
                (slice(None, -1), slice(1, None)),   # shift south → north
                (slice(1, None),  slice(None, -1)),   # shift north → south
            ]:
                valid = ~np.isnan(h_filled[src, :])
                nsum[dst, :] += np.where(valid, h_filled[src, :], 0.0)
                ncnt[dst, :] += valid

            for src, dst in [
                (slice(None, -1), slice(1, None)),   # shift east → west
                (slice(1, None),  slice(None, -1)),   # shift west → east
            ]:
                valid = ~np.isnan(h_filled[:, src])
                nsum[:, dst] += np.where(valid, h_filled[:, src], 0.0)
                ncnt[:, dst] += valid

            # Update only land points that have at least one valid neighbour
            update = land & (ncnt > 0)
            h_filled[update] = nsum[update] / ncnt[update]

            # Convergence: max change on land points that are no longer NaN
            filled_land = land & ~np.isnan(h_filled)
            if not filled_land.any():
                continue
            old_vals = h_old[filled_land]
            new_vals = h_filled[filled_land]
            changed = np.isnan(old_vals) | (np.abs(new_vals - old_vals) > tol)
            if not changed.any():
                break

        # Any remaining NaN (disconnected land) → mean of sea depths
        still_nan = np.isnan(h_filled)
        if still_nan.any():
            h_filled[still_nan] = np.nanmean(h_filled[sea])

        return h_filled

    def boundary_taper(self, width, value_edge=0.0, value_interior=1.0):
        """Create a linear taper weight from edges toward interior.

        Used for blending bathymetry at boundaries.

        Args:
            width: Taper width in grid points.
            value_edge: Weight at the boundary (default 0).
            value_interior: Weight in the interior (default 1).

        Returns:
            weights: Array of shape (eta_rho, xi_rho), values from
                     value_edge at boundary to value_interior inside.
        """
        w = np.full_like(self.h, value_interior)
        ramp = np.linspace(value_edge, value_interior, width, endpoint=False)
        for k in range(width):
            v = ramp[k]
            w[k:self.eta_rho - k, k] = np.minimum(w[k:self.eta_rho - k, k], v)  # west
            w[k, k:self.xi_rho - k] = np.minimum(w[k, k:self.xi_rho - k], v)    # south
            w[self.eta_rho - 1 - k, k:self.xi_rho - k] = np.minimum(
                w[self.eta_rho - 1 - k, k:self.xi_rho - k], v)                   # north
            w[k:self.eta_rho - k, self.xi_rho - 1 - k] = np.minimum(
                w[k:self.eta_rho - k, self.xi_rho - 1 - k], v)                   # east
        return w
