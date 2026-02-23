"""ROMS vertical coordinate computations.

Computes S-coordinate stretching curves (Sc, Cs) and vertical Z levels
for ROMS ocean model grids. Supports Vtransform 1/2 and Vstretching 1/2/3.

Replaces: GRID_GetSc_Cs_V2.m, GetVerticalLevels2.m
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VerticalCoords:
    """ROMS vertical coordinate parameters.

    Attributes:
        N: Number of vertical levels.
        theta_s: Surface stretching parameter (THETA_S in ROMS).
        theta_b: Bottom stretching parameter (THETA_B in ROMS).
        vtransform: Vertical transformation type (1 or 2).
        vstretching: Vertical stretching type (1, 2, or 3).
        hc: Critical depth / TCLINE (m).
    """
    N: int
    theta_s: float
    theta_b: float
    vtransform: int
    vstretching: int
    hc: float

    def get_sc_cs(self):
        """Compute S-coordinate and stretching curves.

        Returns:
            Sc_w: S-coordinate at W-points, shape (N+1,).
            Cs_w: Stretching curve at W-points, shape (N+1,).
            Sc_r: S-coordinate at rho-points, shape (N,).
            Cs_r: Stretching curve at rho-points, shape (N,).
        """
        N = self.N
        Sc_r = -(2 * np.arange(N, 0, -1) - 1) / (2 * N)
        Sc_w = -np.arange(N, -1, -1) / N

        Cs_r = self._stretching(Sc_r)
        Cs_w = self._stretching(Sc_w)

        return Sc_w, Cs_w, Sc_r, Cs_r

    def _stretching(self, sc):
        """Apply stretching function to S-coordinate values."""
        ts = self.theta_s
        tb = self.theta_b

        if self.vstretching == 1:
            cs = ((1 - tb) * (np.sinh(sc * ts) / np.sinh(ts))
                  + tb * (-0.5 + 0.5 * np.tanh(ts * (sc + 0.5))
                          / np.tanh(0.5 * ts)))

        elif self.vstretching == 2:
            Csur = (1 - np.cosh(ts * sc)) / (np.cosh(ts) - 1)
            Cbot = -1 + np.sinh(tb * (sc + 1)) / np.sinh(tb)
            Cweight = (sc + 1) * (1 + (1) * (1 - (sc + 1)))
            cs = Cweight * Csur + (1 - Cweight) * Cbot

        elif self.vstretching == 3:
            Hscale = 3
            alpha = ts
            beta = tb
            Csur = -np.log(np.cosh(Hscale * np.abs(sc) ** alpha)) / np.log(np.cosh(Hscale))
            Cbot = np.log(np.cosh(Hscale * (sc + 1) ** beta)) / np.log(np.cosh(Hscale)) - 1
            Cweight = 0.5 * (1 - np.tanh(Hscale * (sc + 0.5)))
            cs = Cweight * Cbot + (1 - Cweight) * Csur

        else:
            raise ValueError(f"Vstretching must be 1, 2, or 3, got {self.vstretching}")

        return cs

    def get_z_levels(self, h, mask):
        """Compute vertical Z levels from bathymetry.

        Args:
            h: Bathymetry array, shape (eta_rho, xi_rho). Positive depth.
            mask: Land/sea mask, shape (eta_rho, xi_rho). 1=sea, 0=land.

        Returns:
            Z_r: Z at rho-points, shape (N, eta_rho, xi_rho). Negative values.
            Z_w: Z at W-points, shape (N+1, eta_rho, xi_rho). Negative values.
        """
        Sc_w, Cs_w, Sc_r, Cs_r = self.get_sc_cs()

        h_work = h.copy()
        h_work[mask == 0] = 3.0

        eta, xi = h_work.shape
        N = self.N
        hc = self.hc

        Z_r = np.zeros((N, eta, xi))
        Z_w = np.zeros((N + 1, eta, xi))

        if self.vtransform == 1:
            for k in range(N):
                Z_r[k] = hc * Sc_r[k] + (h_work - hc) * Cs_r[k]
            for k in range(N + 1):
                Z_w[k] = hc * Sc_w[k] + (h_work - hc) * Cs_w[k]

        elif self.vtransform == 2:
            for k in range(N):
                Zo = (hc * Sc_r[k] + h_work * Cs_r[k]) / (hc + h_work)
                Z_r[k] = Zo * h_work
            for k in range(N + 1):
                Zo = (hc * Sc_w[k] + h_work * Cs_w[k]) / (hc + h_work)
                Z_w[k] = Zo * h_work

        else:
            raise ValueError(f"Vtransform must be 1 or 2, got {self.vtransform}")

        return Z_r, Z_w
