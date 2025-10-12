#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray
from typing import List, Sequence, Tuple
from objective import Objective
from model import Model

from datatypes import *

class FourierBasis:
    def __init__(self, L: NDArray[np.float64], Kmax: Tuple[int, int]):
        """
        L: [Lx, Ly]
        Kmax: (Kx, Ky) inclusive maximum orders
        """
        self.L = np.asarray(L, dtype=float)
        self.Kmax = (int(Kmax[0]), int(Kmax[1]))
        self.modes = [(kx, ky) for kx in range(self.Kmax[0] + 1)
                                for ky in range(self.Kmax[1] + 1)]
        self.nK = len(self.modes)
        # Sobolev weights Λ_k with exponent (n+1)/2 for n=2
        p = (2 + 1) / 2.0
        self.Lam = np.array([1.0 / (1.0 + kx*kx + ky*ky)**p for (kx, ky) in self.modes], dtype=float)

    def F_vec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """F_k(x) for all modes, shape (nK,). Uses only x[:2]."""
        Lx, Ly = self.L
        Fx = np.empty(self.nK, dtype=float)
        for i, (kx, ky) in enumerate(self.modes):
            Fx[i] = np.cos(np.pi * kx * x[0] / Lx) * np.cos(np.pi * ky * x[1] / Ly)
        return Fx

    def gradF_mat(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """∇F_k(x) for all modes, shape (nK,2). Uses only x[:2]."""
        Lx, Ly = self.L
        G = np.empty((self.nK, 2), dtype=float)
        for i, (kx, ky) in enumerate(self.modes):
            gx = -(np.pi * kx / Lx) * np.sin(np.pi * kx * x[0] / Lx) * np.cos(np.pi * ky * x[1] / Ly)
            gy = -(np.pi * ky / Ly) * np.cos(np.pi * kx * x[0] / Lx) * np.sin(np.pi * ky * x[1] / Ly)
            G[i, 0] = gx
            G[i, 1] = gy
        return G
    
class GaussianMixtureDensity:
    def __init__(self, L, gridN=201):
        self.L = np.asarray(L, dtype=float)
        self.gridN = int(gridN)
        xs = np.linspace(0, self.L[0], self.gridN)
        ys = np.linspace(0, self.L[1], self.gridN)
        self.X, self.Y = np.meshgrid(xs, ys, indexing='xy')
        self.P = np.stack([self.X, self.Y], axis=-1)
        self.dxdy = (self.L[0] / (self.gridN - 1)) * (self.L[1] / (self.gridN - 1))
        self.rho = None

    @staticmethod
    def _gaussian_2d(P, mu, Sigma):
        d = P - mu
        Sinv = np.linalg.inv(Sigma)
        expo = np.einsum('...i,ij,...j->...', d, Sinv, d)
        det = np.linalg.det(Sigma)
        coef = 1.0 / (2.0 * np.pi * np.sqrt(det))
        return coef * np.exp(-0.5 * expo)

    @staticmethod
    def rotated_cov(sig1, sig2, theta_rad):
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        return R @ np.diag([sig1**2, sig2**2]) @ R.T

    def build(self, centers, covs, weights):
        if len(centers) != len(covs) or len(centers) != len(weights):
            raise ValueError("centers, covs, and weights must have equal length.")
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative.")
        w = w / (w.sum() + 1e-16)

        mix = np.zeros(self.P.shape[:2], dtype=float)
        for mu, Sigma, a in zip(centers, covs, w):
            mix += a * self._gaussian_2d(self.P, np.array(mu), np.array(Sigma))

        mix /= (mix.sum() * self.dxdy + 1e-16)  # normalize to integrate to 1
        self.rho = mix
        return self.rho

    def phi_coeffs(self, basis: FourierBasis):
        if self.rho is None:
            raise RuntimeError("Call build(...) first to create rho.")
        phi = np.zeros(basis.nK, dtype=float)
        for i, (kx, ky) in enumerate(basis.modes):
            phi[i] = np.sum(self.rho *
                            np.cos(np.pi * kx * self.X / basis.L[0]) *
                            np.cos(np.pi * ky * self.Y / basis.L[1])) * self.dxdy
        return phi


# ============================================================
# Objective: CLF row (≥0 form) with cached φ
# ============================================================
class ErgodicCoverageObjective(Objective):
    """
    g_local = (-c_clf * E(c) + δ) - ∇E(x,t,c)^T u  >= 0
    E(c) = Σ_k Λ_k (c_k - φ_k)^2
    ∇E   = (2/(t*A)) Σ_k Λ_k (c_k - φ_k) ∇F_k(x)
    """
    def __init__(self, L=(1.0,1.0), Kmax=(4,4)):
        self.basis = FourierBasis(L=np.array(L, dtype=float), Kmax=Kmax)
        self._gmm = GaussianMixtureDensity(L=np.array(L, dtype=float), gridN=201)
        self.phi: Optional[NDArray[np.float64]] = None
        self._cached_target = None  # (centers,covs,weights) tuple to avoid rebuilds

    def set_target(self, centers, covs, weights):
        """Build rho and cache φ once."""
        self._gmm.build(centers, covs, weights)
        self.phi = self._gmm.phi_coeffs(self.basis)
        self._cached_target = (tuple(map(tuple, centers)),
                               tuple(map(tuple, [tuple(row) for row in covs])),
                               tuple(weights))

    def ensure_phi(self, centers, covs, weights):
        key = (tuple(map(tuple, centers)),
               tuple(map(tuple, [tuple(row) for row in covs])),
               tuple(weights))
        if self.phi is None or self._cached_target != key:
            self.set_target(centers, covs, weights)

    def evaluate(self,
                 model: Model,
                 x: NDArray[np.float64],
                 u: NDArray[np.float64],
                 c: NDArray[np.float64],
                 t_now: float,
                 c_clf: float,
                 centers,
                 covs,
                 weights,
                 A: int = 1,
                 delta: float = 0.0) -> float:
        assert t_now > 0.0
        self.ensure_phi(centers, covs, weights)

        diff = (c - self.phi)
        E = float(np.sum(self.basis.Lam * diff * diff))
        gradF = self.basis.gradF_mat(x[:2])
        gradE = (2.0 / (t_now * max(A, 1))) * (self.basis.Lam * diff) @ gradF
        return (-c_clf * E + float(delta)) - float(gradE @ u[:2])

    def progress(self,
                 model: Model,
                 x: NDArray[np.float64],
                 u: NDArray[np.float64],
                 c: NDArray[np.float64],
                 t_now: float,
                 c_clf: float,
                 A: int = 1) -> float:
        assert t_now > 0.0 and self.phi is not None, "Call set_target(...) first."
        diff = (c - self.phi)
        E = float(np.sum(self.basis.Lam * diff * diff))
        gradF = self.basis.gradF_mat(x[:2])
        gradE = (2.0 / (t_now * max(A, 1))) * (self.basis.Lam * diff) @ gradF
        return (-c_clf * E) - float(gradE @ u[:2])