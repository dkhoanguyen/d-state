#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray
from typing import List, Sequence, Tuple
from objective import Objective
from model import Model

# ============================================================
# Modular, N-robot, decentralized Ergodic CLF-only controller
# - Tunable Gaussian Mixture rho(x)
# - Rectangular domain with cosine Fourier basis
# - Each robot solves a *local* CLF-QP
# - Low-bandwidth shared step = average of F_k(x_j) across robots
# ============================================================

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

class ErgodicCoverageObjective(Objective):
    """
    CLF row (≥ 0 form) for ergodic coverage with A robots:

        g = (-c_clf * E + delta) - sum_j gradE_j(x_j,t,c)^T u_j  >= 0,

    where:
        E(c) = sum_k Lam_k (c_k - phi_k)^2,
        gradE_j = (2/(t*A)) * sum_k Lam_k (c_k - phi_k) * gradF_k(x_j).
    """

    def __init__(self, basis: FourierBasis, phi: NDArray[np.float64]):
        self.basis = basis
        self.phi = np.asarray(phi, dtype=float)

    # one-step (instant) CLF constraint for the whole team, sum over j
    def evaluate(self,
                 model: Model,
                 x: NDArray[np.float64],
                 u: NDArray[np.float64],
                 c: NDArray[np.float64],
                 t_now: float,
                 c_clf: float,
                 A: int = 1,
                 delta: float = 0.0) -> float:
        """
        x: current state of THIS agent (2D position used)
        u: control of THIS agent [ux,uy,eps] (eps=delta for this local row)
        c: running-average Fourier coeffs (nK,)
        t_now: averaging time t (>0)
        c_clf: CLF rate
        A: number of robots (affects the gradient scaling)
        delta: slack for THIS local row

        Returns g >= 0 when the CLF row is satisfied for this agent alone:
            g_local = (-c_clf * E + delta) - gradE^T u
        """
        assert t_now > 0.0
        # metric
        diff = (c - self.phi)
        E = float(np.sum(self.basis.Lam * diff * diff))
        # gradient at this agent
        gradF = self.basis.gradF_mat(x[:2])                 # (nK,2)
        gradE = (2.0 / (t_now * max(A, 1))) * (self.basis.Lam * diff) @ gradF  # (2,)
        # local row (use the local agent's slack delta = u[-1] if you like)
        u_xy = u[:2]
        g_local = (-c_clf * E + float(delta)) - float(gradE @ u_xy)
        return g_local

    # monitoring (no slack)
    def progress(self,
                 model: Model,
                 x: NDArray[np.float64],
                 u: NDArray[np.float64],
                 c: NDArray[np.float64],
                 t_now: float,
                 c_clf: float,
                 A: int = 1) -> float:
        assert t_now > 0.0
        diff = (c - self.phi)
        E = float(np.sum(self.basis.Lam * diff * diff))
        gradF = self.basis.gradF_mat(x[:2])
        gradE = (2.0 / (t_now * max(A, 1))) * (self.basis.Lam * diff) @ gradF
        return (-c_clf * E) - float(gradE @ u[:2])