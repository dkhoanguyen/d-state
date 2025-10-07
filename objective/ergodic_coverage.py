#!/usr/bin/env python3
import numpy as np
from typing import Tuple
from objective import Objective
from model import Model


class ErgodicCoverageObjective(Objective):
    """Ergodic control objective to match a target distribution using a CBF."""
    def __init__(self,
                 state_dim: int = 2,
                 control_dim: int = 2,
                 domain: np.ndarray = np.array([1.0, 1.0]),
                 num_cell: int = 50,
                 num_k_per_dim: int = 5,
                 gamma: float = 10.0,
                 gaussian_params: List[Tuple[float, np.ndarray, np.ndarray]] = [
                     (0.5, np.array([0.35, 0.38]), np.array([[0.01, 0.004], [0.004, 0.01]])),
                     (0.2, np.array([0.68, 0.25]), np.array([[0.005, -0.003], [-0.003, 0.005]])),
                     (0.3, np.array([0.56, 0.64]), np.array([[0.008, 0.0], [0.0, 0.004]]))
                 ]):
        """Initialize the ergodic objective.

        Args:
            state_dim: Dimension of the state space (default: 2 for px, py).
            control_dim: Dimension of the control input (default: 2 for ux, uy).
            domain: Domain size [Lx, Ly] (default: [1.0, 1.0]).
            num_cell: Number of grid points per dimension (default: 50).
            num_k_per_dim: Number of Fourier basis terms per dimension (default: 5).
            gamma: CBF parameter (default: 10.0).
            gaussian_params: List of (weight, mean, covariance) for Gaussian mixture.
        """
        self._state_dim = state_dim
        self._control_dim = control_dim
        self._domain = domain
        self._num_cell = num_cell
        self._num_k_per_dim = num_k_per_dim
        self._gamma = gamma
        self._gaussian_params = gaussian_params

        # Initialize grid and Fourier coefficients
        self._grids, self._ks, self._dx, self._dy = self._initialize_grid()
        self._phi_k, _ = self._compute_phi_k()

    def _pdf(self, x: np.ndarray) -> float:
        """Compute the probability density function of the Gaussian mixture."""
        from scipy.stats import multivariate_normal as mvn
        return sum(w * mvn.pdf(x, mean, cov) for w, mean, cov in self._gaussian_params)

    def _initialize_grid(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Discretize the domain and compute Fourier basis indices."""
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self._domain[0], self._num_cell),
            np.linspace(0, self._domain[1], self._num_cell)
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        ks_dim1, ks_dim2 = np.meshgrid(
            np.arange(self._num_k_per_dim), np.arange(self._num_k_per_dim)
        )
        ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T
        dx = self._domain[0] / (self._num_cell - 1)
        dy = self._domain[1] / (self._num_cell - 1)
        return grids, ks, dx, dy

    def _compute_phi_k(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fourier coefficients Phi_k for the target distribution."""
        pdf_vals = self._pdf(self._grids)
        phi_k = np.zeros(self._ks.shape[0])
        for i, k_vec in enumerate(self._ks):
            fk_vals = np.prod(np.cos(np.pi * k_vec / self._domain * self._grids), axis=1)
            hk = np.sqrt(np.sum(np.square(fk_vals)) * self._dx * self._dy)
            fk_vals = fk_vals / hk if hk > 0 else fk_vals
            phi_k[i] = np.sum(fk_vals * pdf_vals) * self._dx * self._dy
        return phi_k, pdf_vals

    def _f_k(self, px: float, py: float, k: np.ndarray) -> float:
        """Fourier basis function."""
        k1, k2 = k
        return np.cos(np.pi * k1 * px) * np.cos(np.pi * k2 * py)

    def _lambda_k(self, k: np.ndarray) -> float:
        """Weight for ergodic metric."""
        k1, k2 = k
        norm_k = np.sqrt(k1**2 + k2**2)
        return 1.0 / (1.0 + norm_k**2)**1.5

    def _compute_c_k(self, t: float, trajectory: List[Tuple[float, np.ndarray]]) -> np.ndarray:
        """Compute time-averaged Fourier coefficients c_k(t)."""
        num_points = len(trajectory)
        if num_points == 0:
            return np.zeros(self._ks.shape[0])
        c_k = np.zeros(self._ks.shape[0])
        for i, k in enumerate(self._ks):
            sum_f = sum(self._f_k(x_s[0], x_s[1], k) for _, x_s in trajectory)
            c_k[i] = sum_f / num_points
        return c_k

    def _h(self, x: np.ndarray, t: float, c_k: np.ndarray) -> float:
        """Control Barrier Function (CBF)."""
        px, py = x[0], x[1]
        return sum(
            self._lambda_k(k) * (self._phi_k[i] - c_k[i]) * self._f_k(px, py, k)
            for i, k in enumerate(self._ks)
        )

    def _grad_h(self, x: np.ndarray, t: float, c_k: np.ndarray) -> np.ndarray:
        """Gradient of the CBF."""
        px, py = x[0], x[1]
        grad = np.zeros(self._state_dim)
        for i, k in enumerate(self._ks):
            k1, k2 = k
            grad_f = np.array([
                -np.pi * k1 * np.sin(np.pi * k1 * px) * np.cos(np.pi * k2 * py),
                -np.pi * k2 * np.cos(np.pi * k1 * px) * np.sin(np.pi * k2 * py)
            ])
            grad += self._lambda_k(k) * (self._phi_k[i] - c_k[i]) * grad_f
        return grad

    def _dh_dt(self, x: np.ndarray, t: float, c_k: np.ndarray) -> float:
        """Time derivative of the CBF."""
        if t <= 1e-6:
            return 0.0
        px, py = x[0], x[1]
        return sum(
            self._lambda_k(k) * (-1/t) * (self._f_k(px, py, k) - c_k[i]) * self._f_k(px, py, k)
            for i, k in enumerate(self._ks)
        )

    def evaluate(self,
                 model: Model,
                 x: np.ndarray,
                 u: np.ndarray,
                 t: float,
                 trajectory: List[Tuple[float, np.ndarray]]) -> float:
        """Evaluate the CBF constraint: dh/dt + dot(∇h, dx/dt) + gamma * h + epsilon >= 0.

        Args:
            model: System model with dynamics dx(x, u).
            x: State vector [px, py].
            u: Control input [ux, uy, epsilon].
            t: Current time.
            trajectory: List of (time, state) tuples for computing c_k.

        Returns:
            CBF constraint value (scalar, should be non-negative).
        """
        c_k = self._compute_c_k(t, trajectory)
        dx = model.dx(x, u[:-1])
        dh_dx = self._grad_h(x, t, c_k)
        dh_dt = self._dh_dt(x, t, c_k)
        h_val = self._h(x, t, c_k)
        epsilon = u[-1]  # Slack variable
        return dh_dt + np.dot(dh_dx, dx) + self._gamma * h_val + epsilon

    def progress(self,
                 model: Model,
                 x: np.ndarray,
                 u: np.ndarray,
                 t: float,
                 trajectory: List[Tuple[float, np.ndarray]]) -> float:
        """Compute the CBF progress without the slack variable.

        Args:
            model: System model with dynamics dx(x, u).
            x: State vector [px, py].
            u: Control input [ux, uy, epsilon].
            t: Current time.
            trajectory: List of (time, state) tuples for computing c_k.

        Returns:
            CBF progress value (scalar).
        """
        c_k = self._compute_c_k(t, trajectory)
        dx = model.dx(x, u[:-1])
        dh_dx = self._grad_h(x, t, c_k)
        dh_dt = self._dh_dt(x, t, c_k)
        h_val = self._h(x, t, c_k)
        return dh_dt + np.dot(dh_dx, dx) + self._gamma * h_val