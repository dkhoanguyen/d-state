#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Optional
from scipy.optimize import minimize, Bounds

from executor.executor import Executor
from datatypes import *
from evaluator import *               # optional, same as your example imports
from model import Model
from objective import Objective, FourierBasis, ErgodicCoverageObjective

class ErgodicCoverageExecutor(Executor):
    """
    Decision per step z_k = [u (m_u dims), δ], bounds on u and δ.
    AL outer loop over horizon-wise CLF rows g_k(z_k) ≤ 0.
    """
    def __init__(self, 
                 w_effort: float = 0.5, 
                 k_eps: float = 1e3, 
                 al_iters: int = 1, 
                 rho0: float = 10.0):
        self.w_effort = float(w_effort)  # 0.5 * ||u||^2 multiplier
        self.k_eps = float(k_eps)        # δ^2 penalty
        self.al_iters = int(al_iters)    # outer AL iterations
        self.rho0 = float(rho0)          # initial ρ

        L=(1.0,1.0)
        Kmax=(4,4)
        self.basis = FourierBasis(L=np.array(L, dtype=float), Kmax=Kmax)
        self._gmm = GaussianMixtureDensity(L=np.array(L, dtype=float), gridN=201)

        self._init = False
        self.t_now = 1e-2
        self.c = None

        self.rho = 1.0

    def init(self):
        pass

    def calculate_reward(
        self,
        agent: Agent,
        task: ReachGoalTask,
        other_agents: list[Agent], 
        horizon: float,
        joint_actions: NDArray[np.float64]
    ) -> float:
        pass

    def calculate_cost(
        self,
        agent: Agent,
        task: ReachGoalTask,
        other_agents: list[Agent],
        objective_controller: Objective,
        horizon: float,
        joint_actions: NDArray[np.float64]
    ) -> float:
        pass

    def execute(self,
                agent: Agent,
                other_agents: List[Agent],
                scenario: Scenario,
                task: ErgodicCoverageTask,
                lambda_k: NDArray[np.float64],   # can pass zeros; will be updated here
                objective: ErgodicCoverageObjective,
                horizon: int,
                maxiter: int = 50,
                ftol: float = 1e-4,
                eps: float = 1e-6) -> Tuple[NDArray[np.float64], bool]:

        U0 = np.tile(agent.control, horizon)
        u_max = agent.u_bounds[0, 1]
        u_min = agent.u_bounds[0, 0]
        bnds = [(u_min, u_max)] * (2 * horizon) + \
            [(-np.inf, np.inf)] * horizon

        constraints = []

        self._gmm.build(task.centers, task.covs, task.weights)
        self.phi = self._gmm.phi_coeffs(self.basis)

        if task.alphas is None:
            alphas = np.ones(len(other_agents)) / len(other_agents)
        else:
            alphas = task.alphas / (np.sum(task.alphas) + 1e-16)
        # print(alphas)

        # Aggregate all states
        X = np.empty((0,2))
        for local_agent in other_agents:
            X = np.vstack((X,local_agent.state))

        # LOCAL evaluations → low-bandwidth average
        F_list = np.stack([self.basis.F_vec(x) for x in X])  # (N, nK)
        F_avg = F_list.mean(axis=0)

        # Shared running-average update
        if self.c is None:
            self.c = np.zeros_like(self.phi)
        self.c = self.c + task.dt * ((F_avg - self.c) / self.t_now)

        E = self.ergodic_metric()
        gE_J = self.gradE_at(agent.state, len(other_agents))

        print(np.linalg.norm(gE_J))

        rhs_clf = -alphas[agent.id] * task.c_clf * E
        u_max = 1.0
        
        def f_cost(z):
            u = z[:2]; d = z[2]
            return 0.5 * np.dot(u, u) + task.w_clf * (d ** 2)

        def g_ineq(z):
            u = z[:2]; d = z[2]
            return float(np.dot(gE_J, u) - d - rhs_clf)  # <= 0 desired
        
        # PHR AL objective
        def Phi(z, lam):
            g = g_ineq(z)
            return f_cost(z) + lam * g + 0.5 * g ** 2
        
        z = np.zeros(3)  # start at 0
        lam = 0.0

        fun = lambda zz: Phi(zz, lam)
        res = minimize(fun, z, method='SLSQP', bounds=bnds)
        z = res.x if res.success else z

        g = g_ineq(z)
        lam = max(0.0, lam + 1 * g)
        self.t_now += task.dt

        return z,res.success
    
    def _compute_cost(self,
                      model: Model,
                      U: np.ndarray,
                      x_0: np.ndarray,
                      horizon: int,
                      dt: float,
                      w_eps: float = 0.000001,
                      k_eps: float = 1000.0) -> float:
        cost = 0.0
        trajectory = self._simulate_trajectory(model, x_0, U, horizon, dt)

        for k in range(horizon):
            u = U[3*k:3*k+2]  # Extract [u_x_k, u_y_k]
            # Control effort
            cost += 0.5 * np.linalg.norm(u)**2

            # Ergodic metric 


    def ergodic_metric(self):
        return float(np.sum(self.basis.Lam * (self.c - self.phi) ** 2))

    def gradE_at(self, x, N_agents):
        gradF = self.basis.gradF_mat(x)         # (nK, 2)
        weights = self.basis.Lam * (self.c - self.phi)  # (nK,)
        return (2.0 / (self.t_now * N_agents)) * (weights @ gradF)  # (2,)
    
    def _simulate_trajectory(self,
                             model: Model,
                             x0: NDArray[np.float64],
                             U: NDArray[np.float64],
                             N: int,
                             dt: float) -> NDArray[np.float64]:
        """Simulate the trajectory given initial state and control inputs."""
        trajectory = [x0.copy()]
        x = x0.copy()
        for k in range(N):
            u = U[3*k:3*k+2]
            x = model.f(x, u, dt)
            trajectory.append(x.copy())
        return trajectory
    
    def _poisson_binomial_distribution(self, probabilities: NDArray[np.float64]) -> float:
        """
        Compute the Poisson binomial distribution P(X = k) using FFT with correct ordering.
        """
        N = len(probabilities)
        p = np.array(probabilities)

        if not np.all((0 <= p) & (p <= 1)):
            raise ValueError("Probabilities must be between 0 and 1")

        n_points = N + 1
        t = 2 * np.pi * np.arange(n_points) / n_points

        phi_X = np.ones(n_points, dtype=complex)
        for i in range(0, N):
            phi_X_i = (1 - p[i]) + p[i] * np.exp(1j * t)
            phi_X *= phi_X_i

        # Compute inverse FFT and scale by 1/(N+1), then shift to correct order
        probs = np.fft.ifft(phi_X) / (N + 1)
        probs = np.fft.fftshift(probs)  # Shift to align k = 0 to N
        probs = np.real(probs)

        # Normalize to ensure sum = 1
        probs_sum = np.sum(probs)
        if abs(probs_sum - 1) > 1e-10:
            probs = probs / probs_sum

        probs = np.maximum(probs, 0)

        # Reorder to ensure k = 0 is first (after shift, take the middle section)
        probs = np.roll(probs, N // 2)
        probs = np.flip(probs)

        return probs
    
    def preprocess(self,
                   joint_action: NDArray[np.float64]):
        prob = self._poisson_binomial_distribution(joint_action)
        return np.sum(prob)
    

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