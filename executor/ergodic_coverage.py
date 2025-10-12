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
        bounds = [(u_min, u_max)] * (2 * horizon) + \
            [(-np.inf, np.inf)] * horizon

        constraints = []

        self._gmm.build(task.centers, task.covs, task.weights)
        self.phi = self._gmm.phi_coeffs(self.basis)

        if task.alphas is None:
            alphas = np.ones(len(other_agents)) / len(other_agents)
        else:
            alphas = task.alphas / (np.sum(task.alphas) + 1e-16)

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

        rhs_clf = -alphas[agent.id] * task.c_clf * E
        u_max = 1.0
        bnds = Bounds(lb=[-u_max, -u_max, 0.0],
                      ub=[u_max,  u_max, np.inf])
        
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

        print(z)

        g = g_ineq(z)
        lam = max(0.0, lam + 10 * g)
        self.t_now += task.dt

        return z,res.success
    
    def _compute_cost(self,
                      model: Model,
                      U: np.ndarray,
                      x_0: np.ndarray,
                      task: ErgodicCoverageTask,
                      lambda_k: np.ndarray,
                      objective: ErgodicCoverageObjective,
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


    # # -------- AL cost over the horizon --------
    # def _al_cost(self,
    #              U: NDArray[np.float64],
    #              model: Model,
    #              x0: NDArray[np.float64],
    #              task: ErgodicCoverageTask,
    #              objective: ErgodicCoverageObjective,
    #              horizon: int,
    #              lam: NDArray[np.float64],
    #              rho: float,
    #              n_dec_step: int,
    #              m_u: int) -> float:

    #     dt = float(task.dt)
    #     A = int(task.A)
    #     c_clf = float(task.c_clf)

    #     c_hat = task.c_init.copy()
    #     t_hat = float(task.t_init)
    #     x = x0.copy()

    #     J = 0.0
    #     for k in range(horizon):
    #         u_k = U[n_dec_step*k : n_dec_step*k + m_u]
    #         delta_k = U[n_dec_step*k + m_u]

    #         # effort + slack
    #         J += self.w_effort * float(u_k @ u_k) + self.k_eps * (delta_k**2)

    #         # Propagate dynamics
    #         x = model.f(x, u_k, dt)

    #         # Predict F_avg (single-agent approx)
    #         F_local = objective.basis.F_vec(x[:2])
    #         F_avg_k = F_local / max(A, 1)

    #         # Update c,t
    #         c_hat = c_hat + dt * ((F_avg_k - c_hat) / t_hat)
    #         t_hat = t_hat + dt

    #         # CLF row => g_k ≤ 0 desired
    #         g_k = objective.evaluate(
    #             model=model,
    #             x=x,
    #             u=np.hstack([u_k, delta_k]),
    #             c=c_hat,
    #             t_now=t_hat,
    #             c_clf=c_clf,
    #             centers=task.centers,
    #             covs=task.covs,
    #             weights=task.weights,
    #             A=A,
    #             delta=delta_k
    #         )

    #         # PHR AL term: 0.5*ρ*(max(0, g_k + λ_k/ρ))^2 - (λ_k^2)/(2ρ)
    #         s = max(0.0, g_k + lam[k]/rho)
    #         J += 0.5 * rho * (s*s) - 0.5 * (lam[k]*lam[k]) / rho

    #     return float(J)

    # # -------- Rollout once and return all g_k --------
    # def _rollout_and_g(self,
    #                    U: NDArray[np.float64],
    #                    model: Model,
    #                    x0: NDArray[np.float64],
    #                    task: ErgodicCoverageTask,
    #                    objective: ErgodicCoverageObjective,
    #                    horizon: int,
    #                    n_dec_step: int,
    #                    m_u: int) -> NDArray[np.float64]:

    #     dt = float(task.dt)
    #     A = int(task.A)
    #     c_clf = float(task.c_clf)
    #     c_hat = task.c_init.copy()
    #     t_hat = float(task.t_init)
    #     x = x0.copy()

    #     g_all = np.zeros(horizon, dtype=float)
    #     for k in range(horizon):
    #         u_k = U[n_dec_step*k : n_dec_step*k + m_u]
    #         delta_k = U[n_dec_step*k + m_u]
    #         x = model.f(x, u_k, dt)

    #         F_local = objective.basis.F_vec(x[:2])
    #         F_avg_k = F_local / max(A, 1)
    #         c_hat = c_hat + dt * ((F_avg_k - c_hat) / t_hat)
    #         t_hat = t_hat + dt

    #         g_all[k] = objective.evaluate(
    #             model=model,
    #             x=x,
    #             u=np.hstack([u_k, delta_k]),
    #             c=c_hat,
    #             t_now=t_hat,
    #             c_clf=c_clf,
    #             centers=task.centers,
    #             covs=task.covs,
    #             weights=task.weights,
    #             A=A,
    #             delta=delta_k
    #         )
    #     return g_all
    
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