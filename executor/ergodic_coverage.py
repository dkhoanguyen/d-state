#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Optional
from scipy.optimize import minimize, Bounds
from scipy.stats import multivariate_normal as mvn

from executor.executor import Executor
from datatypes import *
from evaluator import *               # optional, same as your example imports
from model import Model
from objective import Objective


@dataclass
class BasisPack:
    phi: np.ndarray      # (K,)
    Lam: np.ndarray      # (K,)
    ks: np.ndarray       # (K,2)
    Fk_grid: np.ndarray  # (K, M)
    dxdy: float
    hk: np.ndarray       # (K,)
    pdf_img: Optional[np.ndarray] = None
    gx_shape: Optional[Tuple[int, int]] = None


@dataclass
class ErgodicParams:
    L: np.ndarray                    # domain, shape (2,)
    num_k_per_dim: int               # number of cosine modes per axis
    num_cell: int                    # grid resolution per axis
    c_clf: float = 1.0               # CLF gain
    include_cbf: bool = False        # keep False unless you also pass k1,k2
    # HOCBF params (used only if include_cbf=True)
    k1: float = 2.0
    k2: float = 2.0
    w_xi: float = 1000.0             # slack cost
    u_max: float = 0.1               # max control magnitude per axis
    dt: float = 0.1                  # agent time step
    # random drift for moving targets (0 = static)
    drift_std: float = 0.0


class ErgodicCoverageExecutor(Executor):
    """
    Decision per step z_k = [u (m_u dims), δ], bounds on u and δ.
    AL outer loop over horizon-wise CLF rows g_k(z_k) ≤ 0.
    """

    def __init__(self,
                 num_k_per_dim: int = 10,
                 L: Tuple[float, float] = (1.0, 1.0),
                 num_cell: int = 100,
                 c_clf: float = 1.0,
                 w_xi: float = 1000.0,
                 u_sat: float = 0.05,      # matches the demo main
                 t_init: float = 1e-2):
        self._num_k_per_dim = int(10)
        self._L = np.array(L, dtype=float)
        self._num_cell = int(num_cell)
        self._c_clf = float(c_clf)
        self._w_xi = float(w_xi)
        self._u_sat = float(u_sat)

        # time-like state for coefficient dynamics
        self.t_now = float(t_init)

        # rng for optional target drift; initialized on first execute with scenario.seed if available
        self._rng = None

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
                lambda_k: NDArray[np.float64],
                horizon: int,
                maxiter: int = 50,
                ftol: float = 1e-4,
                eps: float = 1e-6) -> Tuple[Agent, NDArray[np.float64], bool]:
        # Ergodic controller parameters
        params = ErgodicParams(
            L=self._L,
            num_k_per_dim=10,
            num_cell=100,
            c_clf=1.0,
            include_cbf=False,
            w_xi=1000.0,
            u_max=0.8,
            dt=0.1,
            drift_std=0.0   # set >0 to see a moving target over time
        )
        basis = ErgodicCoveragePreprocessor.preprocess_target(
            task.centers, task.covs, task.weights, params, None)

        E, dotE, beta, A_i, dotc_i = self._build_metrics(
            agent=agent, others=other_agents, basis=basis, t_now=self.t_now, L=params.L
        )
        u, new_lambda = self._solve_local_qp(
            A_i=A_i, E=E, dotE=dotE, beta=beta,
            goal_lambda=agent.goal_lambda,
            N_total=len(other_agents), params=params
        )
        agent.goal_lambda = new_lambda
        a_updated = self._update_local_c_and_state(
            agent=agent, basis=basis, params=params, t_now=self.t_now,
            u=u
        )
        agent.c = a_updated.c
        agent.c_dot = a_updated.c_dot
        # agent.state = a_updated.state
        self.t_now += params.dt
        return agent, u, True

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

    def _F_stack(
        self,
        x: NDArray[np.float64],
        ks: NDArray[np.float64],
        L: NDArray[np.float64],
        hk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        cx = np.cos(np.pi * ks[:, 0] * x[0] / L[0])
        cy = np.cos(np.pi * ks[:, 1] * x[1] / L[1])
        fk_raw = cx * cy
        return fk_raw / hk

    def _gradF_stack(
        self,
        x: NDArray[np.float64],
        ks: NDArray[np.float64],
        L: NDArray[np.float64],
        hk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        kx = ks[:, 0]
        ky = ks[:, 1]
        ax = np.pi * kx / L[0]
        ay = np.pi * ky / L[1]
        cosx, cosy = np.cos(ax * x[0]), np.cos(ay * x[1])
        sinx, siny = np.sin(ax * x[0]), np.sin(ay * x[1])
        dcosx = -ax * sinx
        dcosy = -ay * siny
        gx = (dcosx * cosy) / hk
        gy = (cosx * dcosy) / hk
        return np.stack([gx, gy], axis=1)

    def _build_metrics(
        self,
        agent: Agent,
        others: List[Agent],
        basis: BasisPack,
        t_now: float,
        L: NDArray[np.float64]
    ) -> Tuple[float, float, float, NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns:
        E, dotE, beta, A_i, dotc_i
        """
        K = basis.ks.shape[0]
        c_local = agent.c if agent.c is not None else np.zeros(K, dtype=float)

        # consensus on c (logical average of exchanged vectors)
        c_stack = [c_local]
        for ag in others:
            if ag.id == agent.id:
                continue
            c_stack.append(
                ag.c if ag.c is not None else np.zeros(K, dtype=float))
        mean_c = np.mean(np.vstack(c_stack), axis=0)

        resid = (mean_c - basis.phi)
        E = np.sum(basis.Lam * resid**2)

        F_i = self._F_stack(agent.state, basis.ks, L=L, hk=basis.hk)
        G_i = self._gradF_stack(agent.state, basis.ks, L=L, hk=basis.hk)
        A_i = (2.0 / max(t_now, 1e-6)) * (basis.Lam * resid) @ G_i

        # local dot c (only local info)
        dotc_i = (F_i - c_local) / max(t_now, 1e-6)

        # gather others' dotc if they publish; fallback to their own estimate
        dotc_list = [dotc_i]
        for ag in others:
            if ag.id == agent.id:
                continue
            if ag.c_dot is not None:
                dotc_list.append(ag.c_dot)
            else:
                # conservative fallback from their current state & c
                c_other = ag.c if ag.c is not None else np.zeros(
                    K, dtype=float)
                F_other = F_stack(ag.state, basis.ks, L=L, hk=basis.hk)
                dotc_list.append((F_other - c_other) / max(t_now, 1e-6))

        dotc = np.mean(np.vstack(dotc_list), axis=0)
        dotE = 2.0 * np.sum(basis.Lam * resid * dotc)
        beta = 2.0 * np.sum(
            basis.Lam * (
                dotc**2 + resid * (-(1.0/max(t_now, 1e-6))*dotc -
                                   (1.0/(max(t_now, 1e-6)**2))*(max(t_now, 1e-6)*dotc))
            )
        )
        return E, dotE, beta, A_i, dotc_i

    def _solve_local_qp(
        self,
        A_i: np.ndarray,
        E: float,
        dotE: float,
        beta: float,
        goal_lambda: float,
        N_total: int,
        params: ErgodicParams
    ) -> Tuple[np.ndarray, float]:
        """
        Returns:
        u (2,), goal_lambda_new
        """
        rhs_clf = (-params.c_clf * E) / max(N_total, 1)
        gE_i = A_i.copy()

        def f_cost(z):
            u = z[:2]
            xi = z[2]
            return 0.5*np.dot(u, u) + 100*(xi**2)

        def g1(z):
            u = z[:2]
            xi = z[2]
            return float(np.dot(gE_i, u) - xi - rhs_clf)

        def Phi(z, lam1):
            r1 = g1(z)
            return f_cost(z) + lam1*r1 + 0.5*r1**2

        bnds = Bounds(lb=[-10000, -10000, 0.0],
                      ub=[10000,  10000, np.inf])

        res = minimize(lambda z: Phi(z, goal_lambda),
                       x0=np.zeros(3), method="SLSQP", bounds=bnds)
        z = np.zeros(3) if not res.success else res.x
        u = z[:2]
        n = np.linalg.norm(u)
        if n > 0.1:
            u = (u/n) * 0.1

        # dual update
        r1 = max(0.0, g1(z))
        goal_lambda_new = max(0.0, goal_lambda + 1.0*r1)
        return u, goal_lambda_new

    def _update_local_c_and_state(
        self,
        agent: Agent,
        basis: BasisPack,
        params: ErgodicParams,
        t_now: float,
        u: np.ndarray
    ) -> Agent:
        """
        Integrate state and update local coefficient vector using ONLY local quantities.
        Returns updated Agent (copy).
        """
        K = basis.ks.shape[0]
        new_state = agent.model.f(agent.state, u, params.dt)
        # clamp to domain
        new_state = np.minimum(np.maximum(new_state, 0.0), params.L)
        # agent.state = new_state
        if agent.c is None:
            agent.c = np.zeros(K, dtype=float)
        # local F_i and coefficient dynamics
        F_i_new = self._F_stack(agent.state, basis.ks, params.L, basis.hk)
        agent.c = agent.c + params.dt * \
            ((F_i_new - agent.c) / max(t_now, 1e-6))
        agent.c_dot = (F_i_new - agent.c) / max(t_now, 1e-6)

        return agent


class ErgodicCoveragePreprocessor:
    @staticmethod
    def build_modes(num_k_per_dim: int) -> np.ndarray:
        kx, ky = np.meshgrid(np.arange(num_k_per_dim),
                             np.arange(num_k_per_dim), indexing="xy")
        return np.stack([kx.ravel(), ky.ravel()], axis=1)

    @staticmethod
    def build_grid(num_cell: int, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        gx, gy = np.meshgrid(np.linspace(0, L[0], num_cell),
                             np.linspace(0, L[1], num_cell), indexing="xy")
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        dx = L[0] / (num_cell - 1)
        dy = L[1] / (num_cell - 1)
        return gx, gy, grid, dx * dy

    @staticmethod
    def build_hk_basis(ks: np.ndarray, grid: np.ndarray, dxdy: float, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        K = ks.shape[0]
        Fk_grid = np.zeros((K, grid.shape[0]), dtype=float)
        hk = np.zeros(K, dtype=float)
        for i, k in enumerate(ks):
            fk_raw = np.prod(np.cos(np.pi * k / L * grid), axis=1)
            hk[i] = np.sqrt(np.sum(fk_raw**2) * dxdy)
            Fk_grid[i, :] = fk_raw / hk[i]
        return hk, Fk_grid

    @staticmethod
    def sobolev_weight(kx: int, ky: int, n: int = 2) -> float:
        return 1.0 / (1.0 + kx*kx + ky*ky)**((n + 1)/2.0)

    @staticmethod
    def gmm_pdf_vals(grid: np.ndarray,
                     dxdy: float,
                     gmm_params: List[Tuple[float, np.ndarray, np.ndarray]]) -> np.ndarray:
        vals = 0.0
        for (w, m, C) in gmm_params:
            vals += w * mvn.pdf(grid, mean=m, cov=C)
        mass = np.sum(vals) * dxdy
        return vals / mass

    @staticmethod
    def project_target_to_phi(Fk_grid: np.ndarray,
                              grid: np.ndarray,
                              gmm_params: List[Tuple[float, np.ndarray, np.ndarray]],
                              dxdy: float) -> Tuple[np.ndarray, np.ndarray]:
        rho = ErgodicCoveragePreprocessor.gmm_pdf_vals(grid, dxdy, gmm_params)
        phi = (Fk_grid @ rho) * dxdy
        return phi, rho

    @staticmethod
    def reconstruct_from_phi(phi: np.ndarray, Fk_grid: np.ndarray, dxdy: float, gx_shape: Tuple[int, int]) -> np.ndarray:
        recon = (phi @ Fk_grid)
        recon = np.maximum(recon, 0.0)
        s = np.sum(recon) * dxdy
        if s > 0:
            recon /= s
        return recon.reshape(gx_shape)

    @staticmethod
    def preprocess_target(centers: np.ndarray,
                          covs: np.ndarray,
                          weights: np.ndarray,
                          params: ErgodicParams,
                          rng: Optional[np.random.Generator] = None) -> BasisPack:
        # Optionally drift means a bit to simulate time-varying target
        gmm_params = []
        for w, m, C in zip(weights, centers, covs):
            m = np.array(m, dtype=float)
            if rng is not None and params.drift_std > 0:
                m = np.clip(m + rng.normal(0.0, params.drift_std, size=2),
                            a_min=[0.0, 0.0], a_max=params.L)
            gmm_params.append((float(w), m, np.array(C, dtype=float)))

        ks = ErgodicCoveragePreprocessor.build_modes(params.num_k_per_dim)
        gx, gy, grid, dxdy = ErgodicCoveragePreprocessor.build_grid(
            params.num_cell, params.L)
        hk, Fk_grid = ErgodicCoveragePreprocessor.build_hk_basis(
            ks, grid, dxdy, params.L)
        phi, rho_flat = ErgodicCoveragePreprocessor.project_target_to_phi(
            Fk_grid, grid, gmm_params, dxdy)
        pdf_img = rho_flat.reshape(gx.shape)
        Lam = np.array([ErgodicCoveragePreprocessor.sobolev_weight(int(k[0]), int(k[1]))
                        for k in ks], dtype=float)
        return BasisPack(phi=phi, Lam=Lam, ks=ks, Fk_grid=Fk_grid, dxdy=dxdy, hk=hk, pdf_img=pdf_img, gx_shape=gx.shape)
