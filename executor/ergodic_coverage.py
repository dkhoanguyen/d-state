#!/usr/bin/env python3
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Optional
from scipy.optimize import minimize

from executor.executor import Executor
from datatypes import *
from evaluator import *               # optional, same as your example imports
from model import Model
from objective import Objective, FourierBasis, ErgodicCoverageObjective

class ErgodicCoverageExecutor(Executor):
    """
    MPC executor that optimizes controls over a horizon for ONE agent
    (decentralized), using a CLF row from ErgodicCoverageObjective and rolling
    the running average c_k forward with predicted F_avg.

    Task fields expected:
      task.dt: float
      task.c_init: np.ndarray (nK,)
      task.t_init: float
      task.c_clf: float
      task.A: int (number of agents; default 1)
      task.F_avg_provider: Optional[Callable[[int, np.ndarray], np.ndarray]]
          Returns predicted F_avg at step k given predicted state x_k.
          If None, we use single-agent approx: F_avg = F(x_k) / A (or F(x_k) if A==1).
    """

    def __init__(self,
                 w_eps: float = 1e-6, k_eps: float = 1e3):
        self.basis = basis
        self.objective = objective
        self.w_eps = float(w_eps)  # penalty on CLF row (squared)
        self.k_eps = float(k_eps)  # penalty on slack

    def execute(self,
                agent: Agent,
                other_agents: List[Agent],
                scenario: Scenario,
                task: ErgodicCoverageTask,
                lambda_k: NDArray[np.float64],
                objective: ErgodicCoverageObjective,   # kept for signature parity; not used (we use self.objective)
                horizon: int,
                maxiter: int = 30,
                ftol: float = 1e-3,
                eps: float = 1e-4) -> Tuple[NDArray[np.float64], bool]:
        """
        Returns:
            U*: np.ndarray of shape (3*horizon,) stacking [ux,uy,delta] per step
            success: bool from SciPy
        """
        # Initial guess: repeat current control + zero slack
        U0 = np.tile(np.hstack([agent.control, 0.0]), horizon)

        u_max = agent.u_bounds[0, 1]
        u_min = agent.u_bounds[0, 0]
        # Bounds per step: [ux, uy, delta]
        bounds = [(u_min, u_max), (u_min, u_max), (0.0, np.inf)] * horizon

        res = minimize(
            lambda U: self._compute_cost(
                model=agent.model,
                U=U,
                x0=agent.state.copy(),
                task=task,
                lambda_k=lambda_k,
                horizon=horizon
            ),
            U0, method='SLSQP', bounds=bounds, constraints=[],
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False, 'eps': eps}
        )
        return res.x, res.success

    # ---------- rollout and cost ----------
    def _compute_cost(self,
                      model: Model,
                      U: NDArray[np.float64],
                      x0: NDArray[np.float64],
                      task: ErgodicCoverageTask,
                      lambda_k: NDArray[np.float64],
                      objective: ErgodicCoverageObjective,
                      horizon: int,
                      dt: float,
                      num_agents: int) -> float:
        """
        Cost = sum_k [ 0.5 ||u_k||^2  +  λ_k * g_k + (w_eps/2)*g_k^2 + k_eps*delta_k^2 ],
        where g_k is the CLF row value at step k for THIS agent.
        The running-average coefficients c_hat are rolled forward using predicted F_avg.
        """
        dt = dt
        A = num_agents
        F_avg_provider: Optional[Callable[[int, np.ndarray], np.ndarray]] = getattr(task, "F_avg_provider", None)
        c_clf = float(task.c_clf)

        # Predicted c, t along the horizon (copied from task initial values)
        c_hat = task.c_init.copy()   # (nK,)
        t_hat = float(task.t_init)

        x = x0.copy()
        cost = 0.0

        for k in range(horizon):
            uk = U[3*k:3*k+2]
            delta_k = U[3*k+2]

            # control effort + slack penalty
            cost += 0.5 * np.dot(uk, uk) + self.k_eps * (delta_k**2)

            # Advance state with model
            x = model.f(x, uk, dt)

            # Predict Fourier average for c update
            if F_avg_provider is not None:
                F_avg_k = F_avg_provider(k, x)
            else:
                # single-agent approximation (if A>1, scale to average)
                F_local = self.basis.F_vec(x[:2])
                F_avg_k = F_local / max(A, 1)

            # Running-average update for c (discrete Euler)
            #   c <- c + dt * ((F_avg - c) / t)
            # and t <- t + dt
            c_hat = c_hat + dt * ((F_avg_k - c_hat) / t_hat)
            t_hat = t_hat + dt

            # CLF row at step k:
            # g_k = (-c_clf*E + delta_k) - gradE(x_k)^T u_k
            g_k = objective.evaluate(
                model=model,
                x=x,
                u=np.array([uk[0], uk[1], delta_k], dtype=float),
                c=c_hat,
                t_now=t_hat,
                c_clf=c_clf,
                A=A,
                delta=delta_k
            )
            cost += lambda_k[k] * g_k + 0.5 * self.w_eps * (g_k**2)

        return float(cost)