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
        self.w_eps = float(w_eps)  # penalty on CLF row (squared)
        self.k_eps = float(k_eps)  # penalty on slack

    def execute(self,
            agent: Agent,
            other_agents: List[Agent],
            scenario: Scenario,
            task: ErgodicCoverageTask,
            lambda_k: NDArray[np.float64],
            objective: ErgodicCoverageObjective,   # kept for signature parity
            horizon: int,
            maxiter: int = 30,
            ftol: float = 1e-3,
            eps: float = 1e-4) -> Tuple[NDArray[np.float64], bool]:
        """
        Returns:
            U*: (n_dec*horizon,) stacking [u_components..., delta] per step
            success: bool
        """
        # --- decide per-step decision dimension ---
        # controls per step (from agent.control), +1 slack delta
        m_u = int(np.size(agent.control))        # e.g., 2 for [ux,uy]; could be 1 if scalar
        n_dec_step = m_u + 1                     # add slack delta

        # --- initial guess: repeat current control and zero slack ---
        z_step0 = np.hstack([np.atleast_1d(agent.control), 0.0])   # shape (n_dec_step,)
        U0 = np.tile(z_step0, horizon)                              # shape (n_dec_step*horizon,)

        # --- bounds per step ---
        # prefer agent.u_bounds if available; otherwise fall back to [-1,1]
        if hasattr(agent, "u_bounds") and agent.u_bounds is not None:
            u_min = float(agent.u_bounds[0, 0])
            u_max = float(agent.u_bounds[0, 1])
        else:
            u_min, u_max = -1.0, 1.0

        step_bounds = [(u_min, u_max)] * m_u + [(0.0, np.inf)]      # delta >= 0
        bounds = step_bounds * horizon

        # --- sanity checks before calling SciPy ---
        assert U0.ndim == 1
        assert len(bounds) == U0.size, f"bounds ({len(bounds)}) != len(U0) ({U0.size})"
        assert len(lambda_k) == horizon, "lambda_k must have length == horizon"

        # --- call the optimizer ---
        res = minimize(
            lambda U: self._compute_cost(
                model=agent.model,
                U=U,
                x0=agent.state.copy(),
                task=task,
                lambda_k=lambda_k,
                horizon=horizon,
                objective=objective,
                dt=task.dt,
                num_agents=task.A
            ),
            U0, method='SLSQP', bounds=bounds, constraints=[],
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False, 'eps': eps}
        )
        return res.x, bool(res.success)

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

            # single-agent approximation (if A>1, scale to average)
            F_local = objective.basis.F_vec(x[:2])
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