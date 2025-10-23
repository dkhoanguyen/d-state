#!/usr/bin/env python3
import time
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Tuple
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye

from executor.executor import Executor
from datatypes import *
from evaluator import *
from model import Model
from objective import Objective, ReachGoalObjective


class ReachGoalExecutor(Executor):

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
        # Simple reward calculation as it does not explicitly require coordination
        probs = self._poisson_binomial_distribution(joint_actions)
        expected_utility = 0
        for k in range(1, len(probs)):
            utility = (task.reward/k) * \
                np.exp(-k / task.capabilities_required + 1)
            expected_utility += probs[k]*utility
        return expected_utility
    
    def calculate_cost(
        self,
        agent: Agent,
        task: ReachGoalTask,
        other_agents: list[Agent],
        objective_controller: Objective,
        horizon: float,
        joint_actions: NDArray[np.float64]
    ) -> float:
        cost = 0.0
        x0 = agent.state

        # TODO: Change this to use the full control sequence instead of 
        # repeating the same control for the entire horizon
        U = np.tile(agent.control, horizon)

        trajectory = self._simulate_trajectory(
                model=agent.model,
                x0=x0,
                U=U,
                N=horizon,
                dt=0.1
            )
        
        # Reach goal augmented lagrangian
        for k in range(horizon):
            x_k = trajectory[k]
            x_g = task.location
            task_cost = objective_controller.evaluate(
                model=agent.model,
                x=x_k,
                u=agent.control,
                x_g=x_g,
                dx_g_dt=task.velocity,
                gamma=1.0,  # Example gamma value
                r=0.1
            )
            w_eps = 0.000001
            # task_cost = np.maximum(task_cost, 0.0)
            # Form augmented lagrangian
            cost += agent.goal_lambda * task_cost + \
                (w_eps / 2) * task_cost**2

        expected_cost = cost 
        return expected_cost

    def execute(self,
                agent: Agent,
                other_agents: List[Agent],
                scenario: Scenario,
                task: ReachGoalTask,
                lambda_k: np.ndarray,
                objective: Objective,
                horizon: int,
                w_esp: float= 0.000001,
                maxiter: int = 30,
                ftol: float = 0.001,
                eps: float = 1e-4) -> Tuple[NDArray[np.float64], bool]:
        """Execute the reach goal task using MPC"""
        U0 = np.tile(agent.control, horizon)
        u_max = agent.u_bounds[0, 1]
        u_min = agent.u_bounds[0, 0]
        bounds = [(u_min, u_max)] * (2 * horizon) + \
            [(-np.inf, np.inf)] * horizon

        constraints = []

        res = minimize(
            lambda U: self._compute_cost(
                model=agent.model,
                U=U,
                x_0=agent.state,
                x_g=task.location,
                v_g=task.velocity,
                radius=task.radius,
                lambda_k=lambda_k,
                objective=objective,
                horizon=horizon,
                dt=0.1,
                w_eps=w_esp),
            U0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={
                'maxiter': maxiter,
                'ftol': ftol,
                'disp': False,
                'eps': eps
            }
        )
        return res.x, res.success

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

    def _compute_cost(self,
                      model: Model,
                      U: np.ndarray,
                      x_0: np.ndarray,
                      x_g: np.ndarray,
                      v_g: np.ndarray,
                      radius: float, 
                      lambda_k: np.ndarray,
                      objective: Objective,
                      horizon: int,
                      dt: float,
                      w_eps: float = 0.000001,
                      k_eps: float = 1000.0) -> float:
        """Compute the MPC cost including control effort and task-specific costs."""
        cost = 0.0
        trajectory = self._simulate_trajectory(model, x_0, U, horizon, dt)

        for k in range(horizon):
            u = U[3*k:3*k+2]  # Extract [u_x_k, u_y_k]
            # Control effort
            cost += 0.5 * np.linalg.norm(u)**2

            # Task objective
            x_k = trajectory[k+1]
            u_with_slack = U[3*k:3*k+3]  # [u_x_k, u_y_k, eps_k]
            task_cost = objective.evaluate(
                model=model,
                x=x_k,
                u=u_with_slack,
                x_g=x_g,
                dx_g_dt=v_g,
                gamma=1.0,
                r=radius
            )
            # Add Lagrangian and penalty terms
            cost += lambda_k[k] * task_cost + (w_eps / 2) * task_cost**2

        # Slack penalties
        for k in range(horizon):
            eps = U[3*k+2]  # Slack variable for step k
            cost += k_eps * eps**2

        return cost
    
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