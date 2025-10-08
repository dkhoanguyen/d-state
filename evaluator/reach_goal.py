#!/usr/bin/env python3
import numpy as np
from evaluator.evaluator import Evaluator
from datatypes import *
from objective import Objective

class ReachGoalEvaluator(Evaluator):
    def __init__(self):
        """
        Initialize the ReachGoalEvaluator.
        This evaluator calculates utility based on whether the goal is reached.
        """
        super().__init__()

    def calculate_utility(
            self,
            reward: float,
            cost: float,
            optimal_coalition_size: int,
            joint_actions: np.ndarray) -> float:
        """
        Calculate utility based on reaching the goal.

        Args:
            reward (float): The reward received.
            cost (float): The cost incurred.
            joint_actions (np.ndarray): Joint actions here is the choice of the agents for the task,
            excluding the current agent

        Returns:
            float: The calculated utility.
        """
        current_members = np.sum(joint_actions)
        if current_members >= optimal_coalition_size:
            utility = reward - cost
        else:
            utility = -cost
        # Ensure utility is non-negative
        return max(utility, 0.0)

    def calculate_min_number_constraints(self,
                                         joint_actions: np.ndarray,
                                         min_num: int):
        return
    
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
    

    def _simulate_trajectory(self,
                             model: Model,
                             x0: np.ndarray,
                             U: np.ndarray,
                             N: int,
                             dt: float = 0.1):
        trajectory = [x0.copy()]
        x = x0.copy()
        for k in range(N):
            u = U[3*k:3*k+2]
            x = model.f(x, u, dt)
            trajectory.append(x.copy())
        return trajectory
    
    def preprocess(self,
                   joint_action: NDArray[np.float64]):
        prob = self._poisson_binomial_distribution(joint_action)
        return np.sum(prob)