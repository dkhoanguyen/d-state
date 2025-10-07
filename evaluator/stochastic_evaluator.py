#!/usr/bin/env python3
import numpy as np
from evaluator.evaluator import Evaluator
from datatypes import *


class MontoCarloStochasticEvaluator(Evaluator):
    def __init__(self, num_simulations: int = 1000):
        """
        Initialize the MontoCarloStochasticEvaluator.
        This evaluator calculates utility based on a stochastic model using Monte Carlo simulation.

        Args:
            num_simulations (int): Number of simulations to run for utility calculation.
        """
        super().__init__()
        self._num_simulations = num_simulations

    def calculate_utility(
            self,
            reward: float,
            cost: float,
            optimal_coalition_size: int,
            joint_actions: np.ndarray) -> float:
        """
        Calculate utility based on a stochastic model using Monte Carlo simulation.

        Args:
            reward (float): The reward received.
            cost (float): The cost incurred.
            joint_actions (np.ndarray): Joint actions here is the choice of the agents for the task,
            excluding the current agent

        Returns:
            float: The calculated utility.
        """
        utilities = []
        current_members = np.sum(
            joint_actions)

        # Tune diminishing returns
        deterministic_utility = (reward/optimal_coalition_size) * \
            np.exp(-current_members / optimal_coalition_size + 1) - cost

        for _ in range(self._num_simulations):
            joins = np.random.random(len(joint_actions)) < joint_actions
            coalition_size = np.sum(joins)

            if coalition_size > 0:
                utilities.append(deterministic_utility)
            else:
                utilities.append(0.0)
        # Return the expected utility (mean of simulated utilities)
        expected_utility = np.mean(utilities)
        return max(expected_utility, 0)


class IFTStochasticEvaluator(Evaluator):
    def __init__(self):
        """
        Initialize the IFTStochasticEvaluator.
        This evaluator calculates utility based on a stochastic model using Monte Carlo simulation.
        """
        super().__init__()

    def calculate_utility(
            self,
            reward: float,
            cost: float,
            optimal_coalition_size: int,
            joint_actions: np.ndarray) -> float:
        """
        Calculate utility based on a stochastic model using Monte Carlo simulation.

        Args:
            reward (float): The reward received.
            cost (float): The cost incurred.
            joint_actions (np.ndarray): Joint actions here is the choice of the agents for the task,
            excluding the current agent

        Returns:
            float: The calculated utility.
        """

        # Calculate the probabilities for each combination of agents
        probs = self._poisson_binomial_distribution(joint_actions)
        # optimal_coalition_size = 1

        expected_utility = 0
        for k in range(1, len(probs)):
            # Default utility
            utility = (reward/k) * \
                np.exp(-k / optimal_coalition_size + 1) - cost
            expected_utility += probs[k]*utility
        return expected_utility
    
    def calculate_reward(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        probs = self._poisson_binomial_distribution(joint_actions)
        expected_utility = 0
        for k in range(1, len(probs)):
            # Default utility
            utility = (task.reward/k) * \
                np.exp(-k / task.capabilities_required + 1)
            expected_utility += probs[k]*utility
        return 0.0

    def calculate_cost(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        probs = self._poisson_binomial_distribution(joint_actions)
        return 0.0

    def calculate_min_number_constraints(self,
                                         joint_actions: np.ndarray,
                                         min_num: int):
        # Calculate the probabilities for each combination of agents
        probs = self._poisson_binomial_distribution(joint_actions)
        expected_number_of_agents = 0
        for k in range(min_num, len(probs)):
            expected_number_of_agents += probs[k] * k
        return expected_number_of_agents

    def get_prob(self, joint_actions: np.ndarray):
        return self._poisson_binomial_distribution(joint_actions)

    def _poisson_binomial_distribution(self, probabilities):
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

    def _poisson_binomial_distribution_at_k(self, probabilities, k):
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
        for i in range(N):
            phi_X_i = (1 - p[i]) + p[i] * np.exp(1j * t)
            phi_X *= phi_X_i

        # Compute inverse FFT and scale by 1/(N+1), then shift to correct order
        probs = np.fft.ifft(phi_X) / (N + 1)
        probs = np.fft.fftshift(probs)  # Shift to align k = 0 to N
        probs = np.real(probs)
        # probs = np.flip(probs)  # Flip to ensure k = 0 is first

        # Normalize to ensure sum = 1
        probs_sum = np.sum(probs)
        if abs(probs_sum - 1) > 1e-10:
            probs = probs / probs_sum

        probs = np.maximum(probs, 0)

        # Reorder to ensure k = 0 is first (after shift, take the middle section)
        probs = np.roll(probs, N // 2)
        probs = np.flip(probs)

        return probs[k]
