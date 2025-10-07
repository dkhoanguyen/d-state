#!/usr/bin/env python3
import numpy as np
from evaluator.evaluator import Evaluator
from datatypes import *


class PeakedRewardDeterministicEvaluator(Evaluator):
    def __init__(self):
        """
        Initialize the PeakedRewardDeterministicEvaluator.
        This evaluator calculates utility based on a peaked reward model.
        """
        super().__init__()

    def calculate_utility(
            self,
            reward: float,
            cost: float,
            optimal_coalition_size: int,
            joint_actions: np.ndarray) -> float:
        """
        Calculate utility based on a deterministic model.

        Args:
            reward (float): The reward received.
            cost (float): The cost incurred.
            joint_actions (np.ndarray): Joint actions here is the choice of the agents for the task,
            excluding the current agent

        Returns:
            float: The calculated utility.
        """
        current_members = np.sum(
            joint_actions)
        utility = (reward/optimal_coalition_size) * \
            np.exp(-current_members / optimal_coalition_size + 1) - cost
        # Ensure utility is non-negative
        return max(utility, 0.0)

    def calculate_min_number_constraints(self,
                                         joint_actions: np.ndarray,
                                         min_num: int):
        return
    
    def calculate_reward(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        return 0.0

    def calculate_cost(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        return 0.0


class SubmodularRewardDeterministicEvaluator(Evaluator):
    def __init__(self,
                 num_agents: int = 0,
                 num_tasks: int = 0):
        """
        Initialize the PeakedRewardDeterministicEvaluator.
        This evaluator calculates utility based on a peaked reward model.
        """
        super().__init__()

        self._num_agents = num_agents
        self._num_tasks = num_tasks

    def calculate_utility(
            self,
            reward: float,
            cost: float,
            optimal_coalition_size: int,
            joint_actions: np.ndarray) -> float:
        """
        Calculate utility based on a submodular model.

        Args:
            reward (float): The reward received.
            cost (float): The cost incurred.
            joint_actions (np.ndarray): Joint actions here is the choice of the agents for the task,
            excluding the current agent

        Returns:
            float: The calculated utility.
        """
        current_members = np.sum(
            joint_actions)
        utility = (reward / (np.log2(self._num_agents/self._num_tasks + 1))) * \
            (np.log2(current_members+1)/current_members) - cost

        # Ensure utility is non-negative
        return max(utility, 0.0)

    def calculate_reward(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        return 0.0

    def calculate_cost(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        return 0.0
