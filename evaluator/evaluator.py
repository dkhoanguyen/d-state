#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np

from datatypes import *


class Evaluator(ABC):
    @abstractmethod
    def calculate_utility(
            self,
            reward: float,
            optimal_coalition_size: int,
            joint_actions: np.ndarray,
            cost: float = 0.0                # For backward compatibility
    ) -> float:
        pass

    @abstractmethod
    def calculate_reward(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        pass

    @abstractmethod
    def calculate_min_number_constraints(self,
                                         joint_actions: np.ndarray,
                                         min_num: int):
        """
        """

    @abstractmethod
    def preprocess(self,
                   joint_action: NDArray[np.float64]):
        """
        """
