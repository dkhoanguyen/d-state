#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List

from datatypes import Scenario, Agent
from evaluator import Evaluator

class Allocator(ABC):
    """
    Abstract base class for task allocators.
    """

    @abstractmethod
    def plan(self,
             agent: Agent,
             agents: List[Agent],
             scenario: Scenario,
             evaluator: Evaluator,
             planning_budget: int
             ) -> List[Agent]:
        """
        Abstract method to plan the allocation of tasks.
        Returns:
            np.ndarray: An array representing the planned allocation of tasks.
        """
        pass

    @abstractmethod
    def communicate(self):
        """
        """

    @abstractmethod
    def allocate(self, scenario: Scenario, evaluator: Evaluator) -> np.ndarray:
        """
        Abstract method to allocate tasks.
        Returns:
            np.ndarray: An array representing the allocation of tasks.
        """
        pass