#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from datatypes import *


class Executor(ABC):
    @abstractmethod
    def init(self, *arg, **kwargs):
        pass

    @abstractmethod
    def execute(self, *arg, **kwargs):
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
    def calculate_cost(
        self,
        agent: Agent,
        other_agents: list[Agent],
        scenario: Scenario,
        task: Task,
        joint_actions: NDArray[np.float64]
    ) -> float:
        pass

    @abstractmethod
    def preprocess(self,
                   joint_action: NDArray[np.float64]):
        """
        """
