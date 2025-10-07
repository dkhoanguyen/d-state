#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from enum import Enum
from datatypes.agent import Agent

class TaskType(Enum):
    REACH_GOAL = 1
    EXPLORE = 2
    HERDING = 3

@dataclass
class Task(ABC):
    reward: float
    capabilities_required: NDArray[np.float64]
    task_type: TaskType
    is_completed: bool
    progress: float

    @abstractmethod
    def get_complete_status(self, *args, **kwargs) -> bool:
        pass

@dataclass
class ReachGoalTask(Task):
    location: NDArray[np.float64]
    velocity: NDArray[np.float64]
    radius: float
    
    def __init__(self):
        self.task_type: TaskType = TaskType.REACH_GOAL
        self.velocity = np.zeros(2)

    def get_complete_status(self, agent_locations: NDArray[np.float64]):
        distances = np.linalg.norm(agent_locations - self.location, axis=1)

        # Cummulative task so all agents must be within the radius
        self.is_completed = np.all(distances <= self.radius)
        return self.is_completed

    
@dataclass
class ExploreTask(Task):
    pass


