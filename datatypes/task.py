#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple
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
class ErgodicCoverageTask:
    """Template-compatible task describing the ergodic MPC spec."""
    # ---- Task base fields (keep for compatibility with your Task ABC) ----
    reward: float = 0.0
    capabilities_required: NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))
    task_type: TaskType = TaskType.EXPLORE
    is_completed: bool = False
    progress: float = 0.0

    # ---- Ergodic-specific fields ----
    dt: float = 0.1                       # MPC step (s)
    c_init: NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))  # (nK,)
    t_init: float = 1e-2                  # initial averaging time t
    c_clf: float = 0.2                    # CLF rate
    A: int = 1                            # number of robots participating
    E_threshold: float = 1e-3             # completion threshold on E(c)
    centers: NDArray[np.float64] = None
    covs: NDArray[np.float64] = None

    def get_complete_status(self, E_current: float) -> bool:
        """Mark complete when E(c) is below a threshold."""
        self.is_completed = (E_current <= self.E_threshold)
        return self.is_completed
    


