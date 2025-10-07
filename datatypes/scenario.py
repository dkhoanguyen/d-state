#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

from enum import Enum
from datatypes.task import Task

class TaskType(Enum):
    REACH_GOAL = 1
    EXPLORE = 2
    
@dataclass
class Scenario:
    """Dataclass to represent a scenario with agents and tasks."""
    num_agents: int
    num_tasks: int
    agent_comm_matrix: NDArray[np.int32]
    tasks: list[Task] 
    agent_locations: NDArray[np.float64]
    obstacles: NDArray[np.float64]
