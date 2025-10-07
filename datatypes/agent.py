#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Any, Dict, TypeVar, Generic
from enum import Enum
import numpy as np

from model import Model

# Generic type variable for allocation to support different allocation types
T = TypeVar('T')

class AgentType(Enum):
    DRONE = 1
    DIFF_DRIVE = 2
    CAR = 3

@dataclass
class Agent(Generic[T]):
    """
    Generic dataclass to represent an agent's state in task allocation algorithms.

    Attributes:
        id (int): Unique identifier for the agent, used to index and reference the agent.
        action (T): The choice of agent
        allocation (T): The task or resource assigned to the agent (e.g., task ID, -1 for no task, or other types).
        metadata (Dict[str, Any]): Dictionary for algorithm-specific attributes (e.g., iteration, timestamp, utility).
    """
    id: int
    action: T
    capabilities: T
    allocation: T
    multipliers: T
    metadata: Dict[str, Any] = None
    goal_lambda: float = 0.0
    capability: float = 1.0

    # State, which may be mission state or general state
    state: np.ndarray = None
    # Control input, which should also include the slack variable for mission
    control: np.ndarray = None
    # Dynamic model
    model: Model = None # Dynamic model of the agent
    # Agent type
    type: AgentType = AgentType.DRONE # Assuming full traversability
    # Specialization vector
    spec_v: np.ndarray = None
    # Suitability

    mission_status: float = 0.0

    # Generic attributes for agent capabilities and constraints
    sensing_range: float = None
    u_bounds: np.ndarray = None
    min_energy: float = None

    def __post_init__(self):
        """Initialize metadata as an empty dictionary if not provided."""
        if self.metadata is None:
            self.metadata = {}