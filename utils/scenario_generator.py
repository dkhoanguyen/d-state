#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from datatypes import *
from model import SingleIntegrator, DoubleIntegrator, DifferentialDrive
from objective import Objective, ReachGoalObjective


class ScenarioGenerator:
    """A class with static methods to generate and verify scenarios with agents and tasks."""
    @staticmethod
    def generate_agent_location(
        num_agents: int,
        limit: NDArray[np.float64],
        gap: float,
        deployment_type: str = 'circle',
        existing_locations: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """
        Generates locations for agents ensuring they respect specified constraints and deployment type.

        Parameters:
        - num_agents (int): Number of agents to generate.
        - limit (NDArray[np.float64]): Array with the maximum range for location generation.
        - gap (float): Minimum distance between agents.
        - deployment_type (str): Deployment pattern for agents ('circle', 'square', or 'skewed_circle').
        - existing_locations (NDArray[np.float64] | None): Already generated locations (if any).

        Returns:
        - NDArray[np.float64]: Generated locations for the agents.
        """
        locations = np.zeros((num_agents, 2))
        for i in range(num_agents):
            ok = False
            while not ok:
                candidate = np.random.uniform(-limit, limit, 2)
                if i < num_agents/2:
                    candidate = np.random.uniform(-limit, 0.1*limit, 2)

                # if i > 0 and np.any(np.linalg.norm(locations[:i] - candidate, axis=1) < gap):
                #     continue

                if np.linalg.norm(candidate) >= np.min(limit):
                    continue

                locations[i] = candidate
                ok = True
        # locations = np.array(
        #     [[00.0, 0.0]]
        # )
        return locations

    @staticmethod
    def generate_task_location(
        num_tasks: int,
        limit: NDArray[np.float64],
        gap: float,
        existing_locations: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """
        Generates locations for tasks ensuring they respect specified constraints.

        Parameters:
        - num_tasks (int): Number of tasks to generate.
        - limit (NDArray[np.float64]): Array with the maximum range for location generation.
        - gap (float): Minimum distance between tasks.
        - existing_locations (NDArray[np.float64] | None): Already generated locations (if any).

        Returns:
        - NDArray[np.float64]: Generated locations for the tasks.
        """
        locations = np.zeros((num_tasks, 2))
        for i in range(num_tasks):
            ok = False
            while not ok:
                candidate = np.random.uniform(-limit, limit, 2)

                # print(np.linalg.norm(locations[:i] - candidate, axis=1))

                if i > 0 and np.any(np.linalg.norm(locations[:i] - candidate, axis=1) < gap):
                    continue

                if np.linalg.norm(limit * 0.7) > np.linalg.norm(candidate):
                    continue

                locations[i] = candidate
                ok = True
        return locations

    @staticmethod
    def generate_scenario(
        num_agents: int,
        num_tasks: int,
        comm_distance: float = 50,
        gap_agent: float = 15,
        agent_location_range: list[float] = [600, 600],
        tasks: list[Task] = [],
        obstacles: NDArray[np.float64] | None = None,
    ):
        agent_locations = ScenarioGenerator.generate_agent_location(
            num_agents, np.array(agent_location_range), gap=gap_agent,
            deployment_type='circle')
        
        # Communication matrix generation
        dist_agents = np.linalg.norm(
            agent_locations[:, np.newaxis, :] - agent_locations[np.newaxis, :, :], axis=-1)
        agent_comm_matrix = (dist_agents <= comm_distance).astype(
            int) - np.eye(num_agents, dtype=int)
        
        return Scenario(
            num_agents=num_agents,
            num_tasks=num_tasks,
            agent_comm_matrix=agent_comm_matrix,
            tasks=tasks,
            agent_locations=agent_locations,
            obstacles=obstacles
        )

    @staticmethod
    def generate_agents(
            num_tasks: float,
            num_agents: float,
            initial_actions: np.ndarray,
            num_constraints: float,
            scenario: Scenario):
        agents = []

        initial_allocation = np.full(num_agents, -1, dtype=int)

        for i in range(num_agents):
            # Dynamic model
            model = SingleIntegrator()
            x0, u0 = model.init()
            x0[:] = scenario.agent_locations[i, :]

            # Add a slack variable to control input for mission execution
            u0 = np.hstack((u0, 0.0))
            v_max = 0.1
            agent_type = AgentType.DIFF_DRIVE

            agent = Agent(
                id=i,
                allocation=initial_allocation.copy(),
                capabilities=np.random.randint(1, 2, size=(num_tasks,)),
                action=initial_actions[i, :],
                multipliers=np.zeros(num_constraints),
                # State
                state=x0,
                control=u0,
                model=model,
                goal_lambda=0.0,
                # Capabilities
                sensing_range=100.0,
                type=agent_type,
                spec_v=np.ones(scenario.num_tasks),

                u_bounds=np.array(
                    [[-v_max, v_max], [-v_max, v_max], [-np.inf, np.inf]]),
                min_energy=0.1,
                metadata={
                    'iteration': 0,
                    'time_stamp': np.random.rand(),
                    'satisfied_flag': False,
                    'util': 0.0
                },
            )
            agents.append(agent)
        return agents
