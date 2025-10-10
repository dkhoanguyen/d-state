#!/usr/bin/env python3
import time
from copy import deepcopy
import os
import pickle

from utils import ScenarioGenerator
from evaluator import *
from utils import *
from datatypes import *
from objective import *

from allocator import DSTATE


def main():
    num_agents = 2
    num_tasks = 1
    full_comms = True
    # Run mission
    planning_budget = 10
    max_iterations = 350

    comm_distance = 5
    agent_location_range = [5, 5]

    tasks = []
    centers = [[0.25, 0.70],
               [0.75, 0.30]]

    covs = [[[0.02, 0.0], [0.0, 0.02]],
            [[0.02, 0.0], [0.0, 0.02]]]
    
    
    task = ErgodicCoverageTask()
    task.capabilities_required = 0
    task.t_init = 1e-2
    task.c_clf = 0.2
    task.A = 1
    task.E_threshold = 1e-3
    task.centers = centers
    task.covs = covs
    tasks.append(task)

    # Create an instance of ScenarioGenerator
    scenario_generator = ScenarioGenerator()

    scenario = scenario_generator.generate_scenario(
        num_agents=num_agents,
        num_tasks=num_tasks,
        comm_distance=comm_distance,
        gap_agent=15,
        agent_location_range=agent_location_range,
        tasks=tasks,
        obstacles=None
    )
    actions = generate_initial_action(scenario=scenario, deterministic=False)
    
    # Initialise agents
    agents = ScenarioGenerator.generate_agents(
        num_tasks=num_tasks,
        num_agents=num_agents,
        initial_actions=actions,
        num_constraints=num_tasks,
        scenario=scenario,
    )

    allocator = DSTATE(
        tasks=scenario.tasks,
        obstacles=scenario.obstacles)
    # Evaluators
    prob_evaluator = IFTStochasticEvaluator()
    peak_evaluator = PeakedRewardDeterministicEvaluator()

    consensus_step = 0
    allocation_history = []
    satisfied_agents_count_history = []
    iteration_history = []
    total_utility_history = []
    state_history = []  # Store agent states over time
    control_history = []
    mission_status_history = []

    total_utils = np.empty(0)
    break_condition = 0

    local_agents = [deepcopy(agents) for _ in range(num_agents)]

    print("Init done")

    while consensus_step < max_iterations:
        # Record current states
        current_states = np.array([agent.state for agent in agents])
        state_history.append(current_states)

        # Allocation
        current_allocation = np.zeros(num_agents, dtype=np.int32)
        for i, agent in enumerate(agents):
            action_indices = np.argsort(-agent.action)
            current_allocation[i] = action_indices[0]
        allocation_history.append(current_allocation)

        # Control
        current_controls = np.array([agent.control for agent in agents])
        control_history.append(current_controls)

        # Mission
        current_status = np.array([agent.mission_status for agent in agents])
        mission_status_history.append(current_status)
        # print(current_status)

        # Plans
        current_utils = 0.0
        planning_time = 0.0
        for agent_id, agent in enumerate(agents):
            start_time = time.time()
            agents[agent_id] = allocator.plan(
                agent, local_agents[agent_id], scenario, prob_evaluator, planning_budget)
            local_agents[agent_id][agent_id] = agents[agent_id]

            planning_time += time.time() - start_time

        consensus_step += 1


if __name__ == "__main__":
    main()