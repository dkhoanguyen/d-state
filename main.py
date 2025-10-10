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
    # Load settings
    # Parameters
    num_agents = 20
    num_tasks = 2
    full_comms = True
    # Run mission
    planning_budget = 10
    max_iterations = 350

    comm_distance = 5
    agent_location_range = [5, 5]

    # Load scenarios

    # Specify tasks
    tasks = []
    reach_goal_locations = np.array([
        [-2.5, 30.0],
        [25.0, -5.0],
        [37.5, -12.5],
        [-32.5, -20.0],
        [-4.5, -32.5],
    ])

    # reach_goal_locations = np.array([
    #     [-25.5, 0.0],
    #     [25.0, 0.0],
    # ])

    for reach_goal_location in reach_goal_locations:
        task = ReachGoalTask()
        task.location = reach_goal_location
        task.radius = 1.0
        task.capabilities_required = num_agents/num_tasks
        task.reward = 1000.0

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

            # Broadcast plans as soon as planning is done
            if np.random.binomial(1, 1.0):
                # Communicate only to local cluster
                neighbour_ids = np.where(
                    scenario.agent_comm_matrix[agent_id] > 0)[0]

                if full_comms:
                    neighbour_ids = np.arange(num_agents)
                for neighbor_id in neighbour_ids:
                    local_agents[neighbor_id][agent_id] = deepcopy(agent)

            # Apply control
            control = agents[agent_id].control[:2]

            # Control breaks down
            agents[agent_id].state = agents[agent_id].model.f(
                agents[agent_id].state, control)

            # print(agents[agent_id].spec_v)

        planning_time /= num_agents
        print(f"Averaged planning time: {planning_time}")
        # Collect allocation
        record_tasks = np.zeros((num_tasks))
        record_capabilities = np.zeros((num_tasks))
        for agent in agents:
            action_indices = np.argsort(-agent.action)
            record_tasks[action_indices[0]] += 1
            record_capabilities[action_indices[0]
                                ] += agent.capabilities[action_indices[0]]
        print(f"Record tasks: {record_capabilities}")

        optimal_coalition_size = np.round(
            scenario.num_agents/scenario.num_tasks)
        for task_id in range(num_tasks):
            utility = peak_evaluator.calculate_utility(
                reward=scenario.tasks[task_id].reward,
                cost=0.0,
                optimal_coalition_size=optimal_coalition_size,
                joint_actions=np.ones(1) * record_tasks[task_id])
            current_utils += utility * record_tasks[task_id]

        # Broadcast multipliers
        for task_id in range(num_tasks):
            multiplier = 0.0
            for local_agent in agents:
                multiplier += local_agent.multipliers[task_id]
            multiplier = multiplier / num_agents

            for local_agent in agents:
                local_agent.multipliers[task_id] = multiplier

        total_utils = np.hstack((total_utils, np.array(current_utils)))

        if consensus_step > 1:
            if np.abs(total_utils[consensus_step] - total_utils[consensus_step-1]) <= 1.0:
                break_condition += 1
            else:
                break_condition = 0

        consensus_step += 1

    # Record final states
    state_history.append(np.array([agent.state for agent in agents]))
    control_history.append(np.array([agent.control for agent in agents]))
    final_allocation = np.zeros(num_agents, dtype=np.int32)
    for i, agent in enumerate(agents):
        action_indices = np.argsort(-agent.action)
        final_allocation[i] = action_indices[0]
    allocation_history.append(final_allocation)

    record_tasks = np.zeros((num_tasks))
    for agent in agents:
        action_indices = np.argsort(-agent.action)
        record_tasks[action_indices[0]] += 1

    final_allocation = np.zeros((num_agents), dtype=np.int32)
    for agent in agents:
        action_indices = np.argsort(-agent.action)
        final_allocation[agent.id] = action_indices[0]

    print(record_tasks)

    result = {
        'final_utilities': total_utils.tolist(),
        'final_allocation': final_allocation.tolist(),
        'consensus_step': consensus_step,
        'history': {
            'allocation': allocation_history,
            'satisfied_agents_count': satisfied_agents_count_history,
            'iteration': iteration_history,
            'total_utility': total_utils.tolist(),
            'states': state_history,  # Add state history to results
            'control': control_history,
            'mission': mission_status_history
        },
    }

    animate_agent_trajectories(agents=agents, result=result, scenario=scenario,
                               save_format='gif', save_path=f'p_grape_exec.gif')

if __name__ == "__main__":
    main()
