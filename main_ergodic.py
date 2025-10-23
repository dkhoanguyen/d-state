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
    num_agents = 1
    num_tasks = 1
    full_comms = True
    # Run mission
    planning_budget = 10
    max_iterations = 1000

    comm_distance = 5
    agent_location_range = [0.1, 0.1]

    tasks = []
    centers = [[0.5, 0.70],
               [0.5, 0.30]]

    covs = [
        [[0.002, 0.0], [0.0, 0.002]],
        [[0.002, 0.0], [0.0, 0.002]],
    ]
    
    
    task = ErgodicCoverageTask()
    task.capabilities_required = 0
    task.t_init = 1e-2
    task.c_clf = 100
    task.w_clf = 200.0
    task.A = num_agents
    task.E_threshold = 1e-10
    task.centers = centers
    task.covs = covs
    task.weights = [0.5,0.5]
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
    scenario.agent_locations = np.array([[0.5,0.5],[0.9,0.9]])
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

    rho = allocator._ergodic_coverage_executor._gmm.build(task.centers,task.covs,task.weights)

    X = np.empty((0,2))
    for local_agent in agents:
        X = np.vstack((X,local_agent.state))
    # Live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=140)
    im = ax.imshow(rho.T, origin="lower", extent=[0, 1, 0, 1], interpolation="bilinear")
    paths = [ax.plot([], [], lw=1.8, label=f"r{j+1}")[0] for j in range(num_agents)]
    pts = ax.plot(X[:, 0], X[:, 1], 'o', ms=6)[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", ncols=2, fontsize=8)

    # Logging
    traj = [X.copy()]


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
                agent, local_agents[agent_id], scenario, planning_budget)
            local_agents[agent_id][agent_id] = agents[agent_id]
            # neighbour_ids = np.arange(num_agents)
            # for neighbor_id in neighbour_ids:
            #     local_agents[neighbor_id][agent_id] = deepcopy(agent)

            planning_time += time.time() - start_time

            # Apply control
            control = agents[agent_id].control[:2]

            agents[agent_id].state = agents[agent_id].model.f(
                agents[agent_id].state, control)
            
        X = np.empty((0,2))
        for local_agent in agents:
            X = np.vstack((X,local_agent.state))

        X = np.minimum(np.maximum(X, 0.0), np.array([1.0, 1.0]))
        traj.append(X.copy())
        arr = np.stack(traj, axis=0)
        
        for j in range(num_agents):
            paths[j].set_data(arr[:, j, 0], arr[:, j, 1])
        pts.set_data(X[:, 0], X[:, 1])
        plt.pause(0.01)

        consensus_step += 1


    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()