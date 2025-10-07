#!/usr/bin/env python3

import time
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Tuple
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from scipy.sparse import csc_matrix, eye

from allocator.allocator import Allocator
from datatypes import *
from evaluator import *
from model import Model
from objective import Objective, ReachGoalObjective
from constraint import Constraint, ObstacleAvoidanceConstraint, InterRobotCollisionAvoidance


class DSTATE(Allocator):
    def __init__(self,
                 tasks: List[Task],
                 obstacles: NDArray[np.float64],
                 model: Model = None,
                 display_progress: bool = True,
                 local_lagrangian: bool = False,
                 global_lagrangian: bool = False):
        """
        Initialize the GRAPE allocator with display settings.

        Parameters:
        - display_progress (bool): Whether to print progress during allocation.
        """
        self._display_progress = display_progress
        self._local_lagrangian = local_lagrangian
        self._global_lagrangian = global_lagrangian

        self._model = model
        self._tasks = tasks
        self._obstacles = obstacles

        self._reach_goal_objective = ReachGoalObjective()

        self._objective_dict = {}
        for id, task in enumerate(tasks):
            if task.task_type == TaskType.REACH_GOAL:
                self._objective_dict[id] = ReachGoalObjective()
            elif task.task_type == TaskType.EXPLORE:
                pass
            elif task.task_type == TaskType.HERDING:
                pass

        # Evaluator
        self._reach_goal_evaluator = ReachGoalEvaluator()

        # For keeping track of mission
        self._max_progress = 0.0
        self._max_init = False
        self._progress = []
        self._progress_counter = 0
        self._progress_dt = []

        # Constraints
        self._obstable_avoidance_constraint = ObstacleAvoidanceConstraint()
        self._inter_robot_ca = InterRobotCollisionAvoidance()

        self._reward_weight = 5.0
        self._cost_weight = 0.2

        # Generate constraints
        self._constraint_dict = {}

    def init(self, scenario: Scenario):
        """
        Initialize the allocator with the given scenario.

        Parameters:
        - scenario (Scenario): The scenario containing agents and tasks.
        """

    def plan(self,
             agent: Agent,
             agents: List[Agent],
             scenario: Scenario,
             evaluator: Evaluator,
             planning_budget: int
             ) -> List[Agent]:
        """
        Plan the agent's task choice by updating action probabilities based on expected utilities.

        Args:
            agent: The agent planning its strategy.
            agents: List of all agents in the system.
            scenario: Scenario containing task demands and system parameters.
            evaluator: Utility evaluator for task choices.

        Returns:
            List of agents with updated action for the planning agent.
        """
        num_tasks = scenario.num_tasks

        T = 20  # Temperature for gradient update
        alpha = 0.012  # Learning rate
        rho = 0.7
        horizon = 10

        # Calculate current expected utility (system-wide)
        # PC substep - Primal update
        # Plan task first - optimize for f(x)
        agent = self._plan_task(
            agent=agent,
            agents=agents,
            scenario=scenario,
            evaluator=evaluator,
            temperature=T,
            max_iter=15,
            annealing_rate=0.0005,
            alpha=alpha,
            rho=rho,
            horizon=horizon)
        # Plan control actions later
        best_task = np.argsort(-agent.action)[0]
        agent = self._plan_control_action(
            agent=agent,
            agents=agents,
            task_id=best_task,
            scenario=scenario)

        # # Update task progress
        agent = self._update_progress(agent, scenario, horizon)

        return agent

    def communicate(self, agent: Agent, agents: List[Agent[NDArray[np.int32]]], scenario: Scenario) -> List[Agent[NDArray[np.int32]]]:
        """
        Communication phase: Synchronize agent plans using distributed mutex.

        Parameters:
        - agents (List[Agent[NDArray[np.int32]]]): List of agents with proposed allocations.
        - scenario (Scenario): Scenario data containing communication matrix.

        Returns:
        - List[Agent[NDArray[np.int32]]]: Agents with synchronized allocations.
        """
        return deepcopy(agent)

    def allocate(self, scenario: Scenario, evaluator: Evaluator) -> Dict[str, Any]:
        """
        Allocate tasks to agents based on the scenario and evaluator.

        Parameters:
        - scenario (Scenario): The scenario containing agents and tasks.
        - evaluator (Evaluator): The evaluator to assess allocations.

        Returns:
        - Dict[str, Any]: A dictionary containing the allocation results.
        """
        raise NotImplementedError

# ====== TASK PLANNING FORMULATION ====== #

    def _plan_task(self,
                   agent: Agent,
                   agents: List[Agent],
                   scenario: Scenario,
                   evaluator: Evaluator,
                   temperature: float,
                   max_iter: int,
                   annealing_rate: float,
                   alpha: float,
                   rho: float,
                   horizon: int):
        # Pre - PC -> Non-myopic sims N steps ahead to evaluate how "good" doing one task
        # over the other is
        num_tasks = scenario.num_tasks

        current_task_id = agent.allocation[agent.id]
        individual_rewards = self._calculate_individual_reward(agent=agent,
                                                               scenario=scenario,
                                                               horizon=horizon)

        # Run PC
        for fx_step in range(max_iter):
            # Current action probabilities
            q = agent.action.copy()
            new_q = q.copy()

            individual_reward = individual_rewards[current_task_id]
            sigma = agent.spec_v[current_task_id]

            # Obtain the current utility of the current task assigned to the agent
            E_G_x = sigma * self._calculate_E_G_x(
                agent, agents, current_task_id, scenario)

            # Evaluate the cost of current task x, sj to characteristics of the robot
            E_C_x = self._calculate_E_C_x_cost(
                agent, agents, current_task_id, scenario, evaluator)

            # Evaluate each task by assuming agent chooses it deterministically
            for task_id in range(num_tasks):
                individual_reward = individual_rewards[task_id]
                sigma = agent.spec_v[task_id]

                # Calculate expected utility for this task
                E_G_x_i = sigma * self._calculate_E_G_x_i(
                    agent, agents, task_id, scenario, evaluator, individual_reward)

                # Evaluate the cost of the potential task xi, sj to characteristics of the robot
                E_C_x_i = self._calculate_E_C_x_i_cost(
                    agent, agents, task_id, scenario, evaluator)

                # Update action probabilities using gradient
                # Similar to calculating the gradient of the function
                delta_E = (E_G_x_i - E_G_x)

                # Check cost
                delta_C = E_C_x_i - E_C_x

                # Optimization step
                combined_E_G_L = self._reward_weight * delta_E - self._cost_weight * delta_C

                entropy_q = self._compute_entropy(q)
                grad = entropy_q + \
                    np.log(q[task_id] + 1e-12) + (combined_E_G_L) / temperature

                # NN Update step
                new_q[task_id] += alpha * q[task_id] * grad

            # Temperature annealing - optional I think
            temperature = temperature - annealing_rate * temperature

            # Normalize probabilities and handle ties
            agent.action = self._normalize_and_tiebreak(new_q)
            # Update task allocation
            best_task = np.argmax(agent.action)
            agent.allocation[agent.id] = best_task

        return agent

    def _get_task_reward(self, agent: Agent[NDArray[np.int32]], task_id: int, scenario: Scenario) -> float:
        """
        Computes the reward for a specific task based on the scenario's task demands.

        Parameters:
        - agent (Agent[NDArray[np.int32]]): The agent.
        - task_id (int): The ID of the task to evaluate.
        - scenario (Scenario): The scenario containing task demands.

        Returns:
        - float: The reward value of the specified task.
        """
        return scenario.tasks[task_id].reward

    def _compute_entropy(self, p: np.ndarray) -> float:
        """
        Compute the entropy of a probability distribution.

        Args:
            p (np.ndarray): A numpy array representing a probability distribution.

        Returns:
            float: Entropy of the distribution.
        """
        return -np.sum(p * np.log(p + 1e-12))

    def _get_joint_action(
            self,
            agent_id: int,
            task_id: int,
            current_alloc: NDArray[np.int32],
            exclude_current: bool = True
    ) -> NDArray[np.int32]:
        """
        Generates a joint action vector representing the choices of agents for a task's coalition.

        Parameters:
        - agent_id (int): ID of the current agent.
        - task_id (int): ID of the task.
        - current_alloc (NDArray[np.int32]): Current allocation array.
        - exclude_current (bool): Whether to exclude the current agent from the joint action vector.

        Returns:
        - NDArray[np.int32]: Binary vector where 1 indicates an agent is allocated to the task.
        """
        task_mask = (current_alloc == task_id)
        if not exclude_current:
            return task_mask.astype(np.int32)

        task_mask[agent_id] = False
        joint_action = np.zeros(len(current_alloc) - 1, dtype=np.int32)

        other_agent_indices = np.delete(
            np.arange(len(current_alloc)), agent_id)
        joint_action = task_mask[other_agent_indices].astype(np.int32)

        return joint_action

    def _calculate_E_G_x(self, agent: Agent, agents: List[Agent], task_id: int,
                         scenario: Scenario) -> float:
        """
        Calculate the expected utility for the current task.

        Parameters:
        - agent (Agent): The agent for which the utility is calculated.
        - agents (List[Agent]): List of all agents.
        - task_id (int): The ID of the task.
        - scenario (Scenario): The scenario containing task demands and system parameters.
        - evaluator (Evaluator): Utility evaluator for task choices.    
        """
        horizon = 10
        if task_id < 0:
            return 0.0
        num_agents = len(agents)
        joint_actions = np.zeros(num_agents)
        for idx, a in enumerate(agents):
            joint_actions[idx] = a.action[task_id]

        E_G = 0.0
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            E_G = self._reach_goal_evaluator.calculate_reward(
                agent=agent,
                task=scenario.tasks[task_id],
                other_agents=agents,
                horizon=horizon,
                joint_actions=joint_actions
            )

        return E_G

    def _calculate_E_G_x_i(self, agent: Agent, agents: List[Agent], task_id: int,
                           scenario: Scenario, evaluator: Evaluator,
                           individual_reward: float) -> float:
        """
        Calculate the expected utility for a task allocation.

        Parameters:
        - agent (Agent): The agent for which the utility is calculated.
        - agents (List[Agent]): List of all agents.
        - task_id (int): The ID of the task.
        - scenario (Scenario): The scenario containing task demands and system parameters.
        - evaluator (Evaluator): Utility evaluator for task choices.    
        """
        horizon = 10
        if task_id < 0:
            return 0.0
        num_agents = len(agents)
        joint_actions = np.zeros(num_agents)
        for idx, a in enumerate(agents):
            joint_actions[idx] = a.action[task_id]

        # Agent chooses task_id deterministically
        joint_actions[agent.id] = 1.0

        E_G_x_i = 0.0
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            E_G_x_i = self._reach_goal_evaluator.calculate_reward(
                agent=agent,
                task=scenario.tasks[task_id],
                other_agents=agents,
                horizon=horizon,
                joint_actions=joint_actions
            )
        return E_G_x_i

    def _normalize_and_tiebreak(self, q: np.ndarray) -> np.ndarray:
        """Normalize probabilities and handle ties."""
        q = np.maximum(q, 1e-8)
        q /= np.abs(np.sum(q))
        max_val = np.max(q)
        indices = np.where(q == max_val)[0]
        if len(indices) > 1:
            chosen_index = np.random.choice(indices)
            q[chosen_index] += 0.05 * q[chosen_index]
            q = np.maximum(q, 1e-8)
            q /= np.sum(q)
        return q

    def _calculate_E_C_x_cost(self,
                              agent: Agent,
                              agents: List[Agent],
                              task_id: int,
                              scenario: Scenario,
                              evaluator: Evaluator) -> float:
        if task_id < 0:
            return 0.0
        joint_actions = np.zeros(len(agents))
        for idx, a in enumerate(agents):
            # Probability of choosing task_id
            joint_actions[idx] = a.action[task_id]
        joint_actions[agent.id] = agent.action[task_id]
        prob = evaluator.get_prob(joint_actions)

        expected_cost = 0.0

        # Evaluate constraints
        # Simple reach goal constraint
        cost = 0.0
        horizon = 10
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            x0 = agent.state
            # Propagate the current control over the entire horizon
            U = np.tile(agent.control, horizon)
            trajectory = self._simulate_trajectory(
                model=agent.model,
                x0=x0,
                U=U,
                N=horizon,
            )

            current_task: ReachGoalTask = scenario.tasks[task_id]

            # Reach goal augmented lagrangian
            for k in range(horizon):
                x_k = trajectory[k]
                x_g = current_task.location
                task_cost = self._reach_goal_objective.evaluate(
                    model=agent.model,
                    x=x_k,
                    u=agent.control,
                    x_g=x_g,
                    # Assuming no time derivative for goal - static goal for now
                    dx_g_dt=np.zeros(2),
                    gamma=1.0,  # Example gamma value
                    r=0.1
                )
                w_eps = 0.000001
                # Form augmented lagrangian
                cost += agent.goal_lambda * task_cost + \
                    (w_eps / 2) * task_cost**2

        # Ensure that it is the expected cost
        for k in range(1, len(prob)):
            expected_cost += prob[k] * cost
        return expected_cost

    def _calculate_E_C_x_i_cost(self,
                                agent: Agent,
                                agents: List[Agent],
                                task_id: int,
                                scenario: Scenario,
                                evaluator: Evaluator) -> float:
        joint_actions = np.zeros(len(agents))
        for idx, a in enumerate(agents):
            joint_actions[idx] = a.action[task_id]

        # Agent chooses task_id deterministically
        joint_actions[agent.id] = 1.0

        prob = evaluator.get_prob(joint_actions)
        expected_cost = 0.0

        # Evaluate constraints
        # Simple reach goal constraint
        cost = 0.0
        horizon = 10
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            x0 = agent.state
            # Propagate the current control over the entire horizon
            U = np.tile(agent.control, horizon)
            trajectory = self._simulate_trajectory(
                model=agent.model,
                x0=x0,
                U=U,
                N=horizon,
            )

            current_task: ReachGoalTask = scenario.tasks[task_id]

            # Reach goal augmented lagrangian
            for k in range(horizon):
                x_k = trajectory[k]
                x_g = current_task.location
                task_cost = self._reach_goal_objective.evaluate(
                    model=agent.model,
                    x=x_k,
                    u=agent.control,
                    x_g=x_g,
                    # Assuming no time derivative for goal - static goal for now
                    dx_g_dt=np.zeros(2),
                    gamma=1.0,  # Example gamma value
                    r=0.1
                )
                w_eps = 0.000001
                # task_cost = np.maximum(task_cost, 0.0)
                # Form augmented lagrangian
                cost += agent.goal_lambda * task_cost + \
                    (w_eps / 2) * task_cost**2

        # Ensure that it is the expected cost
        for k in range(1, len(prob)):
            expected_cost += prob[k] * cost
        return expected_cost

    def _simulate_trajectory(self,
                             model: Model,
                             x0: np.ndarray,
                             U: np.ndarray,
                             N: int,
                             dt: float = 0.1):
        trajectory = [x0.copy()]
        x = x0.copy()
        for k in range(N):
            u = U[3*k:3*k+2]
            x = model.f(x, u, dt)
            trajectory.append(x.copy())
        return trajectory

    def _calculate_individual_reward(self,
                                     agent: Agent,
                                     scenario: Scenario,
                                     horizon: int):

        def compute_instantaneous_reward(data, t):
            """
            Compute instantaneous reward based on the angle of the raw change vector relative to the x-axis, in radians.

            Parameters:
            data (np.ndarray): Input dataset (raw, not normalized)
            t (np.ndarray): Time array
            dt (float): Time step

            Returns:
            np.ndarray: Instantaneous reward array (based on angles)
            np.ndarray: Angles in radians
            np.ndarray: Raw changes (Δdata / dt)
            """
            # Compute raw changes
            delta_data = np.diff(np.diff(data))  # Δdata
            delta_t = np.linspace(0, t, len(delta_data))

            # Compute angles with x-axis using arctan2 (in radians)
            angles_rad = np.arctan2(delta_data, delta_t)

            # Compute instantaneous reward based on angles
            r_instant = np.zeros(len(angles_rad))

            for i in range(len(angles_rad)):
                if angles_rad[i] > 0:  # Negative angle (decrease)
                    # Normalize to [0, 1], sharper decrease → higher reward
                    r_instant[i] = angles_rad[i]
                else:  # Positive angle (increase)
                    r_instant[i] = 0.0

            return r_instant
        num_tasks = scenario.num_tasks
        hypothetical_performance = []
        # Hypothetical performance for each task (rough estimation)
        for task_id in range(num_tasks):
            if scenario.tasks[task_id].task_type != TaskType.REACH_GOAL:
                hypothetical_performance.append(0.0)
                continue
            # Augment the task reward with individual reward to incentivise robot to
            # prefer tasks that they are more capable
            # Hypothetical progress given cunrrent control and state
            current_task: ReachGoalTask = scenario.tasks[task_id]
            x0 = agent.state
            x_g = current_task.location

            # Simple estimation of the control
            # This part can technically be heuristic
            u0 = (x_g - x0)
            v_max = agent.u_bounds[0, 1]
            if np.linalg.norm(u0) > v_max:
                u0 = v_max * (u0 / np.linalg.norm(u0))
            u0 = np.hstack((u0, 0.0))

            u = np.tile(u0, horizon)
            trajectory = self._simulate_trajectory(
                model=agent.model,
                x0=x0,
                U=u,
                N=horizon,
            )

            progress = []
            for k in range(horizon):
                x_k = trajectory[k]
                progress_k = self._reach_goal_objective.progress(
                    model=agent.model,
                    x=x_k,
                    u=u0,
                    x_g=x_g,
                    dx_g_dt=np.zeros(2),
                    gamma=1.0,
                    r=0.1
                )
                progress.append(progress_k)

            individual_reward = 100.0 * \
                np.sum(compute_instantaneous_reward(progress, 10))
            hypothetical_performance.append(individual_reward)
        return hypothetical_performance

    def _update_progress(self, agent: Agent,
                         scenario: Scenario,
                         horizon: int):
        # Check and update task progress
        task_id = agent.allocation[agent.id]

        current_task: ReachGoalTask = scenario.tasks[task_id]

        x0 = agent.state
        x_g = current_task.location

        # Hypothetical progress
        u0 = (x_g - x0)
        v_max = agent.u_bounds[0, 1]
        if np.linalg.norm(u0) > v_max:
            u0 = v_max * (u0 / np.linalg.norm(u0))
        u0 = np.hstack((u0, 0.0))

        u = np.tile(u0, horizon)
        trajectory = self._simulate_trajectory(
            model=agent.model,
            x0=x0,
            U=u,
            N=horizon,
        )

        hypo_progress = []
        for k in range(horizon):
            x_k = trajectory[k]
            progress_k = self._reach_goal_objective.progress(
                model=agent.model,
                x=x_k,
                u=u0,
                x_g=x_g,
                dx_g_dt=np.zeros(2),
                gamma=1.0,
                r=0.1
            )
            hypo_progress.append(progress_k)

        # True progress
        u0 = agent.control

        u = np.tile(u0, horizon)
        trajectory = self._simulate_trajectory(
            model=agent.model,
            x0=x0,
            U=u,
            N=horizon,
        )

        true_progress = []
        for k in range(horizon):
            x_k = trajectory[k]
            progress_k = self._reach_goal_objective.progress(
                model=agent.model,
                x=x_k,
                u=u0,
                x_g=x_g,
                dx_g_dt=np.zeros(2),
                gamma=1.0,
                r=0.1
            )
            true_progress.append(progress_k)

        hypo_progress = np.diff(hypo_progress)
        true_progress = np.diff(true_progress)

        progress = np.maximum(np.minimum(np.abs(
            true_progress[-1] - true_progress[0]) / np.abs(hypo_progress[-1] - hypo_progress[0]), 1.0), 0.0)
        d_sigma = 0.05 * (1 - progress)

        # if np.abs(np.sum(true_progress)) >= 20.0:
        #     agent.spec_v[task_id] -= d_sigma

        return agent

    # ===== TASK EXECUTION FORMULATION =====
    def _plan_control_action(self,
                             agent: Agent,
                             agents: Agent,
                             task_id: int,
                             scenario: Scenario):
        """
        Plan the agent's control actions using MPC optimization for the given task.

        Parameters:
        - agent (Agent): The agent for which to plan control actions.
        - task_id (int): The ID of the task to plan for.

        Returns:
        - Agent: The agent with updated control and goal_lambda.
        """
        horizon = 10
        dt = 0.1
        u_max = agent.u_bounds[0, 1]
        u_min = agent.u_bounds[0, 0]
        w_eps = 0.000001
        gamma_g = 1.0

        # Ignore if invalid task_id
        if task_id < 0:
            return agent

        current_task: ReachGoalTask = scenario.tasks[task_id]
        if current_task.task_type == TaskType.REACH_GOAL:
            x_0 = agent.state
            x_g = current_task.location
            v_g = np.zeros((2))
            lambda_k = np.ones(horizon) * agent.goal_lambda

            U0 = np.tile(agent.control, horizon)
            bounds = [(u_min, u_max)] * (2 * horizon) + \
                [(-np.inf, np.inf)] * horizon

            constraints = []

            # Perform MPC optimization
            res = minimize(lambda U: self._mpc_cost(
                model=agent.model, U=U, x_0=x_0, x_g=x_g,
                v_g=v_g, N=horizon, dt=dt, w_eps=w_eps, lambda_k=lambda_k),
                U0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 30, 'ftol': 0.001, 'disp': False, 'eps': 1e-4})

            if res.success:
                U_opt = res.x
                # Update agent control with the first control input [u_x_0, u_y_0]
                agent.control = U_opt[:3]
                agent.control[2] = U_opt[2*horizon + 1]

                # Update Lagrange multipliers
                new_lambda_k = lambda_k.copy()
                trajectory = self._simulate_trajectory(
                    model=agent.model,
                    x0=x_0,
                    U=U_opt,
                    N=horizon,
                    dt=dt
                )
                for k in range(horizon):
                    x_k = trajectory[k+1]
                    u_with_slack = U_opt[3*k:3*k+3]  # [u_x_k, u_y_k, eps_k]
                    task_cost = self._reach_goal_objective.evaluate(
                        model=agent.model,
                        x=x_k,
                        u=u_with_slack,
                        x_g=x_g,
                        dx_g_dt=np.zeros(2),
                        gamma=gamma_g,
                        r=0.1
                    )
                    new_lambda_k[k] = max(0, lambda_k[k] + w_eps * task_cost)
                    if k == 0:
                        agent.mission_status = task_cost
                # Update agent's goal_lambda
                agent.goal_lambda = new_lambda_k[0]
            else:
                # If optimization fails, keep current control and lambda
                agent.control = agent.control if agent.control is not None else np.zeros(
                    2)
                agent.goal_lambda = lambda_k[0]
        elif current_task.task_type == TaskType.EXPLORE:
            pass
        elif current_task.task_type == TaskType.HERDING:
            pass
        return agent

    def _mpc_cost(self,
                  model: Model,
                  U: np.ndarray,
                  x_0: np.ndarray,
                  x_g: np.ndarray,
                  v_g: np.ndarray,
                  N: int,
                  dt: float,
                  w_eps: float,
                  lambda_k: np.ndarray) -> float:
        cost = 0.0
        trajectory = self._simulate_trajectory(
            model=model,
            x0=x_0,
            U=U,
            N=N,
            dt=dt
        )

        for k in range(N):
            u = U[3*k:3*k+2]  # Extract [u_x_k, u_y_k]
            # Control effort
            cost += 0.5 * u[0]**2 + 0.5 * u[1]**2

            # Task objective
            # Augmented Lagrangian for goal CBF constraint
            x_k = trajectory[k+1]
            u_with_slack = U[3*k:3*k+3]  # [u_x_k, u_y_k, eps_k]
            task_cost = self._reach_goal_objective.evaluate(
                model=model,
                x=x_k,
                u=u_with_slack,
                x_g=x_g,
                dx_g_dt=v_g,
                gamma=1.0,
                r=0.1
            )

            # Add Lagrangian and penalty terms
            cost += lambda_k[k] * task_cost + (w_eps / 2) * task_cost**2

        # Slack penalties
        for k in range(N):
            eps = U[3*k+2]  # Slack variable for step k
            cost += 1000.0 * eps**2

        return cost

    def _inter_robot_collision_avoidance(self,
                                         agent: Agent,
                                         agents: Agent,
                                         horizon=int,
                                         dt: float = 0.1):
        constraints = []
        control_dim = 2
        x_0 = agent.state.copy()  # Initial state

        def predict_agent_states(agent_id: int) -> np.ndarray:
            """Predict states and velocity of another agent using SingleIntegrator model."""
            x_j = agents[agent_id].state.copy()
            # Assume zero control input for other agents (can be modified if u_j available)
            # 2D control input (excluding slack)
            u_j = np.zeros(control_dim - 1)
            v_j = agents[agent_id].control[:2]
            states = np.zeros((horizon, len(x_j)))
            states[0] = x_j
            for k in range(1, horizon):
                states[k] = agents[agent_id].model.f(
                    states[k - 1], u_j, dt)  # x_j[k] = x_j[k-1] + u_j * dt
            return states, v_j

        for k in range(horizon):
            for agent_id in range(len(agents)):
                if agent_id == agent.id:
                    continue

                # Predict other agent's state and velocity
                x_j_traj, v_j = predict_agent_states(agent_id)
                x_j_k = x_j_traj[k]

                dist = np.linalg.norm(x_0[:2] - x_j_k[:2])

                # Ignore if too far away
                if dist >= 100:
                    continue

                # Constraint function for step k and agent pair
                def constraint_fun(U, k=k, x_j_k=x_j_k, v_j=v_j):
                    # Propagate state up to step k
                    x_k = x_0.copy()
                    for i in range(k):
                        u_i = U[i * control_dim:(i + 1) * control_dim]
                        A, B = agent.model.df(x_k, u_i[:2], dt)
                        x_k = A @ x_k + B @ u_i[:2]  # Linearized dynamics
                    u_k = U[k * control_dim:(k + 1) * control_dim]
                    u_k = np.hstack((u_k, 0.0))
                    # Compute CBF constraint
                    return self._inter_robot_ca.evaluate(
                        model=agent.model,
                        x_i=x_k,
                        u=u_k,
                        x_j=x_j_k,
                        v_j=v_j,
                        gamma=1.0,
                        r=30.0,
                        t=1.0
                    )

                constraints.append({'type': 'ineq', 'fun': constraint_fun})
        return constraints

    def _obs_collision_avoidance(self, agent: Agent,
                                 scenario: Scenario,
                                 horizon: int,
                                 dt: float = 0.1):

        x_0 = agent.state[:2].copy()
        num_obs = scenario.obstacles.shape[0]
        constraints = []
        for k in range(horizon):
            v_o = np.zeros((2,))
            # Time-varying obstacle position: o_k = o_0 + k * dt * v_o
            for obs_id in range(num_obs):
                o_k = scenario.obstacles[obs_id, :]

                # Constraint function for step k and obstacle
                def constraint_fun(U, k=k):
                    # Propagate state up to step k
                    x_k = x_0.copy()
                    for i in range(k):
                        u_i = U[3*i:3*i+2]  # Extract u_x,i, u_y,i
                        A, B = agent.model.df(x_k, u_i, dt)
                        x_k = A @ x_k + B @ u_i
                    u_k = U[3*k:3*k+3]  # Extract [u_x,k, u_y,k, eps_k]
                    # Compute CBF constraint
                    return self._obstable_avoidance_constraint.evaluate(
                        model=agent.model,
                        x=x_k,
                        u=u_k,
                        o=o_k,
                        do_dt=v_o,
                        gamma=1.0,
                        r=10.0,
                        t=k * dt
                    )
                constraints.append({'type': 'ineq', 'fun': constraint_fun})
        return constraints
