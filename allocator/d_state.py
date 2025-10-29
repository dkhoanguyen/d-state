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
from executor import ReachGoalExecutor, ErgodicCoverageExecutor
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
        """
        self._display_progress = display_progress
        self._local_lagrangian = local_lagrangian
        self._global_lagrangian = global_lagrangian

        self._model = model
        self._tasks = tasks
        self._obstacles = obstacles

        self._reach_goal_objective = ReachGoalObjective()

        # Executor
        self._reach_goal_executor = ReachGoalExecutor()
        self._ergodic_coverage_executor = ErgodicCoverageExecutor()

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
        """
        pass

    def plan(self,
             agent: Agent,
             agents: List[Agent],
             scenario: Scenario,
             planning_budget: int
             ) -> List[Agent]:
        """
        Plan the agent's task choice and control actions.
        """
        num_tasks = scenario.num_tasks
        T = 20  # Temperature for gradient update
        alpha = 0.012  # Learning rate
        horizon = 10

        # Plan task
        agent = self._plan_task(
            agent=agent,
            agents=agents,
            scenario=scenario,
            temperature=T,
            max_iter=15,
            annealing_rate=0.0005,
            alpha=alpha,
            horizon=horizon)

        # Plan control actions
        best_task = np.argsort(-agent.action)[0]
        agent = self._plan_control_action(
            agent=agent,
            agents=agents,
            task_id=best_task,
            scenario=scenario,
            horizon=horizon)

        # # Update task progress
        # agent = self._update_progress(agent, scenario, horizon)

        return agent

    def communicate(self, agent: Agent, agents: List[Agent[NDArray[np.int32]]], scenario: Scenario) -> List[Agent[NDArray[np.int32]]]:
        """
        Communication phase: Synchronize agent plans.
        """
        return deepcopy(agent)

    def allocate(self, scenario: Scenario, evaluator: Evaluator) -> Dict[str, Any]:
        """
        Allocate tasks to agents.
        """
        raise NotImplementedError

    # ====== TASK PLANNING FORMULATION ====== #

    def _plan_task(self,
                   agent: Agent,
                   agents: List[Agent],
                   scenario: Scenario,
                   temperature: float,
                   max_iter: int,
                   annealing_rate: float,
                   alpha: float,
                   horizon: int):
        num_tasks = scenario.num_tasks
        current_task_id = agent.allocation[agent.id]

        for fx_step in range(max_iter):
            q = agent.action.copy()
            new_q = q.copy()
            sigma = agent.spec_v[current_task_id]

            E_G_x = sigma * \
                self._calculate_E_G_x(agent, agents, current_task_id, scenario)
            E_C_x = self._calculate_E_C_x_cost(
                agent, agents, current_task_id, scenario)

            for task_id in range(num_tasks):
                sigma = agent.spec_v[task_id]
                E_G_x_i = sigma * \
                    self._calculate_E_G_x_i(agent, agents, task_id, scenario)
                E_C_x_i = self._calculate_E_C_x_i_cost(
                    agent, agents, task_id, scenario)

                delta_E = (E_G_x_i - E_G_x)
                delta_C = E_C_x_i - E_C_x
                combined_E_G_L = self._reward_weight * delta_E - self._cost_weight * delta_C

                entropy_q = self._compute_entropy(q)
                grad = entropy_q + \
                    np.log(q[task_id] + 1e-12) + (combined_E_G_L) / temperature
                new_q[task_id] += alpha * q[task_id] * grad

            temperature = temperature - annealing_rate * temperature
            agent.action = self._normalize_and_tiebreak(new_q)
            best_task = np.argmax(agent.action)
            agent.allocation[agent.id] = best_task

        return agent

    def _get_task_reward(self, agent: Agent[NDArray[np.int32]], task_id: int, scenario: Scenario) -> float:
        return scenario.tasks[task_id].reward

    def _compute_entropy(self, p: np.ndarray) -> float:
        return -np.sum(p * np.log(p + 1e-12))

    def _get_joint_action(self, agent_id: int, task_id: int, current_alloc: NDArray[np.int32], exclude_current: bool = True) -> NDArray[np.int32]:
        task_mask = (current_alloc == task_id)
        if not exclude_current:
            return task_mask.astype(np.int32)
        task_mask[agent_id] = False
        joint_action = np.zeros(len(current_alloc) - 1, dtype=np.int32)
        other_agent_indices = np.delete(
            np.arange(len(current_alloc)), agent_id)
        joint_action = task_mask[other_agent_indices].astype(np.int32)
        return joint_action

    def _calculate_E_G_x(self, agent: Agent, agents: List[Agent], task_id: int, scenario: Scenario) -> float:
        horizon = 10
        if task_id < 0:
            return 0.0
        num_agents = len(agents)
        joint_actions = np.zeros(num_agents)
        for idx, a in enumerate(agents):
            joint_actions[idx] = a.action[task_id]

        E_G = 0.0
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            E_G = self._reach_goal_executor.calculate_reward(
                agent=agent,
                task=scenario.tasks[task_id],
                other_agents=agents,
                horizon=horizon,
                joint_actions=joint_actions
            )
        return E_G

    def _calculate_E_G_x_i(self, agent: Agent, agents: List[Agent], task_id: int, scenario: Scenario) -> float:
        horizon = 10
        if task_id < 0:
            return 0.0
        num_agents = len(agents)
        joint_actions = np.zeros(num_agents)
        for idx, a in enumerate(agents):
            joint_actions[idx] = a.action[task_id]
        joint_actions[agent.id] = 1.0

        E_G_x_i = 0.0
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            E_G_x_i = self._reach_goal_executor.calculate_reward(
                agent=agent,
                task=scenario.tasks[task_id],
                other_agents=agents,
                horizon=horizon,
                joint_actions=joint_actions
            )
        return E_G_x_i

    def _normalize_and_tiebreak(self, q: np.ndarray) -> np.ndarray:
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

    def _calculate_E_C_x_cost(self, agent: Agent, agents: List[Agent], task_id: int, scenario: Scenario) -> float:
        horizon = 10
        if task_id < 0:
            return 0.0
        joint_actions = np.zeros(len(agents))
        for idx, a in enumerate(agents):
            joint_actions[idx] = a.action[task_id]
        joint_actions[agent.id] = agent.action[task_id]

        E_C_x = 0.0
        prob = 0.0
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            prob = self._reach_goal_executor.preprocess(
                joint_action=joint_actions)
            E_C_x = self._reach_goal_executor.calculate_cost(
                agent=agent,
                task=scenario.tasks[task_id],
                other_agents=agents,
                objective_controller=self._reach_goal_objective,
                horizon=horizon,
                joint_actions=joint_actions
            )
        return prob * E_C_x

    def _calculate_E_C_x_i_cost(self, agent: Agent, agents: List[Agent], task_id: int, scenario: Scenario) -> float:
        horizon = 10
        if task_id < 0:
            return 0.0
        joint_actions = np.zeros(len(agents))
        for idx, a in enumerate(agents):
            joint_actions[idx] = a.action[task_id]
        joint_actions[agent.id] = 1.0

        E_C_x_i = 0.0
        prob = 0.0
        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            prob = self._reach_goal_executor.preprocess(
                joint_action=joint_actions)
            E_C_x_i = self._reach_goal_executor.calculate_cost(
                agent=agent,
                task=scenario.tasks[task_id],
                other_agents=agents,
                objective_controller=self._reach_goal_objective,
                horizon=horizon,
                joint_actions=joint_actions
            )
        return prob * E_C_x_i

    def _simulate_trajectory(self, model: Model, x0: np.ndarray, U: np.ndarray, N: int, dt: float = 0.1):
        trajectory = [x0.copy()]
        x = x0.copy()
        for k in range(N):
            u = U[3*k:3*k+2]
            x = model.f(x, u, dt)
            trajectory.append(x.copy())
        return trajectory

    def _update_progress(self, agent: Agent, scenario: Scenario, horizon: int):
        return agent

    def _plan_control_action(self, agent: Agent, agents: List[Agent], task_id: int, scenario: Scenario,horizon:int):
        if task_id < 0:
            return agent

        if scenario.tasks[task_id].task_type == TaskType.REACH_GOAL:
            agent = self._reach_goal_mpc(
                agent, agents, scenario.tasks[task_id], scenario,horizon)
        elif scenario.tasks[task_id].task_type == TaskType.EXPLORE:
            agent = self._ergodic_coverage_mpc(
                agent, agents, scenario.tasks[task_id], scenario,horizon)
        elif scenario.tasks[task_id].task_type == TaskType.HERDING:
            pass
        return agent

    def _reach_goal_mpc(self, 
                        agent: Agent, 
                        other_agents: List[Agent], 
                        task: ReachGoalTask, 
                        scenario: Scenario,
                        horizon: int,
                        w_eps: float = 0.000001):
        lambda_k = np.ones(horizon) * agent.goal_lambda

        # Run optimization
        U_opt, success = self._reach_goal_executor.execute(
            agent=agent,
            other_agents=other_agents,
            scenario=scenario,
            task=task,
            lambda_k=lambda_k,
            objective=self._reach_goal_objective,
            horizon=horizon,
            w_esp=w_eps)

        if success:
            agent.control = U_opt[:3]
            agent.control[2] = U_opt[2 * horizon + 1]

            new_lambda_k = lambda_k.copy()
            trajectory = self._simulate_trajectory(
                model=agent.model,
                x0=agent.state,
                U=U_opt,
                N=horizon
            )
            for k in range(horizon):
                x_k = trajectory[k+1]
                u_with_slack = U_opt[3*k:3*k+3]
                task_cost = self._reach_goal_objective.evaluate(
                    model=agent.model,
                    x=x_k,
                    u=u_with_slack,
                    x_g=task.location,
                    dx_g_dt=task.velocity,
                    gamma=1.0,
                    r=task.radius
                )
                new_lambda_k[k] = max(
                    0, lambda_k[k] + w_eps * task_cost)
                if k == 0:
                    agent.mission_status = task_cost
            agent.goal_lambda = new_lambda_k[0]
        else:
            agent.control = agent.control if agent.control is not None else np.zeros(
                2)
            agent.goal_lambda = lambda_k[0]

        return agent
    
    def _ergodic_coverage_mpc(self, 
                        agent: Agent, 
                        other_agents: List[Agent], 
                        task: ErgodicCoverageTask, 
                        scenario: Scenario,
                        horizon: int,
                        w_eps: float = 0.000001):
        horizon = 1
        lambda_k = np.ones(horizon) * agent.goal_lambda
        # Run optimization
        agent, U_opt, success = self._ergodic_coverage_executor.execute(
            agent=agent,
            other_agents=other_agents,
            scenario=scenario,
            task=task,
            lambda_k=lambda_k,
            horizon=horizon)
        if success:
            agent.control = U_opt            
        return agent
