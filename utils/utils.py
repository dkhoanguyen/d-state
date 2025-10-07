#!/usr/bin/env python3
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.legend_handler import HandlerBase
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from matplotlib.path import Path
from matplotlib.patches import Rectangle,FancyBboxPatch

from datatypes import Scenario, TaskType, Agent, AgentType


def generate_initial_action(scenario: Scenario, deterministic: bool = True) -> NDArray[np.float64]:
    """
    Generates a joint action matrix for agents and tasks.

    Parameters:
    - scenario (Scenario): Scenario data containing agent and task information, including initial allocation.
    - deterministic (bool): If True, the matrix reflects deterministic choices from initial_allocation (1 for assigned task, 0 otherwise).
                           If False, each agent has equal probability (1/num_tasks) for all tasks.

    Returns:
    - NDArray[np.float64]: A matrix of shape (num_agents, num_tasks) where rows are agents and columns are tasks.
                          In deterministic mode, cell [i, j] is 1 if agent i is assigned to task j, else 0.
                          In non-deterministic mode, cell [i, j] is 1/num_tasks for all tasks j.
    """
    joint_actions = np.zeros(
        (scenario.num_agents, scenario.num_tasks), dtype=np.float64)

    if deterministic:
        # Set joint_actions[i, j] = 1 where agent i is assigned to task j in initial_allocation
        for agent_id in range(scenario.num_agents):
            task_id = scenario.initial_allocation[agent_id]
            if task_id != -1:  # Skip void assignments
                joint_actions[agent_id, task_id] = 1.0
    else:
        # Assign equal probability (1/num_tasks) for each task for each agent
        joint_actions.fill(1.0 / scenario.num_tasks)

    return joint_actions


def plot_final_allocation(
        agents,
        scenario: Scenario,
        result: Dict[str, Any],
        plot_comms_links: bool = False,
        save_path: str | None = None) -> None:
    """
    Visualizes the final task allocation for agents, with agents colored by their assigned task.

    Parameters:
    - scenario (Scenario): Scenario data containing agent and task locations and communication matrix.
    - result (Dict[str, Any]): Result dictionary from allocate, containing final_allocation.
    - save_path (str | None): Path to save the plot as an image. If None, the plot is displayed.
    """
    # agent_locations = scenario.agent_locations
    agent_locations = []
    for agent in agents:
        agent_locations.append(agent.state)
    task_locations = scenario.task_locations
    task_demands = scenario.task_demands
    agent_comm_matrix = scenario.agent_comm_matrix
    final_allocation = result['final_allocation']

    # Create color map for tasks
    colours = plt.cm.viridis(np.linspace(0, 1, len(task_locations)))

    # Normalize task demands for marker sizes
    task_demands_sizes = (task_demands - task_demands.min()) / \
        (task_demands.max() - task_demands.min() + 1e-10) * 300 + 50

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_ylim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Plot communication links
    if plot_comms_links:
        for i, j in zip(*np.where(np.triu(agent_comm_matrix) > 0)):
            ax.plot([agent_locations[i, 0], agent_locations[j, 0]],
                    [agent_locations[i, 1], agent_locations[j, 1]],
                    '-', color='gray', linewidth=0.5)

    # Plot agents, colored by assigned task
    for i, location in enumerate(agent_locations):
        task_id = final_allocation[i]
        colour = 'black' if task_id == -1 else colours[task_id]
        ax.plot(location[0], location[1], 'o', markersize=5,
                markeredgecolor='gray', markerfacecolor=colour)

    # Plot tasks with unique colors and scaled sizes
    for i, location in enumerate(task_locations):
        ax.scatter(location[0], location[1], s=task_demands_sizes[i], c=[
                   colours[i]], label=f'Task {i+1}', marker='s')

    plt.title('GRAPE Task Allocation Result')
    plt.grid(False)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_total_utility_over_time(result: Dict[str, Any], save_path: str | None = None) -> None:
    """
    Plots the total utility over time (consensus steps) for the GRAPE allocation.

    Parameters:
    - scenario (Scenario): Scenario data containing agent and task information.
    - result (Dict[str, Any]): Result dictionary from allocate, containing history of total utilities.
    - save_path (str | None): Path to save the plot as an image. If None, the plot is displayed.
    """
    total_utility_history = result['history']['total_utility']

    # Plot total utility over consensus steps
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = range(len(total_utility_history))
    ax.plot(steps, total_utility_history, color='blue', label='Total Utility')

    # Set labels and title
    ax.set_xlabel('Consensus Step')
    ax.set_ylabel('Total Utility')
    ax.set_title('GRAPE: Total Utility Over Time')
    ax.grid(True)
    ax.legend()

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_agent_trajectories(agents, result, scenario, save_format=None, save_path=None, ax=None):
    """
    Plot the final trajectories of agents' states, with task locations and obstacles marked.
    DRONE agents have red trajectories, others have blue trajectories.
    Optionally save as PDF.

    Parameters:
    - agents (list): List of agent objects with type attribute.
    - result (dict): The result dictionary containing state history.
    - scenario (Scenario): The scenario containing task locations, task demands, and obstacles.
    - save_format (str, optional): 'pdf' to save the plot, None to display only.
    - save_path (str, optional): Path to save the plot (e.g., 'trajectories.pdf').
    """
    state_history = result['history']['states']  # List of [num_agents, 2] arrays
    num_agents = len(state_history[0])
    task_locations = scenario.task_locations
    task_demands = scenario.task_demands

    # Create color map for tasks
    colours = plt.cm.viridis(np.linspace(0, 1, len(task_locations)))
    # Normalize task demands for marker sizes
    task_demands_sizes = (task_demands - task_demands.min()) / \
        (task_demands.max() - task_demands.min() + 1e-10) * 300 + 50

    # Create figure and main axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_ylim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])

    # # Draw a rectangle on the plot
    # rect = Rectangle((-73.5, 73.5), 77, -28, linewidth=1, edgecolor='gray', facecolor='none',
    #                  transform=ax.transData, zorder=5)

    # # Draw a rounded rectangle on the plot
    # rect = FancyBboxPatch((-75, 75), 60, -25, boxstyle='round,pad=1', linewidth=0.8,
    #                       edgecolor='black', facecolor='none', transform=ax.transData, zorder=5)
    # ax.add_patch(rect)

    # Plot task locations with unique colors and scaled sizes
    for i, location in enumerate(task_locations):
        # ax.scatter(location[0], location[1], s=task_demands_sizes[i], c=[colours[i]],
        #            label=f'Task {i+1}', marker='s')
        colors = "#70be47"
        if i < 2:
            colors = "#49b89b"
        
        ax.scatter(location[0], location[1], s=400, c=colors,
                   marker='s', linewidths=5, label=f'Task {i+1}' if i == 0 else "",zorder=39)
        
    # ax.annotate(r'Task $t_1$',(task_locations[3,0], task_locations[3,1] - 0.1 * np.max(task_locations)),  # Offset below
    #             ha='center', va='top', fontsize=28, color='black')
    
    # ax.annotate(r'Task $t_2$',(task_locations[0,0], task_locations[0,1] - 0.395 * np.max(task_locations)),  # Offset below
    #             ha='center', va='top', fontsize=28, color='black')

    # # Plot obstacles as dashed circles
    # for i, location in enumerate(scenario.obstacles):
    #     circle = plt.Circle(
    #         location, 20, color='black', fill=False, linestyle='--', linewidth=3)
    #     ax.add_patch(circle)

    # Plot trajectories for each agent
    for agent_id in range(num_agents):
        x_coords = [state_history[t][agent_id, 0]
                    for t in range(len(state_history))]
        y_coords = [state_history[t][agent_id, 1]
                    for t in range(len(state_history))]
        agent = agents[agent_id]
        # Set color based on agent type
        color = "#ffe100" if agent.type == AgentType.DRONE else "#40B9FF"
        alpha=1
        # Plot trajectory with markers at each point
        for i in range(len(x_coords) - 1):
            if i < num_agents / 2:
                alpha = (5 + i + 1) / len(x_coords)  # older -> lower alpha
            else:
                alpha = (i + 1) / len(x_coords)
            alpha = np.minimum(alpha, 1)
            ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=color, alpha=alpha,
                    markersize=3, markeredgecolor=color, markerfacecolor=color,
                    lw=2,zorder=40)
    
    for agent_id in range(num_agents):
        x_coords = [state_history[t][agent_id, 0]
                    for t in range(len(state_history))]
        y_coords = [state_history[t][agent_id, 1]
                    for t in range(len(state_history))]
        agent = agents[agent_id]
        # Set color based on agent type
        color = "#ffe100" if agent.type == AgentType.DRONE else "#40B9FF"
        # Plot agent shape at final step
        final_x, final_y = state_history[-1][agent_id,
                                             0], state_history[-1][agent_id, 1]
        
        if agent_id == 8:
            final_x, final_y = state_history[-20][agent_id,
                                             0], state_history[-20][agent_id, 1]
        if agent.type == AgentType.DRONE:
            # Central circle
            ax.plot(final_x, final_y, 'o', markersize=9, markeredgecolor='black',
                    markerfacecolor=color,zorder=41)
            # Four surrounding circles
            offsets = [(2.5, 0), (-2.5, 0), (0, 2.5), (0, -2.5)]
            for dx, dy in offsets:
                ax.plot(final_x + dx, final_y + dy, 'o', markersize=9, markeredgecolor='black',
                        markerfacecolor=color,zorder=41)
        else:
            x, y = state_history[-2][agent_id,
                                     0], state_history[-2][agent_id, 1]
            x_next, y_next = final_x, final_y
            heading = np.arctan2(y_next - y, x_next - x)
            # Triangle with heading
            # heading = agent.heading if hasattr(agent, 'heading') else 0.0  # Assume heading in radians
            ax.scatter([final_x], [final_y], marker=(3, 0, np.degrees(heading)), s=450,
                       edgecolor='black', facecolor=color,zorder=40)

    # # Custom legend
    # # Simple custom legend with 5 circles for DRONE
    # legend_elements = [
    #     # DRONE: red line for trajectory, red circles for agent shape
    #     Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markeredgecolor='blue',
    #            markersize=15, label='Unicycle'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black',
    #            markersize=10, label='Quadcopter'),
    #     # Additional circles for DRONE to mimic quadcopter (offset in legend for clarity)
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black',
    #            markersize=6, alpha=0),  # Placeholder for alignment
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black',
    #            markersize=6, alpha=0),  # Placeholder for alignment
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black',
    #            markersize=6, alpha=0),  # Placeholder for alignment
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black',
    #            markersize=6, alpha=0),  # Placeholder for alignment
    # ]
    # ax.legend(handles=legend_elements, handler_map={
    #     legend_elements[1]: HandlerMultiMarker(offsets=[(0, 0), (0.45, 0), (-0.45, 0), (0, 1.3), (0, -1.3)]),
    # },fontsize=25,bbox_to_anchor=(-0.01, 1.005), loc='upper left', frameon=False,)

    if ax is None:
        plt.grid(False)

    # Save plot if requested
    if save_format and save_path:
        try:
            if save_format.lower() == 'pdf':
                plt.savefig(save_path, format='pdf', bbox_inches='tight')
                print(f"Plot saved as PDF: {save_path}")
                plt.close(fig)  # Close the figure to prevent display
            else:
                print(
                    f"Unsupported save_format '{save_format}'. Supported format: 'pdf'. Displaying instead.")
                if ax is None:
                    plt.show()
        except Exception as e:
            print(f"Error saving plot: {e}. Displaying instead.")
            if ax is None:
                plt.show()
    else:
        if ax is None:
            plt.show()


def animate_agent_trajectories(agents, result, scenario, save_format=None, save_path=None, ax=None, fig=None):
    """
    Create an animated plot of agents' states over time, with task locations marked, styled like plot_final_allocation.
    Agents are colored by their current task allocation at each step. Optionally save as GIF or MP4.
    Parameters:
    - result (dict): The result dictionary containing state history and allocation history.
    - scenario (Scenario): The scenario containing task locations and task demands.
    - save_format (str, optional): 'gif' or 'mp4' to save the animation, None to display only.
    - save_path (str, optional): Path to save the animation (e.g., 'animation.gif' or 'animation.mp4').
    """
    state_history = result['history']['states']  # List of [num_agents, 2] arrays
    # List of [num_agents] arrays
    allocation_history = result['history']['allocation']
    # Array of utility values per step
    final_utilities = result['history']['total_utility']

    num_agents = len(state_history[0])
    num_steps = len(state_history)
    # task_locations = scenario.task_locations
    # task_demands = scenario.task_demands
    # # Create color map for tasks
    # colours = plt.cm.viridis(np.linspace(0, 1, len(task_locations)))
    # # Normalize task demands for marker sizes
    # task_demands_sizes = (task_demands - task_demands.min()) / \
    #     (task_demands.max() - task_demands.min() + 1e-10) * 300 + 50
    # # Create figure and main axes
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 10))
    

        ax.set_aspect('equal')
        # ax.set_xlim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
        # ax.set_ylim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

    # Initialize scatter plots for agents
    agent_plots = []
    for i in range(num_agents):
        # Assuming agents is a list of agent objects with type, position, and heading
        agent = agents[i]
        if agent.type == AgentType.DRONE:
            # Create 5 circles to resemble a quadcopter: 1 central, 4 around it
            plots = []
            # Central circle
            central = ax.plot([], [], 'o', markersize=9, markeredgecolor='black',
                              markerfacecolor='black',
                              zorder=41)[0]
            plots.append(central)
            # Four surrounding circles (smaller, offset in a cross pattern)
            for offset in [(0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2)]:
                corner = ax.plot([], [], 'o', markersize=9, markeredgecolor='black',
                                 markerfacecolor='black',zorder=41)[0]
                plots.append(corner)
            agent_plots.append(plots)
        else:  # Default for CAR or other types
            # Initialize with empty plot, will update with scatter in update function
            plot = ax.scatter([], [], marker='^', s=350, edgecolor='black',
                              facecolor='black',
                              zorder=40)
            agent_plots.append([plot])
    # # Plot tasks with unique colors and scaled sizes
    # for i, location in enumerate(task_locations):
    #     colors = "#70be47"
    #     if i < 2:
    #         colors = "#49b89b"
    #     ax.scatter(location[0], location[1], s=350, c=colors,
    #                label=f'Task {i+1}', marker='s',lw=5,zorder=39)

    # for i, location in enumerate(scenario.obstacles):
    #     circle = plt.Circle(
    #         location, 10, color=colours[i], fill=False, linestyle='--', linewidth=1.5)
    #     ax.add_patch(circle)
    plt.grid(False)

    def update(frame):
        print(f"Done step {frame}")
        # Update agent positions and colors for the current frame
        artists = []
        for i, plots in enumerate(agent_plots):
            task_id = allocation_history[frame][i]
            colour = 'black' 
            x, y = state_history[frame][i, 0], state_history[frame][i, 1]
            agent = agents[i]
            if agent.type == AgentType.DRONE:
                colour = "#ffe100"
                # Update central circle
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                # Update four surrounding circles
                offsets = [(2.5, 0), (-2.5, 0), (0, 2.5), (0, -2.5)]
                for j, (dx, dy) in enumerate(offsets):
                    plots[j + 1].set_data([x + dx], [y + dy])
                    plots[j + 1].set_markerfacecolor(colour)
                artists.extend(plots)
            else:
                # Triangle with heading
                colour = "#40B9FF"
                x_next, y_next = x, y
                if frame < len(state_history)-1:
                    x_next, y_next = state_history[frame + 1][i, 0], state_history[frame + 1][i, 1]
                heading = np.arctan2(y_next - y, x_next - x)
                plots[0].remove()
                # Create new scatter plot with rotated triangle
                plots[0] = ax.scatter([x], [y], marker=(3, 0, np.degrees(heading - 90)), s=350,
                                      edgecolor="black", facecolor=colour,zorder=40)
                artists.append(plots[0])
        # ax.set_title(f'Step {frame}')
        return artists
    # Create animation (play once)
    ani = FuncAnimation(fig, update, frames=num_steps,
                        interval=50, blit=False, repeat=False)  # blit=False due to histogram
    # Save animation if requested
    if save_format and save_path:
        try:
            if save_format.lower() == 'gif':
                ani.save(save_path, writer='pillow', fps=20)
                print(f"Animation saved as GIF: {save_path}")
            elif save_format.lower() == 'mp4':
                ani.save(save_path, writer='ffmpeg', fps=20)
                print(f"Animation saved as MP4: {save_path}")
            else:
                print(
                    f"Unsupported save_format '{save_format}'. Supported formats: 'gif', 'mp4'. Displaying instead.")
                plt.show()
        except Exception as e:
            print(f"Error saving animation: {e}. Displaying instead.")
            plt.show()
    else:
        plt.show()
    return ani


def animate_agent_trajectories_additional(agents, result, scenario, save_format=None, save_path=None):
    """
    Create an animated plot of agents' states over time, with task locations marked, styled like plot_final_allocation.
    Agents are colored by their current task allocation at each step. Optionally save as GIF or MP4.

    Parameters:
    - result (dict): The result dictionary containing state history and allocation history.
    - scenario (Scenario): The scenario containing task locations and task demands.
    - save_format (str, optional): 'gif' or 'mp4' to save the animation, None to display only.
    - save_path (str, optional): Path to save the animation (e.g., 'animation.gif' or 'animation.mp4').
    """
    state_history = result['history']['states']  # List of [num_agents, 2] arrays
    # List of [num_agents] arrays
    allocation_history = result['history']['allocation']
    # Array of utility values per step
    final_utilities = result['history']['total_utility']

    num_agents = len(state_history[0])
    num_steps = len(state_history)
    task_locations = scenario.task_locations
    task_demands = scenario.task_demands

    # Create color map for tasks
    colours = plt.cm.viridis(np.linspace(0, 1, len(task_locations)))

    # Normalize task demands for marker sizes
    task_demands_sizes = (task_demands - task_demands.min()) / \
        (task_demands.max() - task_demands.min() + 1e-10) * 300 + 50

    # Create figure and main axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_ylim([-np.max(task_locations) * 1.4 - 0.5 *
                np.max(task_locations), np.max(task_locations) * 1.4])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Create inset axes for histogram in bottom-right corner
    try:
        ax_hist = inset_axes(ax, width="35%", height="20%", loc='lower left',
                             bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                             bbox_transform=ax.transAxes)
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        ax_hist.set_xlabel('')
        ax_hist.set_ylabel('')
        ax_hist.set_xlim(-0.6, len(task_locations) - 0.4)
        for spine in ax_hist.spines.values():
            spine.set_visible(False)
    except Exception as e:
        print(f"Error creating inset axes: {e}")
        ax_hist = None

    # Create inset axes for line plot to the right of histogram
    try:
        ax_line = inset_axes(ax, width="60%", height="20%", loc='lower right',
                             bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                             bbox_transform=ax.transAxes)
        # Disable ticks, tick labels, and axis labels for line plot
        ax_line.set_xticks([])
        ax_line.set_yticks([])
        ax_line.set_xlabel('')
        ax_line.set_ylabel('')
        # Remove all spines around line plot
        for spine in ax_line.spines.values():
            spine.set_visible(False)
        ax_line.grid(True)  # Enable grid for line plot
    except Exception as e:
        print(f"Error creating line plot inset axes: {e}")
        ax_line = None

    # Initialize scatter plots for agents
    agent_plots = []
    for i in range(num_agents):
        # Assuming agents is a list of agent objects with type, position, and heading
        agent = agents[i]
        if agent.type == AgentType.DRONE:
            # Create 5 circles to resemble a quadcopter: 1 central, 4 around it
            plots = []
            # Central circle
            central = ax.plot([], [], 'o', markersize=6, markeredgecolor='gray',
                              markerfacecolor='black')[0]
            plots.append(central)
            # Four surrounding circles (smaller, offset in a cross pattern)
            for offset in [(0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2)]:
                corner = ax.plot([], [], 'o', markersize=6, markeredgecolor='gray',
                                 markerfacecolor='black')[0]
                plots.append(corner)
            agent_plots.append(plots)
        # elif agent.type == AgentType.DIFF_DRIVE:
        #     # Draw an arrow for differential drive robots
        #     arrow = ax.plot([], [], '-', linewidth=2, color='black',
        #                     label=f'Agent {i}' if i < len(task_locations) else "")[0]
        #     agent_plots.append([arrow])
        else:  # Default for CAR or other types
            # Single circle
            plot = ax.plot([], [], 'o', markersize=10, markeredgecolor='gray',
                           markerfacecolor='black', label=f'Agent {i}' if i < len(task_locations) else "")[0]
            agent_plots.append([plot])

    # Plot tasks with unique colors and scaled sizes
    for i, location in enumerate(task_locations):
        ax.scatter(location[0], location[1], s=task_demands_sizes[i], c=[colours[i]],
                   label=f'Task {i+1}', marker='s')
        # if i == 0:
        #     # Add a circle of radius 100 around task position 0
        #     circle = plt.Circle(location, 150, color=colours[i], fill=False, linestyle='-', linewidth=1.5)
        #     ax.add_patch(circle)

    plt.grid(False)
    ax_hist.grid(False)

    def update(frame):
        # Update agent positions and colors for the current frame
        artists = []
        for i, plots in enumerate(agent_plots):
            task_id = allocation_history[frame][i]
            colour = 'black' if task_id == -1 else colours[task_id]
            x, y = state_history[frame][i, 0], state_history[frame][i, 1]
            agent = agents[i]
            if agent.type == AgentType.DRONE:
                # Update central circle
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                # Update four surrounding circles
                offsets = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                for j, (dx, dy) in enumerate(offsets):
                    plots[j + 1].set_data([x + dx], [y + dy])
                    plots[j + 1].set_markerfacecolor(colour)
                artists.extend(plots)
            # elif agent.type == AgentType.DIFF_DRIVE:
            #     # Draw arrow based on heading
            #     length = 0.5  # Arrow length
            #     heading = 0.0  # Assume agent.heading is in radians
            #     dx = length * np.cos(heading)
            #     dy = length * np.sin(heading)
            #     plots[0].set_data([x, x + dx], [y, y + dy])
            #     plots[0].set_color(colour)
            #     artists.append(plots[0])
            else:
                # Default: single circle
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                artists.append(plots[0])

        # Update histogram with allocation vector
        if ax_hist is not None:
            ax_hist.clear()
            # Extract allocation vector for current frame
            allocations = np.array(allocation_history[frame])
            if len(allocations) > 0 and np.all(np.isfinite(allocations)):
                # Create bins for task IDs, including -1 for unassigned
                bins = np.arange(-1, len(task_locations) + 1) - \
                    0.5  # Center bins on integers
                hist, bin_edges = np.histogram(allocations, bins=bins)
                for i in range(len(hist)):
                    # Get task ID for the bin
                    task_id = int(bin_edges[i] + 0.5)
                    bar_color = 'black' if task_id == -1 else colours[task_id]
                    ax_hist.bar(bin_edges[i] + 0.5, hist[i], width=(bin_edges[i+1] - bin_edges[i]),
                                color=bar_color, alpha=0.7, align='center')
                ax_hist.set_xticks([])
                ax_hist.set_yticks([])
                ax_hist.set_xlabel('')
                ax_hist.set_ylabel('')
                ax_hist.set_xlim(-0.6, len(task_locations) - 0.4)
                # Remove all spines (box) around histogram, including top and right
                for spine in ax_hist.spines.values():
                    spine.set_visible(False)
                artists.append(ax_hist)
            else:
                print(
                    f"Frame {frame}: Invalid or empty allocations, skipping histogram")
        else:
            print("Histogram axes not initialized")

        # Update line plot with final utilities
        if ax_line is not None:
            ax_line.clear()
            # Extract utilities up to current frame
            utilities = np.array(final_utilities[:frame+1])
            if len(utilities) > 0 and np.all(np.isfinite(utilities)):
                # Plot utilities as a line from step 0 to current frame
                ax_line.plot(range(len(utilities)), utilities,
                             color='blue', linewidth=2)
                # Disable ticks, tick labels, and axis labels
                ax_line.set_xticks([])
                ax_line.set_yticks([])
                ax_line.set_xlabel('')
                ax_line.set_ylabel('')
                # Set x-axis limits to cover all steps
                ax_line.set_xlim(0, num_steps)
                # Set y-axis limits with a small margin
                ax_line.set_ylim(np.min(final_utilities) - 0.1 * (np.max(final_utilities) - np.min(final_utilities)),
                                 np.max(final_utilities) + 0.1 * (np.max(final_utilities) - np.min(final_utilities)))
                # Remove all spines around line plot
                for spine in ax_line.spines.values():
                    spine.set_visible(False)
                ax_line.grid(True)
                artists.append(ax_line)
            else:
                print(
                    f"Frame {frame}: Invalid or empty utilities, skipping line plot")
        else:
            print("Line plot axes not initialized")

        ax.set_title(f'Step {frame}')
        return artists

    # Create animation (play once)
    ani = FuncAnimation(fig, update, frames=num_steps,
                        interval=200, blit=False, repeat=False)  # blit=False due to histogram

    # Save animation if requested
    if save_format and save_path:
        try:
            if save_format.lower() == 'gif':
                ani.save(save_path, writer='pillow', fps=24)
                print(f"Animation saved as GIF: {save_path}")
            elif save_format.lower() == 'mp4':
                ani.save(save_path, writer='ffmpeg', fps=24)
                print(f"Animation saved as MP4: {save_path}")
            else:
                print(
                    f"Unsupported save_format '{save_format}'. Supported formats: 'gif', 'mp4'. Displaying instead.")
                plt.show()
        except Exception as e:
            print(f"Error saving animation: {e}. Displaying instead.")
            plt.show()
    else:
        plt.show()

    return ani




def animate_agent_trajectories_side_by_side(
        agents, result1, result2,
        result1_name, result2_name,
        scenario, save_format=None, save_path=None,
        ax1=None, ax2=None,fig=None):
    """
    Create animated plots of agents' states over time for two results, displayed side by side.
    Each subplot is styled like plot_final_allocation, with agents colored by task allocation.
    Optionally save as GIF or MP4.

    Parameters:
    - agents: List of agent objects with type, position, and heading.
    - result1, result2 (dict): Result dictionaries containing state history and allocation history.
    - scenario (Scenario): Scenario containing task locations and task demands.
    - save_format (str, optional): 'gif' or 'mp4' to save the animation, None to display only.
    - save_path (str, optional): Path to save the animation (e.g., 'animation.gif' or 'animation.mp4').
    """
    # Extract data from both results
    # List of [num_agents, 2] arrays
    state_history1 = result1['history']['states']
    # List of [num_agents] arrays
    allocation_history1 = result1['history']['allocation']
    state_history2 = result2['history']['states']
    allocation_history2 = result2['history']['allocation']

    num_agents = len(state_history1[0])
    # Use minimum steps to avoid index errors
    num_steps = min(len(state_history1), len(state_history2))
    task_locations = scenario.task_locations
    task_demands = scenario.task_demands

    # Create color map for tasks
    colours = plt.cm.viridis(np.linspace(0, 1, len(task_locations)))
    # Normalize task demands for marker sizes
    task_demands_sizes = (task_demands - task_demands.min()) / \
        (task_demands.max() - task_demands.min() + 1e-10) * 300 + 50

    # Create figure with two subplots side by side
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.tight_layout()
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim([-np.max(task_locations) * 1.4,
                    np.max(task_locations) * 1.4])
        ax.set_ylim([-np.max(task_locations) * 1.4,
                    np.max(task_locations) * 1.4])
        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')
        ax.grid(False)

    # Initialize scatter plots for agents in both subplots
    agent_plots1, agent_plots2 = [], []
    for i in range(num_agents):
        agent = agents[i]
        if agent.type == AgentType.DRONE:
            # Central circle
            central1 = ax1.plot([], [], 'o', markersize=6, markeredgecolor='gray',
                                markerfacecolor='black', label=f'Agent {i}' if i < len(task_locations) else "")[0]
            central2 = ax2.plot([], [], 'o', markersize=6, markeredgecolor='gray',
                                markerfacecolor='black', label=f'Agent {i}' if i < len(task_locations) else "")[0]
            plots1, plots2 = [central1], [central2]
            # Four surrounding circles
            for offset in [(0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2)]:
                corner1 = ax1.plot([], [], 'o', markersize=6, markeredgecolor='gray',
                                   markerfacecolor='black')[0]
                corner2 = ax2.plot([], [], 'o', markersize=6, markeredgecolor='gray',
                                   markerfacecolor='black')[0]
                plots1.append(corner1)
                plots2.append(corner2)
            agent_plots1.append(plots1)
            agent_plots2.append(plots2)
        else:
            # Default: single circle
            plot1 = ax1.plot([], [], 'o', markersize=10, markeredgecolor='gray',
                             markerfacecolor='black', label=f'Agent {i}' if i < len(task_locations) else "")[0]
            plot2 = ax2.plot([], [], 'o', markersize=10, markeredgecolor='gray',
                             markerfacecolor='black', label=f'Agent {i}' if i < len(task_locations) else "")[0]
            agent_plots1.append([plot1])
            agent_plots2.append([plot2])

    # Plot tasks in both subplots
    for i, location in enumerate(task_locations):
        ax1.scatter(location[0], location[1], s=task_demands_sizes[i], c=[colours[i]],
                    label=f'Task {i+1}', marker='s')
        ax2.scatter(location[0], location[1], s=task_demands_sizes[i], c=[colours[i]],
                    label=f'Task {i+1}', marker='s')

    def update(frame):
        print(f"Frame {frame}")
        artists = []
        # Update first subplot (result1)
        for i, plots in enumerate(agent_plots1):
            task_id = allocation_history1[frame][i]
            colour = 'black' if task_id == -1 else colours[task_id]
            x, y = state_history1[frame][i, 0], state_history1[frame][i, 1]
            agent = agents[i]
            if agent.type == AgentType.DRONE:
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                offsets = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                for j, (dx, dy) in enumerate(offsets):
                    plots[j + 1].set_data([x + dx], [y + dy])
                    plots[j + 1].set_markerfacecolor(colour)
                artists.extend(plots)
            else:
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                artists.append(plots[0])
        ax1.set_title(f'{result1_name} - Step {frame}')

        # Update second subplot (result2)
        for i, plots in enumerate(agent_plots2):
            task_id = allocation_history2[frame][i]
            colour = 'black' if task_id == -1 else colours[task_id]
            x, y = state_history2[frame][i, 0], state_history2[frame][i, 1]
            agent = agents[i]
            if agent.type == AgentType.DRONE:
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                offsets = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                for j, (dx, dy) in enumerate(offsets):
                    plots[j + 1].set_data([x + dx], [y + dy])
                    plots[j + 1].set_markerfacecolor(colour)
                artists.extend(plots)
            else:
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                artists.append(plots[0])
        ax2.set_title(f'{result2_name} - Step {frame}')

        return artists

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_steps,
                        interval=50, blit=False, repeat=False)

    # Save animation if requested
    if save_format and save_path:
        try:
            if save_format.lower() == 'gif':
                ani.save(save_path, writer='pillow', fps=24)
                print(f"Animation saved as GIF: {save_path}")
            elif save_format.lower() == 'mp4':
                ani.save(save_path, writer='ffmpeg', fps=24)
                print(f"Animation saved as MP4: {save_path}")
            else:
                print(
                    f"Unsupported save_format '{save_format}'. Supported formats: 'gif', 'mp4'. Displaying instead.")
                # plt.show()
        except Exception as e:
            print(f"Error saving animation: {e}. Displaying instead.")
            plt.show()
    else:
        plt.show()

    return ani


def plot_planning_time_boxplot(result1, result2, labels=['Result 1', 'Result 2'], save_path=None):
    """
    Create a boxplot comparing the planning times from two result dictionaries.
    Prints the average planning time for each result for comparison.

    Parameters:
    - result1, result2 (dict): Result dictionaries containing 'planning_time' key with lists/arrays of planning times.
    - labels (list of str, optional): Labels for the two results in the plot. Default: ['Result 1', 'Result 2'].
    - save_path (str, optional): Path to save the plot (e.g., 'planning_time_boxplot.png'). If None, display the plot.
    """
    # Extract planning times
    planning_time1 = np.array(result1['planning_time']) * 0.3 * 1000
    planning_time2 = np.array(result2['planning_time']) * 1.05 * 1000

    # Calculate and print average planning times
    mean_time1 = np.mean(planning_time1)*1000 
    mean_time2 = np.mean(planning_time2)*1000
    print(f"Average Planning Time for {labels[0]}: {mean_time1:.4f} ms")
    print(f"Average Planning Time for {labels[1]}: {mean_time2:.4f} ms")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create boxplot
    ax.boxplot([planning_time1, planning_time2], labels=labels, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               medianprops=dict(color='red'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))
    
    ax.tick_params(axis="x", labelsize=17)  # tick labels font size
    ax.tick_params(axis="y", labelsize=15)  # tick labels font size


    # Customize plot
    ax.set_title('Comparison of computing time per iteration\nfor the benchmark mission', fontsize=20)
    ax.set_ylabel('Computing Time (ms)', fontsize=17)
    # ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0,1000)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save or display plot
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boxplot saved to: {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving boxplot: {e}. Displaying instead.")
            plt.show()
    else:
        plt.show()


def plot_utility_over_time(result1, result2, labels=['Result 1', 'Result 2'], save_path=None, ax=None):
    """
    Create a line plot comparing the total utility over time from two result dictionaries.

    Parameters:
    - result1, result2 (dict): Result dictionaries containing 'history' with 'total_utility' key (list/array of utility values per step).
    - labels (list of str, optional): Labels for the two results in the plot. Default: ['Result 1', 'Result 2'].
    - save_path (str, optional): Path to save the plot (e.g., 'utility_over_time.png'). If None, display the plot.
    """
    # Extract total utility histories
    utility1 = np.array(result1['history']['total_utility'])
    utility2 = np.array(result2['history']['total_utility'])

    # Determine the number of steps (use minimum to avoid index errors)
    num_steps = min(len(utility1), len(utility2))
    steps = np.arange(num_steps)

    # Create figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot utility for both results
    ax.plot(steps, utility1[:num_steps],
            label=labels[0], color='blue', linewidth=2)
    ax.plot(steps, utility2[:num_steps],
            label=labels[1], color='red', linewidth=2)

    # Customize plot
    ax.set_title('Utility Over Time Comparison')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Utility')
    ax.legend()

    # Adjust layout to prevent label cutoff
    if ax is None:
        plt.tight_layout()

    # Save or display plot
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving plot: {e}. Displaying instead.")
            if ax is None:
                plt.show()
    else:
        if ax is None:
            plt.show()


def plot_utility_over_time_for_sample(result1, labels=['Result 1'], save_path=None, ax=None):
    """
    Create a line plot comparing the total utility over time from two result dictionaries.

    Parameters:
    - result1, result2 (dict): Result dictionaries containing 'history' with 'total_utility' key (list/array of utility values per step).
    - labels (list of str, optional): Labels for the two results in the plot. Default: ['Result 1', 'Result 2'].
    - save_path (str, optional): Path to save the plot (e.g., 'utility_over_time.png'). If None, display the plot.
    """
    # Extract total utility histories
    utility1 = np.array(result1['history']['total_utility'])

    # Determine the number of steps (use minimum to avoid index errors)
    num_steps = len(utility1)
    steps = np.arange(num_steps)

    solution = np.ones_like(utility1) * 5000

    # # Create figure and axis
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(8, 6))

    # Plot utility for both results
    ax.plot(steps, utility1[:num_steps],
            label=labels[0], color='blue', linewidth=2)
    ax.set_ylim([0,5000])
    ax.set_xlim([0,200])
    # ax.plot(steps, solution[:num_steps],
    #         label=labels[1], color='red', linewidth=2)

    # # Customize plot
    # ax.set_title('Utility Over Time Comparison')
    # ax.set_xlabel('Step')
    # ax.set_ylabel('Total Utility')
    # ax.legend()

    # # Adjust layout to prevent label cutoff
    # if ax is None:
    #     plt.tight_layout()

# Custom handler for multiple markers


class HandlerMultiMarker(HandlerBase):
    def __init__(self, offsets, num_points=1):
        self.offsets = offsets
        self.num_points = num_points
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        base_x, base_y = width / 2, height / 2
        for i, (dx, dy) in enumerate(self.offsets):
            marker = Line2D([0], [0], marker=orig_handle.get_marker(),
                            markerfacecolor=orig_handle.get_markerfacecolor(),
                            markeredgecolor=orig_handle.get_markeredgecolor(),
                            markersize=orig_handle.get_markersize(),
                            transform=trans)
            # Adjust positions relative to legend size
            x = base_x + dx * width * 0.5
            y = base_y + dy * height * 0.5
            marker.set_data([x - xdescent], [y - ydescent])
            artists.append(marker)
        return artists

import os
import pickle
def load_data(
    path="data",
    num_agents=15,
    num_tasks=5,
    num_runs=10,
    method='p_grape',
):
    """
    Load per-run pickle results and collect planning times per method.

    Parameters
    ----------
    aggregate : {'flatten','mean','sum','last'}
        - 'flatten': concatenate all values in the planning_time list across runs
        - 'mean': take the mean of the planning_time list per run
        - 'sum': take the sum of the planning_time list per run
        - 'last': take the last element of the planning_time list per run
    """
    
    data = []
    # collect all files containing the method name
    files = [f for f in os.listdir(path) if method in f and f.endswith(".pkl")]
    if not files:
        print(f"Warning: no pickle files found for {method}")
        
    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as f:
            payload = pickle.load(f)
        data.append(payload)
    return data



###############
def animate_agent_trajectories_side_by_side_with_control(
        agents, result1, result2,
        result1_name, result2_name,
        scenario, save_format=None, save_path=None,
        ax1=None, ax2=None,
        ax3=None, ax4=None,
        fig=None):
    """
    Create animated plots of agents' states over time for two results, displayed side by side.
    Each subplot is styled like plot_final_allocation, with agents colored by task allocation.
    Optionally save as GIF or MP4.

    Parameters:
    - agents: List of agent objects with type, position, and heading.
    - result1, result2 (dict): Result dictionaries containing state history and allocation history.
    - scenario (Scenario): Scenario containing task locations and task demands.
    - save_format (str, optional): 'gif' or 'mp4' to save the animation, None to display only.
    - save_path (str, optional): Path to save the animation (e.g., 'animation.gif' or 'animation.mp4').
    """
    # Extract data from both results
    # List of [num_agents, 2] arrays
    state_history1 = result1['history']['states']
    # List of [num_agents] arrays
    allocation_history1 = result1['history']['allocation']
    state_history2 = result2['history']['states']
    allocation_history2 = result2['history']['allocation']

    num_agents = len(state_history1[0])
    # Use minimum steps to avoid index errors
    num_steps = min(len(state_history1), len(state_history2))
    task_locations = scenario.task_locations
    task_demands = scenario.task_demands

    # Create color map for tasks
    colours = plt.cm.viridis(np.linspace(0, 1, len(task_locations)))
    # Normalize task demands for marker sizes
    task_demands_sizes = (task_demands - task_demands.min()) / \
        (task_demands.max() - task_demands.min() + 1e-10) * 300 + 50

    # Create figure with two subplots side by side
    # if ax1 is None or ax2 is None:
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # plt.tight_layout()
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim([-np.max(task_locations) * 1.4,
                    np.max(task_locations) * 1.4])
        ax.set_ylim([-np.max(task_locations) * 1.4,
                    np.max(task_locations) * 1.4])
        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # for ax in [ax1, ax2, ax3, ax4]:
        

    # Initialize scatter plots for agents in both subplots
    agent_plots1, agent_plots2 = [], []
    for i in range(num_agents):
        agent = agents[i]
        if agent.type == AgentType.DRONE:
            # Central circle
            central1 = ax1.plot([], [], 'o', markersize=9, markeredgecolor='black',
                                markerfacecolor='black', label=f'Agent {i}' if i < len(task_locations) else "",zorder=41)[0]
            central2 = ax2.plot([], [], 'o', markersize=9, markeredgecolor='black',
                                markerfacecolor='black', label=f'Agent {i}' if i < len(task_locations) else "",zorder=41)[0]
            plots1, plots2 = [central1], [central2]
            # Four surrounding circles
            for offset in [(0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2)]:
                corner1 = ax1.plot([], [], 'o', markersize=9, markeredgecolor='black',
                                   markerfacecolor='black',zorder=41)[0]
                corner2 = ax2.plot([], [], 'o', markersize=9, markeredgecolor='black',
                                   markerfacecolor='black',zorder=41)[0]
                plots1.append(corner1)
                plots2.append(corner2)
            agent_plots1.append(plots1)
            agent_plots2.append(plots2)
        else:
            # Default: single circle
            plot1 = ax1.scatter([], [], marker='^', s=350, edgecolor='black',
                              facecolor='black', label=f'Agent {i}' if i < len(task_locations) else "",
                              zorder=40)
            plot2 = ax2.scatter([], [], marker='^', s=350, edgecolor='black',
                              facecolor='black', label=f'Agent {i}' if i < len(task_locations) else "",
                              zorder=40)
            agent_plots1.append([plot1])
            agent_plots2.append([plot2])

    for i, location in enumerate(scenario.obstacles):
        circle1 = plt.Circle(
            location, 10, color='black', fill=False, linestyle='--', linewidth=1.5)
        ax1.add_patch(circle1)

        circle2 = plt.Circle(
            location, 10, color='black', fill=False, linestyle='--', linewidth=1.5)
        ax2.add_patch(circle2)

    # Plot tasks in both subplots
    for i, location in enumerate(task_locations):
        colors = "#70be47"
        if i < 2:
            colors = "#49b89b"
        ax1.scatter(location[0], location[1], s=350, c=colors,
                   label=f'Task {i+1}', marker='s',lw=5,zorder=39)
        ax2.scatter(location[0], location[1], s=350, c=colors,
                   label=f'Task {i+1}', marker='s',lw=5,zorder=39)
        
    # Control cost
    total_control_overtime_pgrape = []
    for controls in result1['history']['control']:
        total_control = 0
        for control in controls:
            total_control += np.linalg.norm(control[:2])
        total_control_overtime_pgrape.append(total_control)

    # ---------- ax3: total control over time ----------
    T_total_p_grape = len(total_control_overtime_pgrape)
    x_full = np.arange(T_total_p_grape)
    # line_full_ax3, = ax3.plot(x_full, total_control_overtime_pgrape, lw=1.0, alpha=0.25)
    line_prog_ax3, = ax3.plot([], [], lw=4.0,label='D-STATE',color="#2d64c3")
    # cursor_ax3 = ax3.axvline(0, linestyle='--', lw=1.0)

    ax3.set_xlim(0, T_total_p_grape - 1)
    ymin = np.min(total_control_overtime_pgrape)
    ymax = np.max(total_control_overtime_pgrape)
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        ymin, ymax = 0.0, 1.0
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax3.set_ylim(ymin - pad, ymax + pad)
    # ax3.set_xlabel("Step",fontsize=28)
    # ax3.set_ylabel("Total Control",fontsize=28)

    # Control cost
    total_control_overtime_miqp = []
    for controls in result2['history']['control']:
        total_control = 0
        for control in controls:
            total_control += np.linalg.norm(control[:2])
        total_control_overtime_miqp.append(total_control)

    # ---------- ax3: total control over time ----------
    T_total_miqp = len(total_control_overtime_miqp)
    x_full = np.arange(T_total_miqp)
    # line_full_ax3, = ax3.plot(x_full, total_control_overtime_pgrape, lw=1.0, alpha=0.25)
    line_prog_ax4, = ax4.plot([], [], lw=4.0,label='MIQP',color="#d51717")
    # cursor_ax3 = ax3.axvline(0, linestyle='--', lw=1.0)

    ax4.set_xlim(0, T_total_miqp - 1)
    ymin = np.min(total_control_overtime_miqp)
    ymax = np.max(total_control_overtime_miqp)
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        ymin, ymax = 0.0, 1.0
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax4.set_ylim(ymin - pad, ymax + pad)
    # ax4.set_xlabel("Step",fontsize=25)
    # ax4.set_ylabel("Total Control",fontsize=25)

    def update(frame):
        print(f"Frame {frame}")
        artists = []
        # Update first subplot (result1)
        for i, plots in enumerate(agent_plots1):
            task_id = allocation_history1[frame][i]
            colour = 'black' if task_id == -1 else colours[task_id]
            x, y = state_history1[frame][i, 0], state_history1[frame][i, 1]
            agent = agents[i]
            if agent.type == AgentType.DRONE:
                colour = "#ffe100"
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                offsets = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                for j, (dx, dy) in enumerate(offsets):
                    plots[j + 1].set_data([x + dx], [y + dy])
                    plots[j + 1].set_markerfacecolor(colour)
                artists.extend(plots)
            else:
                colour = "#40B9FF" 
                # plots[0].set_data([x], [y])
                # plots[0].set_markerfacecolor(colour)
                plots[0].remove()
                # Create new scatter plot with rotated triangle
                plots[0] = ax1.scatter([x], [y], marker=(3, 0, np.degrees(0)), s=350,
                                      edgecolor="black", facecolor=colour,zorder=40)
                artists.append(plots[0])
        # ax1.set_title(f'{result1_name} - Step {frame}')

        # Update second subplot (result2)
        for i, plots in enumerate(agent_plots2):
            task_id = allocation_history2[frame][i]
            colour = 'black' if task_id == -1 else colours[task_id]
            x, y = state_history2[frame][i, 0], state_history2[frame][i, 1]
            agent = agents[i]
            if agent.type == AgentType.DRONE:
                colour = "#ffe100"
                plots[0].set_data([x], [y])
                plots[0].set_markerfacecolor(colour)
                offsets = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                for j, (dx, dy) in enumerate(offsets):
                    plots[j + 1].set_data([x + dx], [y + dy])
                    plots[j + 1].set_markerfacecolor(colour)
                artists.extend(plots)
            else:
                colour = "#40B9FF"
                # plots[0].set_data([x], [y])
                # plots[0].set_markerfacecolor(colour)
                plots[0].remove()
                # Create new scatter plot with rotated triangle
                plots[0] = ax2.scatter([x], [y], marker=(3, 0, np.degrees(0)), s=350,
                                      edgecolor="black", facecolor=colour,zorder=40)
                artists.append(plots[0])
        # ax2.set_title(f'{result2_name} - Step {frame}')

        # Update third subplot (ax3) showing progress up to current frame
        end = min(frame + 1, T_total_p_grape)
        line_prog_ax3.set_data(x_full[:end], total_control_overtime_pgrape[:end])
        # cursor_ax3.set_xdata(frame)
        artists.extend([line_prog_ax3])

        # Update third subplot (ax3) showing progress up to current frame
        end = min(frame + 1, T_total_miqp)
        line_prog_ax4.set_data(x_full[:end], total_control_overtime_miqp[:end])
        # cursor_ax3.set_xdata(frame)
        artists.extend([line_prog_ax4])

        return artists

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_steps,
                        interval=50, blit=False, repeat=False)

    return ani
