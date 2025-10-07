#!/usr/bin/env python3
import numpy as np
from typing import Tuple
from objective import Objective
from model import Model


class ReachGoalObjective(Objective):
    def __init__(self, r: float = 1.0):
        self._state_dim = 2
        self._control_dim = 2

    def evaluate(self,
                 model: Model,
                 x: np.ndarray,
                 u: np.ndarray,
                 x_g: np.ndarray,
                 dx_g_dt: np.ndarray,
                 gamma: float,
                 r: float) -> float:
        """Evaluate the CBF constraint: dh/dt + dot(∇h, dx/dt) + gamma * h + epsilon >= 0.

        Args:
            x: State vector [x, y, theta] (or other model-specific state)
            u: Control input [v, omega, epsilon] (or model-specific input)
            x_g: Time-varying goal position [x_g(t), y_g(t)]
            dx_g_dt: Time derivative of the goal position [dx_g(t)/dt, dy_g(t)/dt]
            t: Current time (scalar)

        Returns:
            CBF constraint value (scalar, should be non-negative)
        """
        dx = model.dx(x, u[:-1])
        dh_dx = self._dh_dx(x, x_g)
        dh_dt = self._dh_dt(x, x_g, dx_g_dt)
        h_val = self._h(x, x_g, r)
        epsilon = u[-1]  # Slack variable
        return dh_dt + np.dot(dh_dx, dx) + gamma * h_val + epsilon
    
    def progress(self,
                 model: Model,
                 x: np.ndarray,
                 u: np.ndarray,
                 x_g: np.ndarray,
                 dx_g_dt: np.ndarray,
                 gamma: float,
                 r: float) -> float:
        dx = model.dx(x, u[:-1])
        dh_dx = self._dh_dx(x, x_g)
        dh_dt = self._dh_dt(x, x_g, dx_g_dt)
        h_val = self._h(x, x_g, r)
        # Progress is the CBF without any slack variable
        return dh_dt + np.dot(dh_dx, dx) + gamma * h_val

    def _h(self, x: np.ndarray, x_g: np.ndarray, r: float) -> float:
        """Barrier function: h(x, t) = ||x - x_g(t)||^2 - r^2.

        Args:
            x: State vector [x, y, theta] (or [x, y] for single integrator)
            x_g: Time-varying goal position [x_g(t), y_g(t)]

        Returns:
            Barrier function value (scalar)
        """
        diff = x[:2] - x_g
        return np.dot(diff, diff) - r**2

    def _dh_dx(self, x: np.ndarray, x_g: np.ndarray) -> np.ndarray:
        """Gradient of the barrier function with respect to x.

        Args:
            x: State vector [x, y, theta] (or [x, y] for single integrator)
            x_g: Time-varying goal position [x_g(t), y_g(t)]

        Returns:
            Gradient vector [dh/dx, dh/dy, dh/dtheta] (or [dh/dx, dh/dy])
        """
        diff = x[:2] - x_g
        return np.array([2.0 * diff[0], 2.0 * diff[1]] + [0.0] * (x.shape[0] - 2))

    def _dh_dt(self, x: np.ndarray, x_g: np.ndarray, dx_g_dt: np.ndarray) -> float:
        """Partial derivative of the barrier function with respect to time.

        Args:
            x: State vector [x, y, theta] (or [x, y] for single integrator)
            x_g: Time-varying goal position [x_g(t), y_g(t)]
            dx_g_dt: Time derivative of the goal position [dx_g(t)/dt, dy_g(t)/dt]

        Returns:
            Partial derivative dh/dt (scalar)
        """
        diff = x[:2] - x_g
        return 2.0 * np.dot(diff, dx_g_dt)