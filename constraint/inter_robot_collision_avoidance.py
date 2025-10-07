#!/usr/bin/env python3
import numpy as np
from model import Model
from constraint import Constraint

class InterRobotCollisionAvoidance(Constraint):
    def __init__(self):
        self._state_dim = 3  # 3D position components
        self._control_dim = 3  # Control inputs plus slack variable

    def _h(self, x_i: np.ndarray, x_j: np.ndarray, r: float) -> float:
        """Barrier function for 3D collision avoidance.

        Args:
            x_i: State vector [x_i1, x_i2, x_i3, ...] (position + other states)
            x_j: Obstacle position [x_j1(t), x_j2(t), x_j3(t)]
            r: Minimum safe distance (obstacle radius)

        Returns:
            Barrier function value h(x_i, t) = ||x_i - x_j(t)||^2 - r^2
        """
        diff = x_i - x_j
        return np.dot(diff, diff) - r**2

    def _dh_dx(self, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
        """Partial derivative of the barrier function with respect to x_i.

        Args:
            x_i: State vector [x_i1, x_i2, x_i3, ...]
            x_j: Obstacle position [x_j1(t), x_j2(t), x_j3(t)]

        Returns:
            Gradient dh/dx_i as a vector
        """
        diff = x_i - x_j
        grad = np.zeros(x_i.shape[0])
        grad = 2.0 * diff  # [2(x_i1 - x_j1), 2(x_i2 - x_j2), 2(x_i3 - x_j3)]
        return grad

    def _dh_dt(self, x_i: np.ndarray, x_j: np.ndarray, v_j: np.ndarray) -> float:
        """Partial derivative of the barrier function with respect to time.

        Args:
            x_i: State vector [x_i1, x_i2, x_i3, ...]
            x_j: Obstacle position [x_j1(t), x_j2(t), x_j3(t)]
            v_j: Obstacle velocity [v_j1(t), v_j2(t), v_j3(t)]

        Returns:
            Partial derivative dh/dt (scalar)
        """
        diff = x_i - x_j
        return -2.0 * np.dot(diff, v_j)

    def evaluate(self,
                 model: Model,  # Type hint for custom model class
                 x_i: np.ndarray,
                 u: np.ndarray,
                 x_j: np.ndarray,
                 v_j: np.ndarray,
                 gamma: float,
                 r: float,
                 t: float) -> float:
        """Evaluate the time-varying CBF constraint: dh/dt + dot(∇h, dx_i/dt) + gamma * h + epsilon >= 0.

        Args:
            model: System dynamics model with dx method
            x_i: State vector [x_i1, x_i2, x_i3, ...]
            u: Control input [u_1, u_2, ..., epsilon] (slack variable last)
            x_j: Obstacle position [x_j1(t), x_j2(t), x_j3(t)]
            v_j: Obstacle velocity [v_j1(t), v_j2(t), v_j3(t)]
            gamma: CBF parameter
            r: Minimum safe distance
            t: Current time (scalar)

        Returns:
            CBF constraint value (scalar, should be non-negative)
        """
        dx_i = model.dx(x_i, u[:-1])  # State derivative without slack variable
        dh_dx = self._dh_dx(x_i, x_j)
        dh_dt = self._dh_dt(x_i, x_j, v_j)
        h_val = self._h(x_i, x_j, r)
        return dh_dt + np.dot(dh_dx, dx_i) + gamma * h_val