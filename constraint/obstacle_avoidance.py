#!/usr/bin/env python3
import numpy as np
from typing import Tuple
from model import Model
from constraint import Constraint


class ObstacleAvoidanceConstraint(Constraint):
    def __init__(self):
        self._state_dim = 2
        self._control_dim = 2

    def _h(self, x: np.ndarray, o: np.ndarray, r: float) -> float:
        """Time-varying barrier function: h(x, t) = (x - o(t))^T (x - o(t)) - r^2.

        Args:
            x: State vector [x, y, theta] (or [x, y] for single integrator)
            o: Time-varying obstacle position [o_x(t), o_y(t)] at time t
            r: Obstacle radius (scalar)

        Returns:
            Barrier function value (scalar)
        """
        diff = x[:2] - o
        return np.dot(diff, diff) - r**2

    def _dh_dx(self, x: np.ndarray, o: np.ndarray) -> np.ndarray:
        """Gradient of the barrier function with respect to x.

        Args:
            x: State vector [x, y, theta] (or [x, y] for single integrator)
            o: Time-varying obstacle position [o_x(t), o_y(t)]

        Returns:
            Gradient vector [dh/dx, dh/dy, dh/dtheta] (or [dh/dx, dh/dy])
        """
        diff = x[:2] - o
        return np.array([2.0 * diff[0], 2.0 * diff[1]] + [0.0] * (x.shape[0] - 2))

    def _dh_dt(self, x: np.ndarray, o: np.ndarray, do_dt: np.ndarray) -> float:
        """Partial derivative of the barrier function with respect to time.

        Args:
            x: State vector [x, y, theta] (or [x, y] for single integrator)
            o: Time-varying obstacle position [o_x(t), o_y(t)]
            do_dt: Time derivative of the obstacle position [do_x(t)/dt, do_y(t)/dt]

        Returns:
            Partial derivative dh/dt (scalar)
        """
        diff = x[:2] - o
        return -2.0 * np.dot(diff, do_dt)

    def evaluate(self,
                 model: Model,
                 x: np.ndarray,
                 u: np.ndarray,
                 o: np.ndarray,
                 do_dt: np.ndarray,
                 gamma: float,
                 r: float,
                 t: float) -> float:
        """Evaluate the time-varying CBF constraint: dh/dt + dot(∇h, dx/dt) + gamma * h + epsilon >= 0.

        Args:
            x: State vector [x, y, theta] (or other model-specific state)
            u: Control input [v, omega, epsilon] (or model-specific input)
            o: Time-varying obstacle position [o_x(t), o_y(t)]
            do_dt: Time derivative of the obstacle position [do_x(t)/dt, do_y(t)/dt]
            gamma: CBF parameter
            r: Obstacle radius
            t: Current time (scalar)

        Returns:
            CBF constraint value (scalar, should be non-negative)
        """

        dx = model.dx(x, u[:-1])
        dh_dx = self._dh_dx(x, o)
        dh_dt = self._dh_dt(x, o, do_dt)
        h_val = self._h(x, o, r)
        epsilon = u[-1]  # Assume last element of u is the slack variable
        return dh_dt + np.dot(dh_dx, dx) + gamma * h_val + epsilon