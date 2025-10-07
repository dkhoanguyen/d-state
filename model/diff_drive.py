#!/usr/bin/env python3
import numpy as np
from model import Model

class DifferentialDrive(Model):
    """Differential drive robot model"""
    def f(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """State transition for differential drive robot.
        
        State x = [x, y, theta], control u = [v, w] (linear and angular velocity)
        """
        x, y, theta = x
        v, w = u
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + w * dt
        return np.array([x_new, y_new, theta_new])

    def df(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Jacobian for differential drive robot"""
        _, _, theta = x
        v, _ = u
        A = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1, v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        B = np.array([
            [np.cos(theta) * dt, 0],
            [np.sin(theta) * dt, 0],
            [0, dt]
        ])
        return A, B

    def dx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time derivative: dx/dt = [v*cos(theta), v*sin(theta), w]
        
        State x = [x, y, theta], control u = [v, w]
        """
        _, _, theta = x
        v, w = u
        return np.array([v * np.cos(theta), v * np.sin(theta), w])
    
    def init(self) -> tuple[np.ndarray, np.ndarray]:
        """Initial state and control input."""
        return np.zeros(3), np.zeros(2)