#!/usr/bin/env python3
import numpy as np
from model import Model


class DoubleIntegrator(Model):
    """Double integrator model: d^2x/dt^2 = u"""
    def f(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """State transition: [p, v]_{t+1} = [p + v*dt + 0.5*u*dt^2, v + u*dt]
        
        State x = [position, velocity], control u = acceleration
        """
        n = x.shape[0] // 2
        p = x[:n]  # position
        v = x[n:]  # velocity
        p_new = p + v * dt + 0.5 * u * dt**2
        v_new = v + u * dt
        return np.hstack((p_new, v_new))

    def df(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Jacobian: A = [[I, dt*I], [0, I]], B = [[0.5*dt^2*I], [dt*I]]"""
        n = x.shape[0] // 2
        A = np.block([
            [np.eye(n), dt * np.eye(n)],
            [np.zeros((n, n)), np.eye(n)]
        ])
        B = np.block([
            [0.5 * dt**2 * np.eye(n)],
            [dt * np.eye(n)]
        ])
        return A, B

    def dx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time derivative: dx/dt = [v, u]
        
        State x = [position, velocity], control u = acceleration
        """
        n = x.shape[0] // 2
        v = x[n:]  # velocity
        return np.hstack((v, u))
    
    def init(self) -> tuple[np.ndarray, np.ndarray]:
        """Initial state and control input."""
        return np.zeros(4), np.zeros(2)
