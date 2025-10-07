#!/usr/bin/env python3
import numpy as np
from model import Model


class SingleIntegrator(Model):
    def __init__(self):
        self._control_dim = 3 # Including the slack
        self._state_dim = 2
        
    """Single integrator model: dx/dt = u"""
    def f(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """State transition: x_{t+1} = x_t + u * dt"""
        return x + u * dt
    
    def df(self,x: np.ndarray, u: np.ndarray, dt: float = 0.1):
        """Jacobian: A = I, B = dt * I"""
        n = x.shape[0]
        A = np.eye(n)
        B = dt * np.eye(n)
        return A, B

    def dx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time derivative: dx/dt = u"""
        return u
    
    def init(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(2), np.zeros(2)  # Initial state and control input
