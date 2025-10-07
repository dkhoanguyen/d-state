#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Model(ABC):
    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """State transition function.

        Args:
            x: State vector
            u: Control input vector
            dt: Time step (seconds)

        Returns:
            Next state vector
        """
        pass

    @abstractmethod
    def df(self,x: np.ndarray, u: np.ndarray, dt: float = 0.1):
        """
        """

    @abstractmethod
    def df(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Jacobian of the state transition function.

        Args:
            x: State vector
            u: Control input vector
            dt: Time step (seconds)

        Returns:
            Tuple of (A, B) matrices where A is df/dx and B is df/du
        """
        pass

    @abstractmethod
    def dx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time state derivative (dx/dt).

        Args:
            x: State vector
            u: Control input vector

        Returns:
            State derivative vector
        """
        pass

    @abstractmethod
    def init(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initial state and control input.

        Returns:
            Tuple of initial state vector and control input vector
        """
        pass
