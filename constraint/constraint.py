#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np
from model import Model


class Constraint(ABC):
    @abstractmethod
    def evaluate(self,
                 model: Model,
                 x: np.ndarray,
                 u: np.ndarray,
                 *args) -> float:
        """Evaluate the constraint function.

        Args:
            x: State vector
            u: Control input vector
            *args: Additional arguments specific to the constraint

        Returns:
            Constraint value (scalar)
        """
        pass
