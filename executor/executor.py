#!/usr/bin/env python3
from abc import ABC, abstractmethod

class Executor(ABC):
    @abstractmethod
    def execute(self, *arg, **kwargs):
        pass