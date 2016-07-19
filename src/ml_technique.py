from abc import ABC, abstractmethod


class MLTechnique(ABC):
    """An abstract class that encapsulates the concept of a machine learning technique"""
    @abstractmethod
    def train_and_evaluate(self, defaulter_set):
        pass
