from typing import Callable, Optional
import numpy as np

LearningRuleType = Callable[[np.array], np.ndarray]


class LearningRules:

    @staticmethod
    def hebb(patterns: np.array):
        return (1 / patterns.shape[0]) * patterns.T @ patterns

    @staticmethod
    def oja(patterns: np.array):
        raise NotImplementedError

