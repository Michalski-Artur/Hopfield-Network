from typing import Callable, Optional
from math_utils import MathUtils
import numpy as np

LearningRuleType = Callable[[np.ndarray], np.ndarray]


class LearningRules:

    @staticmethod
    def hebb(patterns: np.ndarray):
        weights = (1 / patterns.shape[0]) * patterns.T @ patterns
        np.fill_diagonal(weights, 0)
        return weights

    @staticmethod
    def oja(patterns: np.ndarray):
        # weights = np.zeros((patterns.shape[1], patterns.shape[1])) # Maybe Hebb for initialization?
        weights = LearningRules.hebb(patterns) # Maybe Hebb for initialization?
        y_array = np.zeros((patterns.shape[1], patterns.shape[1]))
        max_iter = 10
        print(weights)
        for iter in range(max_iter):
            delta_weights = np.zeros((patterns.shape[1], patterns.shape[1]))
            y_array = weights @ patterns.T
            for i in range(patterns.shape[1]):
                for j in range(patterns.shape[1]):
                    if i == j:
                        continue
                    delta_weights[i, j] = y_array[i, :] @ (patterns[:, j] - weights[i, j] * y_array[i, :]).T
            new_weights = MathUtils.normalize(weights + MathUtils.normalize(delta_weights, -1, 1), -1, 1)
            np.fill_diagonal(new_weights, 0)
            print(f'Iteration {iter}: \n', new_weights)
            diff = np.linalg.norm(new_weights - weights)
            weights = new_weights
            if diff < 1e-5:
                print (f'Converged after {iter} iterations')
                break
        print(f'Final weights:', weights)
        return weights
