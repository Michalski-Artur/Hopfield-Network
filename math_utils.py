import numpy as np

class MathUtils:
    @staticmethod
    def normalize(data: np.ndarray, range_min = -1, range_max = 1) -> np.ndarray:
        return range_min + (range_max - range_min) * (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def normalize_max_abs(data: np.ndarray) -> np.ndarray:
        return data / np.max(np.abs(data))