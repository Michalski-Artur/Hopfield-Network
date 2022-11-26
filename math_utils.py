import numpy as np

# This class is currently not in use but those methods can be used in the future
class MathUtils:
    @staticmethod
    def normalize(data: np.ndarray, range_min = -1, range_max = 1) -> np.ndarray:
        if np.max(data) == np.min(data):
            return np.zeros(data.shape)
        return range_min + (range_max - range_min) * (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def normalize_max_abs(data: np.ndarray) -> np.ndarray:
        if np.max(np.abs(data)) == 0:
            return data
        return data / np.max(np.abs(data))