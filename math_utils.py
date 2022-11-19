import numpy as np

class MathUtils:
    @staticmethod
    def normalize(data, range_min = -1, range_max = 1):
        return range_min + (range_max - range_min) * (data - np.min(data)) / (np.max(data) - np.min(data))