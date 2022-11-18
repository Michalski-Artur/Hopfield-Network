import csv
import numpy as np


class DataReader:

    @staticmethod
    def read_data(path_to_file):
        with open(path_to_file, newline='') as csvfile:
            data = np.array(list(csv.reader(csvfile)), dtype=float)
        return data

