import csv
import warnings
from bisect import bisect

import torch


class CSVDict:
    """Convert the CSV file into a linearly interpolatable dictionary

    Values of the dictionary can be normalized or clamped with the standard deviation.
    If a key outside the key range of the CSV is entered, a warning will be issued,
        and the end value of that side will be returned.
    """

    def __init__(self, filename, key_index=0, value_index=1, is_norm=False, clamp=0):
        self.keys = []
        self.values = []
        with open(filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                self.keys.append(float(row[key_index]))
                self.values.append(float(row[value_index]))

        if is_norm:
            tensor = torch.tensor(self.values)
            self.std = tensor.std().item()
            if clamp:
                tensor.clamp_(-clamp * self.std, clamp * self.std)
            # I don't think it makes sense to shift the tensor by mean so that mean=0 ???
            tensor /= self.std
            self.values = tensor.tolist()

    def __getitem__(self, key):
        if key < self.keys[0] or key > self.keys[-1]:
            warnings.warn('The key entered is not in the CSV')
            return self.values[0] if key < self.keys[0] else self.values[-1]

        index = bisect(self.keys, key)
        k = (key - self.keys[index - 1]) / (self.keys[index] - self.keys[index - 1])
        return k * self.values[index] + (1 - k) * self.values[index - 1]
