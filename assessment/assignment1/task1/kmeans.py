import numpy as np

import misc


class KMeans(object):
    @misc.check_types
    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"'k' must be positive (got {k})")
        self._k = k

        self._data = None

    @property
    def k(self):
        return self._k

    @misc.check_types
    def fit(self, data: np.ndarray):
        self._data = data

