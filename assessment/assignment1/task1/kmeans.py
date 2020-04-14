import numpy as np

import misc


class KMeans(object):
    @misc.check_types
    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"'k' must be positive (got {k})")
        self._k = k

        self._data = None
        self._labels = None
        self._centroids = None

    @property
    def k(self):
        return self._k

    @property
    def labels(self):
        return self._labels

    @property
    def centroids(self):
        return self._centroids

    def _pick_centroids(self, n_centroids=None):
        n_centroids = n_centroids or self._k
        return self._data[np.random.choice(np.arange(self._data.shape[0]), n_centroids)]

    @misc.check_types
    def fit(self, data: np.ndarray):
        self._data = data
        self._centroids = self._pick_centroids()

