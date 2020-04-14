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

    @staticmethod
    def _compute_euclidean_distances(arr1, arr2):
        """Compute Euclidean distances between two arrays.

        Parameters
        ----------
        arr1, arr2  :   numpy.ndarray
            two arrays to compute the distances for; shape: (n_samples, n_dimensions)

        Returns
        -------
        a np.ndarray of Euclidean distances; shape: (arr1.shape[0], arr2.shape[0])
        """

        def expand(a1, a2, axis=0):
            return np.stack(a2.shape[0]*(a1,), axis=axis)

        diff = expand(arr1, arr2, 0) - expand(arr2, arr1, 1)

        return (diff**2).sum(axis=-1)**0.5




