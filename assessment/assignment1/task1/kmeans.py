import numpy as np


def kmeans(data: np.ndarray, k: int):
    _check_arguments(data, k)


def _check_arguments(data, k):
    _check_type('data', data, np.ndarray)
    _check_type('k', k, int)

    if k < 1:
        raise ValueError(f"'k' must be positive (got {k})")


def _check_type(param_name, param, req_type):
    if not isinstance(param, req_type):
        raise TypeError(f"parameter '{param_name}': expected {req_type}, got {type(param)}")
