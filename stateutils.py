"""Utility functions to process state."""
from typing import Tuple

import numpy as np


def desired_directions(state) -> np.ndarray:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state['dest'] - state['loc']
    norm_factors = np.linalg.norm(destination_vectors, axis=1)
    directions = destination_vectors / np.expand_dims(norm_factors, 1)
    directions[norm_factors == 0] = [0, 0, 0]
    return directions


def speeds(state) -> np.ndarray:
    """Return the speeds corresponding to a given state."""
    velocity = state['vel']
    return np.linalg.norm(velocity, axis=1)


def all_diffs(array, remove_diagonal=True, keep_dims=True) -> np.ndarray:
    """
    Calculate the differences of every element in the array with all other elements using broadcasting.
    :param keep_dims:
    :param array: input array
    :param remove_diagonal: bool determining if the diagonal values of the diff matrix(diff of an element with itself)
                            will be removed
    :return: diff matrix
    """
    diff_matrix = np.expand_dims(array, 1) - np.expand_dims(array, 0)

    if remove_diagonal:
        diff_matrix = diff_matrix[~np.eye(diff_matrix.shape[0], dtype=bool), :]

    if keep_dims:
        diff_matrix = diff_matrix.reshape(array.shape[0], -1, array.shape[1])

    return diff_matrix


def normalize(array, axis=-1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize vectors in array along given axis.
    :param array: input array
    :param axis: axis to normalize along
    :return: normalized vectors and corresponding norm factors
    """
    norm_factors = np.linalg.norm(array, axis=axis)

    # create a modified version of the norm factors array to avoid division by zero
    norm_factors_div = np.copy(norm_factors)
    norm_factors_div[norm_factors_div == 0.0] = 1.0
    normalized = array / np.expand_dims(norm_factors_div, axis)

    return normalized, norm_factors


def angle_diff_2d(vecs1, vecs2) -> np.ndarray:
    """
    Calculate angle diffs between two arrays of vectors (only 2D)
    :param vecs1:
    :param vecs2:
    :return:
    """
    # get vector angles with arctan2(y, x)
    angles1 = np.arctan2(vecs1[..., 1], vecs1[..., 0])
    angles2 = np.arctan2(vecs2[..., 1], vecs2[..., 0])

    # compute angle diffs
    angle_diffs = angles1 - angles2

    # normalize angles
    angle_diffs[angle_diffs > np.pi] -= 2 * np.pi
    angle_diffs[angle_diffs < -np.pi] += 2 * np.pi

    return angle_diffs


def minmax(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = np.min(vecs['loc'][:, 0])
    y_min = np.min(vecs['loc'][:, 1])
    x_max = np.max(vecs['loc'][:, 0])
    y_max = np.max(vecs['loc'][:, 1])
    return x_min, y_min, x_max, y_max
