"""Utility functions to process state."""

import numpy as np


def desired_directions(state):
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state['dest'] - state['loc']
    norm_factors = np.linalg.norm(destination_vectors, axis=1)
    directions = destination_vectors / np.expand_dims(norm_factors, 1)
    directions[norm_factors == 0] = [0, 0, 0]
    return directions


def speeds(state):
    """Return the speeds corresponding to a given state."""
    velocity = state['vel']
    return np.linalg.norm(velocity, axis=1)
