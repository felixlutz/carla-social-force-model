"""Field of view computation."""

import numpy as np


class FieldOfView:
    """Compute field of view prefactors.

    The field of view angle twophi is given in degrees.
    out_of_view_factor is C in the paper.
    """
    def __init__(self, two_phi=200.0, out_of_view_factor=0.5):
        self.cos_phi = np.cos(two_phi / 2.0 / 180.0 * np.pi)
        self.out_of_view_factor = out_of_view_factor

    def __call__(self, e, f):
        """Weighting factor for field of view.

        e is rank 2 and normalized in the last index.
        f is a rank 3 tensor.
        """

        # calculate angles between desired direction e and social forces f via scalar product
        angles = np.einsum('aj,abj->ab', e, f)

        # check if calculated angles are within field of view and apply weighting factor if not
        in_sight = angles > np.linalg.norm(f, axis=-1) * self.cos_phi
        out = self.out_of_view_factor * np.ones_like(in_sight)
        out[in_sight] = 1.0
        np.fill_diagonal(out, 0.0)
        return out
