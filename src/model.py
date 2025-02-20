from typing import Callable, Union

import numpy as np
from scipy.signal import cont2discrete


class LinearTimeInvariantModel:
    """
    A linear time-invariant state-space model of the form
        x' = Ax + Bu,
        y  = Cx + Du.
    The matrices A, B, C, and D are constants or functions of the parameters `theta`.
    """

    def __init__(
        self,
        A: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0,
        B: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0,
        C: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0,
        D: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def discretize(self, dt, theta=None):
        A = self.A(theta) if callable(self.A) else self.A
        B = self.B(theta) if callable(self.B) else self.B
        C = self.C(theta) if callable(self.C) else self.C
        D = self.D(theta) if callable(self.D) else self.D

        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method="impulse")
        return Ad, Bd, Cd, Dd
