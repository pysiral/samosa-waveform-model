# -*- coding: utf-8 -*-

"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from dataclasses import dataclass


@dataclass
class FitOptions:
    method: str = 'trf'  # acronym of the minimization solver, see scipy.optimize.least_squares for details
    ftol: float = 1e-2  # exit tolerance on f
    gtol: float = 1e-2  # exit tolerance on gradient norm of f
    xtol: float = 2 * 1e-3  # exit tolerance on x
    diff_step: int = None  # relative step size for the finite difference approximation of the Jacobian
    max_nfev: int = None  # maximum number of function evaluations
    loss: str = 'linear'  # loss function , see scipy.optimize.least_squares for details
