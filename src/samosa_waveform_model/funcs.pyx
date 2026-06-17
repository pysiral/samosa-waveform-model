import numpy as np
cimport numpy as np

import cython
cimport cython

from samosa_waveform_model.dataclasses import CONSTANTS

# Define the types for the numpy arrays
ctypedef np.float64_t DTYPE_t
ctypedef np.npy_bool DTYPE_BOOL_t
ctypedef np.int32_t DTYPE_LONG_t
ctypedef np.int64_t DTYPE_LONGLONG_t


@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
def compute_gamma0(double alpha_y, double yp, double alpha_x, double nu, double alt, np.ndarray[DTYPE_t, ndim=1] xl, double xp, np.ndarray[DTYPE_t, ndim=1] yk):
    cdef np.ndarray[DTYPE_t, ndim=2] xl_ = xl[None, :]
    cdef np.ndarray[DTYPE_t, ndim=2] yk_ = yk[:, None]
    cdef double alt2 = alt ** 2
    return np.exp(
        -alpha_y * yp ** 2 - alpha_x * (xl_ - xp) ** 2 - xl_ ** 2 * nu / alt2 -
        (alpha_y + nu / alt2) * yk_ ** 2) * np.cosh(2. * alpha_y * yp * yk_)

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
def compute_t_kappa(np.ndarray[DTYPE_t, ndim=2] z, np.ndarray[DTYPE_t, ndim=1] dk, double nu, double alt, double alpha_y, double yp, double ly):
    cdef np.ndarray[DTYPE_t, ndim=2] t_kappa = np.zeros(np.shape(z), dtype=np.float64)
    cdef np.ndarray[DTYPE_BOOL_t, ndim=1] dk_positive = dk > 0
    dk_positive_idx = np.where(dk_positive)
    dk_negative_idx = np.where(np.logical_not(dk_positive))
    dk_positive_sqrt = np.sqrt(dk[dk_positive_idx])

    t_kappa[dk_positive_idx, :] = (
            (1. + nu / ((alt ** 2) * alpha_y)) - yp / (ly * dk_positive_sqrt) * np.tanh(
             2. * alpha_y * yp * ly * dk_positive_sqrt)[None, :]).T
    t_kappa[dk_negative_idx, :] = (1. + nu / ((alt ** 2) * alpha_y)) - 2. * alpha_y * yp ** 2
    return t_kappa

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
def compute_f0(np.ndarray[DTYPE_t, ndim=2] csi, double csi_min_f0, double csi_max_f0, np.ndarray[DTYPE_t, ndim=2] z, lut):
    cdef np.ndarray[DTYPE_t, ndim=2] f0 = np.zeros(np.shape(z), dtype=np.float64)
    cdef np.ndarray[DTYPE_BOOL_t, ndim=2] clip_f0 = np.bitwise_and(csi >= csi_min_f0, csi <= csi_max_f0)
    cdef np.ndarray[DTYPE_LONG_t, ndim=1] idx = np.floor((len(lut.f0[:, 0]) - 1) * ((csi[clip_f0] - csi_min_f0) / (csi_max_f0 - csi_min_f0))).astype(np.int32)
    f0[clip_f0] = (csi[clip_f0] - lut.f0[idx, 0]) * ((lut.f0[idx + 1, 1] - lut.f0[idx, 1]) / (
            lut.f0[idx + 1, 0] - lut.f0[idx, 0])) + lut.f0[idx, 1]

    idx_max_f0 = np.where(csi > csi_max_f0)
    z_max_f0 = z[idx_max_f0]
    f0[idx_max_f0] = 1. / 2. * np.sqrt(np.pi) / (z_max_f0) ** (1. / 4.) * (
            1. + 3. / (32. * z_max_f0) + 105. / (
            2048. * (z[(csi > csi_max_f0)]) ** 2) + 10395. / (
                    196608. * (z_max_f0) ** 3))
    f0[np.where(csi == 0)] = CONSTANTS.f0_csi_0
    f0[np.where(csi < csi_min_f0)] = 0.
    return f0

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
def compute_f1(np.ndarray[DTYPE_t, ndim=2] csi, double csi_min_f1, double csi_max_f1, np.ndarray[DTYPE_t, ndim=2] z, lut):
    cdef np.ndarray[DTYPE_BOOL_t, ndim=2] clip_f1 = np.bitwise_and(csi >= csi_min_f1, csi <= csi_max_f1)
    cdef np.ndarray[DTYPE_t, ndim=2] f1 = np.zeros(np.shape(z), dtype=np.float64)
    cdef np.ndarray[DTYPE_LONG_t, ndim=1] idx = np.floor((len(lut.f1[:, 0]) - 1) * ((csi[clip_f1] - csi_min_f1) / (csi_max_f1 - csi_min_f1))).astype(np.int32)
    f1[clip_f1] = (csi[clip_f1] - lut.f1[idx, 0]) * ((lut.f1[idx + 1, 1] - lut.f1[idx, 1]) / (
            lut.f1[idx + 1, 0] - lut.f1[idx, 0])) + lut.f1[idx, 1]
    idx_max_f1 = np.where(csi > csi_max_f1)
    f1[idx_max_f1] = (1. / 2.) * 1. / 4. * np.sqrt(np.pi) / (z[idx_max_f1]) ** (3. / 4.)
    f1[np.where(csi == 0)] = CONSTANTS.f1_csi_0
    f1[np.where(csi < csi_min_f1)] = 0.
    return f1

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
def compute_gl(double alpha_p, double lx, double Ly, double Lz, np.ndarray[DTYPE_t, ndim=1] l, double ls, double swh):
    return 1. / np.sqrt(
        alpha_p ** 2 + 4. * (alpha_p ** 2) * (lx / Ly) ** 4 * (l - ls) ** 2 + np.sign(swh) * (swh / (4. * Lz)) ** 2
    )

@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
def ddm_mask_ranges(np.ndarray[DTYPE_t, ndim=2] ddm, mask_ranges, geo, double lx, span, double dr, np.ndarray[DTYPE_t, ndim=1] beam_index):
    if mask_ranges is None:
        mask_ranges_demin = geo.altitude * (np.sqrt(1 + (geo.kappa * ((lx * beam_index) / geo.altitude) ** 2)) - 1)
    else:
        mask_ranges = np.delete(mask_ranges, span)
        mask_ranges_demin = mask_ranges - min(mask_ranges)
    cdef int num_range_gates = ddm.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] r = np.tile(mask_ranges_demin, (num_range_gates, 1))
    cdef np.ndarray[DTYPE_t, ndim=2] dr_tiled = np.tile(dr * np.arange(num_range_gates - 1, -1, -1), (len(beam_index), 1)).T
    cdef np.ndarray[DTYPE_t, ndim=2] ddm_masked = ddm.copy()
    ddm_masked[np.where(r >= dr_tiled)] = 0
    return ddm_masked