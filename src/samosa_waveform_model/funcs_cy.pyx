import cython
cimport cython

import numpy as np
cimport numpy as np

# Define the types for the numpy arrays
ctypedef np.float64_t DTYPE_t
ctypedef np.npy_bool DTYPE_BOOL_t
ctypedef np.int32_t DTYPE_LONG_t
ctypedef np.int64_t DTYPE_LONGLONG_t


@cython.boundscheck(False)
@cython.wraparound(True)
@cython.nonecheck(False)
def get_clipped_f0(np.ndarray[DTYPE_t, ndim=1] csi, float csi_min_f0, float csi_max_f0, np.ndarray[DTYPE_t, ndim=2] lut_f0):
    cdef np.ndarray[DTYPE_t, ndim=1] f0 = np.zeros(np.shape(csi))
    cdef np.ndarray[DTYPE_BOOL_t, ndim=1] clip_f0 = np.bitwise_and(csi >= csi_min_f0, csi <= csi_max_f0)
    cdef np.ndarray[DTYPE_LONGLONG_t, ndim=1] idx = np.floor((lut_f0[:, 0].size - 1) * ((csi[clip_f0] - csi_min_f0) / (csi_max_f0 - csi_min_f0))).astype(np.int64)
    f0[clip_f0] = (csi[clip_f0] - lut_f0[idx, 0]) * ((lut_f0[idx + 1, 1] - lut_f0[idx, 1]) / (lut_f0[idx + 1, 0] - lut_f0[idx, 0])) + lut_f0[idx, 1]
    return f0
