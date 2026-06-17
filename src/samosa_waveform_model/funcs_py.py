import numpy as np
from samosa_waveform_model.dataclasses import CONSTANTS


def compute_gamma0(alpha_y, yp, alpha_x, nu, alt, xl, xp, yk):
    xl_ = xl[None, :]
    yk_ = yk[:, None]
    alt2 = alt ** 2
    return np.exp(
        -alpha_y * yp ** 2 - alpha_x * (xl_ - xp) ** 2 - xl_ ** 2 * nu / alt2 -
        (alpha_y + nu / alt2) * yk_ ** 2) * np.cosh(2. * alpha_y * yp * yk_)


def compute_t_kappa(z, dk, nu, alt, alpha_y, yp, ly):
    # TODO: Can dimension be inferred from other parameter
    # TODO: dk_positive_idx and dk_negative_idx has been computed before, can be passed as parameter
    t_kappa = np.zeros(np.shape(z))
    dk_positive = dk > 0
    dk_positive_idx = np.where(dk_positive)
    dk_negative_idx = np.where(np.logical_not(dk_positive))
    dk_positive_sqrt = np.sqrt(dk[dk_positive_idx])
    t_kappa[dk_positive_idx, :] = (
            (1. + nu / ((alt ** 2) * alpha_y)) - yp / (ly * dk_positive_sqrt) * np.tanh(
        2. * alpha_y * yp * ly * dk_positive_sqrt)[None, :]).T
    t_kappa[dk_negative_idx, :] = (1. + nu / ((alt ** 2) * alpha_y)) - 2. * alpha_y * yp ** 2
    return t_kappa


def compute_f0(csi, csi_min_f0, csi_max_f0, z, lut):
    f0 = get_clipped_f0(csi, csi_min_f0, csi_max_f0, lut.f0)
    f0 = set_f0_z_idx_max_f0(csi, csi_max_f0, f0, z)
    f0 = set_f0_csi_eq_0(f0, csi)
    f0 = set_f0_csi_lt_csi_min(f0, csi, csi_min_f0)
    return f0


def get_clipped_f0(csi, csi_min_f0, csi_max_f0, lut_f0):
    f0 = np.zeros(np.shape(csi))
    clip_f0 = np.bitwise_and(csi >= csi_min_f0, csi <= csi_max_f0)
    idx = np.floor((lut_f0[:, 0].size - 1) * ((csi[clip_f0] - csi_min_f0) / (csi_max_f0 - csi_min_f0))).astype(int)
    f0[clip_f0] = (csi[clip_f0] - lut_f0[idx, 0]) * ((lut_f0[idx + 1, 1] - lut_f0[idx, 1]) / (
            lut_f0[idx + 1, 0] - lut_f0[idx, 0])) + lut_f0[idx, 1]
    return f0


def set_f0_z_idx_max_f0(csi, csi_max_f0, f0, z):
    idx_max_f0 = find_idx_max_f0(csi, csi_max_f0)
    z_idx_max_f0 = z[idx_max_f0]
    f0[idx_max_f0] = 1. / 2. * CONSTANTS.sqrt_pi / z_idx_max_f0 ** (1. / 4.) * (
            1. + 3. / (32. * z_idx_max_f0) + 105. / (
            2048. * z_idx_max_f0 ** 2) + 10395. / (
                    196608. * z_idx_max_f0 ** 3))
    return f0


def find_idx_max_f0(csi, csi_max_f0):
    return np.where(csi > csi_max_f0)


def set_f0_csi_eq_0(f0, csi):
    f0[np.where(csi == 0)] = CONSTANTS.f0_csi_0
    return f0


def set_f0_csi_lt_csi_min(f0, csi, csi_min_f0):
    f0[np.where(csi < csi_min_f0)] = 0
    return f0


def compute_f1(csi, csi_min_f1, csi_max_f1, z, lut):
    f1 = get_clipped_f1(csi, csi_min_f1, csi_max_f1, lut.f1)
    # clip_f1 = np.bitwise_and(csi >= csi_min_f1, csi <= csi_max_f1)
    # f1 = np.zeros(np.shape(z))
    # idx = np.floor((len(lut.f1[:, 0]) - 1) * ((csi[clip_f1] - csi_min_f1) / (csi_max_f1 - csi_min_f1))).astype(int)
    # f1[clip_f1] = (csi[clip_f1] - lut.f1[idx, 0]) * ((lut.f1[idx + 1, 1] - lut.f1[idx, 1]) / (
    #         lut.f1[idx + 1, 0] - lut.f1[idx, 0])) + lut.f1[idx, 1]
    f1 = set_f1_csi_gt_csi_max(f1, csi, csi_max_f1, z)
    # idx_max_f1 = np.where(csi > csi_max_f1)
    # f1[idx_max_f1] = (1. / 2.) * 1. / 4. * CONSTANTS.sqrt_pi / (z[idx_max_f1]) ** (3. / 4.)
    f1 = set_f1_csi_eq_0(f1, csi)
    # f1[np.where(csi == 0)] = CONSTANTS.f1_csi_0
    f1 = set_f1_csi_lt_csi_min(f1, csi, csi_min_f1)
    # f1[np.where(csi < csi_min_f1)] = 0
    return f1


def get_clipped_f1(csi, csi_min_f1, csi_max_f1, lut_f1):
    clip_f1 = np.bitwise_and(csi >= csi_min_f1, csi <= csi_max_f1)
    f1 = np.zeros(np.shape(csi))
    idx = np.floor((len(lut_f1[:, 0]) - 1) * ((csi[clip_f1] - csi_min_f1) / (csi_max_f1 - csi_min_f1))).astype(int)
    f1[clip_f1] = (csi[clip_f1] - lut_f1[idx, 0]) * ((lut_f1[idx + 1, 1] - lut_f1[idx, 1]) / (
            lut_f1[idx + 1, 0] - lut_f1[idx, 0])) + lut_f1[idx, 1]
    return f1


def set_f1_csi_gt_csi_max(f1, csi, csi_max_f1, z):
    idx_max_f1 = np.where(csi > csi_max_f1)
    f1[idx_max_f1] = (1. / 2.) * 1. / 4. * CONSTANTS.sqrt_pi / (z[idx_max_f1]) ** (3. / 4.)
    return f1


def set_f1_csi_eq_0(f1, csi):
    f1[np.where(csi == 0)] = CONSTANTS.f1_csi_0
    return f1


def set_f1_csi_lt_csi_min(f1, csi, csi_min_f1):
    f1[np.where(csi < csi_min_f1)] = 0
    return f1


def compute_gl(alpha_p, lx, Ly, Lz, l, ls, swh):
    return 1. / np.sqrt(
        alpha_p ** 2 + 4. * (alpha_p ** 2) * (lx / Ly) ** 4 * (l - ls) ** 2 + np.sign(swh) * (swh / (4. * Lz)) ** 2
    )


def ddm_mask_ranges(ddm, mask_ranges, geo, lx, span, dr, beam_index):
    if mask_ranges is None:
        mask_ranges_demin = geo.altitude * (np.sqrt(1 + (geo.kappa * ((lx * beam_index) / geo.altitude) ** 2)) - 1)
    else:
        mask_ranges = np.delete(mask_ranges, span)
        mask_ranges_demin = mask_ranges - min(mask_ranges)
    num_range_gates = ddm.shape[0]
    r = np.tile(mask_ranges_demin, (num_range_gates, 1))
    dr_tiled = np.tile(dr * np.arange(num_range_gates - 1, -1, -1), (len(beam_index), 1)).T
    ddm_masked = ddm.copy()
    ddm_masked[np.where(r >= dr_tiled)] = 0
    return ddm_masked
