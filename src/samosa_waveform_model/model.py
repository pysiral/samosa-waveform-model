# -*- coding: utf-8 -*-

"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import warnings
from warnings import warn
import bottleneck as bn
import pandas as pd
import numpy as np
from typing import Union
from typing import Dict, Optional, Literal

from samosa_waveform_model.dataclasses import (SensorParameters, PlatformLocation, SARParameters,
                                               CONSTANTS, WaveformModelOutput, WaveformModelParameters)
from samosa_waveform_model.lut import CS2_LOOKUP_TABLES


from samosa_waveform_model.funcs_py import (compute_gl, compute_gamma0, compute_t_kappa, compute_f0, compute_f1,
                                            ddm_mask_ranges)


# try:
#     from samosa_waveform_model.funcs import (compute_gl, compute_gamma0, compute_t_kappa, compute_f0, compute_f1,
#                                             ddm_mask_ranges)
# except ImportError:
#     msg = """
#     Could not import the compiled functions for the SAMOSA+ waveform model.
#     Please install the compiled functions module func by running
#     'python setup.py build_ext --inplace
#     -> Using python implementation instead
#     """
#     warnings.warn(msg)
#     from samosa_waveform_model.funcs_py import (compute_gl, compute_gamma0, compute_t_kappa, compute_f0, compute_f1,
#                                             ddm_mask_ranges)


class ScenarioData(object):
    """ Scenario (radar parameters, pl for running the SAMOSA+ waveform model """

    def __init__(
            self,
            rp: "SensorParameters",
            geo: "PlatformLocation",
            sar: "SARParameters"

    ) -> None:
        """
        A class for the waveform model input

        """
        self.rp = rp
        self.geo = geo
        self.sar = sar

    @classmethod
    def cryosat2_sar_example(
            cls,
            loc_parameters: Optional[Dict] = None
    ):
        """ Real life CryoSat-2 lead example """

        loc_parameters = {} if loc_parameters is None else loc_parameters

        # Radar altimeter parameters
        sp = SensorParameters.cryosat2_sar()
        # Example position/attitude
        # example_loc = dict(latitude=83.9625006,
        #                    longitude=27.407605,
        #                    altitude=728518.615,
        #                    height_rate=0.466,
        #                    pitch=-0.0010057948506807881,
        #                    roll=-0.0015263707160146328,
        #                    velocity=7518.711587141643)

        loc_dict = dict(
            latitude=83.9625006,
            longitude=27.407605,
            altitude=728518.615,
            height_rate=0.,
            pitch=0.,
            roll=0.,
            velocity=7518.711587141643
        )
        loc_dict.update(loc_parameters)

        geo = PlatformLocation(**loc_dict)
        sar = SARParameters()
        sar.compute_multi_look_parameters(geo=geo, sp=sp)

        return cls(sp, geo, sar)


class SAMOSAWaveformModel(object):
    """
    A class for the modeling of waveforms using the SAMOSA+ waveform model
    (Currently ocean waveforms with free parameter of range and significant waveheight only)
    """

    def __init__(
            self,
            scenario: ScenarioData,
            engine: str = "samosa+",
            use_slope: bool = False,
            weight_factor: float = 1.4705,
            mask_ranges: bool = None,
            mode: Literal[1, 2] = 1,
            collect_fit_params: bool = False
    ) -> None:
        """
        Initialize the forward model

        :param scenario:
        :param engine:
        :param use_slope:
        :param weight_factor:
        :param mask_ranges:
        :param mode:
        """
        self.scenario = scenario
        self.engine = engine
        self.flag_slope = int(use_slope)
        self.weighted = weight_factor is not None
        self.weight_factor = weight_factor
        self.mode = mode
        self.mask_ranges = mask_ranges
        self.lut = CS2_LOOKUP_TABLES
        self.static_parameters = {}
        self.set_mode(self.mode)
        self.collect_fit_params = collect_fit_params
        self.fit_params = []
        self.generate_ddm_counter = 0

    def set_mode(self, mode_num: Literal[1, 2]) -> None:
        """
        Sets the waveform model computation mode. The mode is equivalent
        to the step parameter in SAMPy and determines the computation, of
        alpha_p and alpha_power values.

        Setting this mode triggers the pre-computation of parameters

        :param mode_num: Mode number, must be 1 or 2

        :raises ValueError: Incorrect mode number

        :return: None
        """
        if mode_num not in [1, 2]:
            raise ValueError(f"mode number {mode_num} not in [1, 2")
        self.mode = mode_num
        self._precompute_static_parameters()

    def get_alpha_power(self, swh):
        if self.weighted:
            if self.mode == 1:
                ind = bn.nanargmin(abs(self.lut.alphap_weight[:, 0] - swh))
                alpha_p = self.lut.alphap_weight[:, 1][ind]
                alpha_power = 0.47356
            elif self.mode == 2:
                alpha_p = 0.42349
                alpha_power = 0.47356
            else:
                raise ValueError(f"Invalid mode: {self.mode} (must be 1 or 2)")
        else:
            alpha_p, alpha_power = self.get_alpha_power_no_weights(swh)
        return alpha_p, alpha_power

    def get_alpha_power_no_weights(self, swh):
        ind = np.argmin(abs(self.lut.alphap_noweight[:, 0] - swh))
        alpha_p = self.lut.alphapower_noweight[:, 1][ind]

        ind = np.argmin(abs(self.lut.alphapower_noweight[:, 0] - swh))
        alpha_power = self.lut.alphapower_noweight[:, 1][ind]
        return alpha_p, alpha_power

    def generate_delay_doppler_waveform(
            self,
            waveform_model_parameters: "WaveformModelParameters",
    ) -> "WaveformModelOutput":
        """
        Compute a delay doppler waveform. This is derived from sampy.SAMOSA.__Generate_SamosaDDM

        :param waveform_model_parameters:

        :return:
        """

        # Create short variables name for brevity of some parameters
        geo = self.scenario.geo
        rp = self.scenario.rp
        wfm = waveform_model_parameters
        lut = self.lut
        swh = wfm.significant_wave_height
        # nu = 1. / wfm.mean_square_slope
        nu = wfm.nu
        alt = geo.altitude
        tau = self.scenario.rp.tau - wfm.epoch
        beam_index = self.scenario.sar.beam_index

        # --- Collect the waveform model parameters if requested --->
        # This is useful to restore the parameters variations
        # in an optimization process
        if self.collect_fit_params:
            self.fit_params.append(wfm)

        # --- Compute variables independent of waveform model parameters --->
        # NOTE: For repeated computations, these all need to be computed once
        p = self.static_parameters

        dk = (tau * rp.bandwidth)
        yk = 0 * dk
        dk_positive = np.where(dk > 0)
        yk[dk_positive] = p["Ly"] * np.sqrt(dk[dk_positive])

        sigma_s = (swh / (4. * p["Lz"]))

        # surface elevation standard deviation
        sigma_z = (swh / 4.)

        alpha_p, alpha_power = self.get_alpha_power(swh)

        gl = compute_gl(alpha_p, p["Lx"], p["Ly"], p["Lz"], beam_index, p["ls"], swh)

        csi = gl[None, :] * dk[:, None]
        z = 1. / 4. * csi ** 2

        # gamma0: Surface backscatter response
        gamma_0 = compute_gamma0(p["alpha_y"], p["yp"], p["alpha_x"], nu, alt, p["xl"], p["xp"], yk)

        # Equation 3.19 in Dinardo
        t_kappa = compute_t_kappa(z, dk, nu, alt, p["alpha_y"], p["yp"], p["Ly"])

        # f0 : zero order term of the SAMOSA SAR return waveform model
        f0 = compute_f0(csi, p["csi_min_F0"], p["csi_max_F0"], z, lut)

        # f1 : first order term of the SAMOSA SAR return waveform model
        f1 = compute_f1(csi, p["csi_min_F1"], p["csi_max_F1"], z, lut)

        f = (f0 + sigma_z / p["Lg"] * t_kappa * gl * sigma_s * f1)

        # ddm: delay doppler map
        const = np.sqrt(2. * np.pi * alpha_power ** 4)
        delay_doppler_map = const * np.sqrt(gl) * gamma_0 * f

        delay_doppler_map_masked = ddm_mask_ranges(
            delay_doppler_map,
            self.mask_ranges,
            geo,
            p["Lx"],
            self.scenario.sar.span,
            rp.dr,
            beam_index
        )

        # compute the return power model
        waveform_power = bn.nansum(delay_doppler_map, 1) / len(beam_index)
        peak_power = bn.nanmax(waveform_power)

        waveform_model = wfm.amplitude_scale * (waveform_power/peak_power + wfm.thermal_noise)

        # waveform_model_scaled_power = amplitude_scale * (pr / np.nanmax(pr)) + self.normed_waveform.thermal_noise

        self.generate_ddm_counter += 1

        # Compile the output
        return WaveformModelOutput(
            tau,
            waveform_model,
            wfm.amplitude_scale,
            delay_doppler_map,
            delay_doppler_map_masked,
            wfm.epoch,
            wfm.significant_wave_height,
            wfm.mean_square_slope,
            gamma_0
        )

    def _precompute_static_parameters(self) -> None:

        geo = self.scenario.geo
        rp = self.scenario.rp
        lut = self.lut
        beam_index = self.scenario.sar.beam_index

        p = {}

        p["Lx"] = CONSTANTS.c0 * geo.altitude / (2. * geo.velocity * rp.frequency * rp.pulses_per_burst * rp.pri_sar)
        if self.weighted and self.mode == 2:
            p["Lx"] *= self.weight_factor

        # Ly: pulse-limited radius
        p["Ly"] = np.sqrt(CONSTANTS.c0 * geo.altitude / (geo.kappa * rp.bandwidth))

        # Lz: vertical resolution
        p["Lz"] = CONSTANTS.c0 / (2. * rp.bandwidth)
        factor = 8. * np.log(2.)
        p["alpha_x"] = factor / (geo.altitude ** 2. * rp.beam_width_along ** 2.)
        p["alpha_y"] = factor / (geo.altitude ** 2. * rp.beam_width_across ** 2.)
        p["Lg"] = geo.kappa / (2. * geo.altitude * p["alpha_y"])
        p["xl"] = p["Lx"] * beam_index
        p["ls"] = self.flag_slope * geo.orbit_slope * geo.altitude / (geo.kappa * p["Lx"])
        p["xp"] = +geo.altitude * geo.pitch
        p["yp"] = -geo.altitude * geo.roll
        p["csi_max_F0"] = np.max(lut.f0[:, 0])
        p["csi_min_F0"] = np.min(lut.f0[:, 0])
        p["csi_max_F1"] = np.max(lut.f1[:, 0])
        p["csi_min_F1"] = np.min(lut.f1[:, 0])

        self.static_parameters = p

    def get_fit_params(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self.fit_params) if self.collect_fit_params else None
