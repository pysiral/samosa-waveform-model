# -*- coding: utf-8 -*-

"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from dataclasses import dataclass
from functools import cached_property
from typing import List
import numpy as np

PLATFORM_PRESETS = {
    "cryosat2": {
        "sar": dict(
            platform="cryosat2",
            sensor="siral",
            mode="sar",
            pulses_per_burst=64,
            range_gates_per_pulse=128,
            zero_padding_factor=2,
            pulse_repetition_frequency=18181.8181818181,
            burst_repetition_interval=0.0117929625,
            frequency=13.575e9,
            bandwidth=320e6,
            beam_width_along=np.deg2rad(1.10),
            beam_width_across=np.deg2rad(1.22)
        ),
        "sin": dict(
            platform="cryosat2",
            sensor="siral",
            mode="sar",
            pulses_per_burst=64,
            range_gates_per_pulse=512,
            zero_padding_factor=2,
            pulse_repetition_frequency=18181.8181818181,
            burst_repetition_interval=0.0117929625,
            frequency=13.575e9,
            bandwidth=320e6,
            beam_width_along=np.deg2rad(1.10),
            beam_width_across=np.deg2rad(1.22)
        )
    }
}

SUPPORTED_MODES = [
    f"{platform_key}:{radar_mode_key}"
    for platform_key in PLATFORM_PRESETS.keys()
    for radar_mode_key in PLATFORM_PRESETS[platform_key].keys()
]


@dataclass
class Constants:
    """ A set of constants needed for the waveform model """
    c0: float = 299792458.  # speed of light in m/sec
    R_e: float = 6378137.  # Reference Ellipsoid Earth Radius in m
    f_e: float = 1. / 298.257223563  # Reference Ellipsoid Earth Flatness
    gamma_3_4: float = 1.2254167024651779  # Gamma Function Value at 3/4

    @cached_property
    def ecc_e(self) -> float:
        return np.sqrt((2. - self.f_e) * self.f_e)

    @cached_property
    def b_e(self) -> float:
        return self.R_e * np.sqrt(1. - self.ecc_e ** 2.)


CONSTANTS = Constants()


@dataclass
class SensorParameters:
    """ Information of the Radar Altimeter and Processing Configuration (hard coded to CryoSat-2 SIRAL SAR) """
    platform: str
    sensor: str
    mode: str
    pulses_per_burst: int  # number of pulses per burst [Np_burst]
    range_gates_per_pulse: int  # number of the range gates per pulse (without zero-padding) [Npulse]
    zero_padding_factor: int
    pulse_repetition_frequency: float  # Pulse Repetition Frequency in Hz (SAR mode) [PRF_SAR]
    burst_repetition_interval: float  # Burst Repetition Interval in sec [BRI]
    frequency: float  # Carrier Frequency in Hz [f_0]
    bandwidth: float  # Sampled Bandwidth in Hz [Bs]
    beam_width_along: float  # (rad) Antenna 3 dB beamwidth (along-track) [theta_3x]
    beam_width_across: float  # (rad) Antenna 3 dB beamwidth (cross-track) [theta_3y]
    num_look_min: float = -90.0
    num_look_max: float = 90.0
    beamsamp_factor: float = 1.0

    @classmethod
    def get(cls, platform_name: str, radar_mode_name: str) -> "SensorParameters":
        """
        Retrieve sensor parameters for given platform and radar mode.

        :param platform_name: Platform identifier (e.g. `cryosat2`)
        :param radar_mode_name: Radar mode identifier (e.g. `sar`)

        :raises KeyError: Invalid platform / radar mode combination

        :return: Sensor parameter
        """
        try:
            return cls(**PLATFORM_PRESETS[platform_name][radar_mode_name])
        except KeyError as ke:
            msg = f"Invalid platform/radar mode combination: {platform_name}/{radar_mode_name} [{SUPPORTED_MODES}]"
            raise KeyError(msg) from ke

    @classmethod
    def cryosat2_sar(cls) -> "SensorParameters":
        """
        Create a class instance for CryoSat-2 SAR waveforms

        :return:
        """
        return cls(**PLATFORM_PRESETS["cryosat2"]["sar"])

    @cached_property
    def pri_sar(self) -> float:
        return 1. / self.pulse_repetition_frequency

    @cached_property
    def wavelength(self) -> float:
        return CONSTANTS.c0 / self.frequency

    @cached_property
    def dfa(self) -> float:
        return self.pulse_repetition_frequency / self.pulses_per_burst

    @cached_property
    def dr(self) -> float:
        return CONSTANTS.c0 / (2. * self.bandwidth * self.zero_padding_factor)

    @cached_property
    def tau(self) -> np.ndarray:
        num_gates = self.range_gates_per_pulse * self.zero_padding_factor
        dt = 1. / (self.bandwidth * self.zero_padding_factor)
        return np.arange(-(num_gates / 2.) * dt, ((num_gates - 1) / 2) * dt, dt)


@dataclass
class SARParameters:

    look_angles: np.ndarray = None
    doppler_frequencies: np.ndarray = None
    span: np.ndarray = None
    beam_index: np.ndarray = None

    def compute_multi_look_parameters(
            self,
            geo: "PlatformLocation",
            sp: "SensorParameters"
    ) -> None:
        """
        Compute a set of beam parameters. Note the methods called depend on
        the order in which these are called

        :param geo:
        :param sp:

        :return: Nothing
        """

        # Compute default parameters if look angles has not been set to actual data
        if self.look_angles is None:
            self._compute_look_angles(
                geo.velocity,
                geo.altitude,
                geo.kappa,
                sp.burst_repetition_interval,
                sp.num_look_min,
                sp.num_look_max
            )
        self._compute_doppler_frequencies(geo.velocity, sp.wavelength)
        self._compute_beam_index(sp.beamsamp_factor, sp.dfa)

    def _compute_look_angles(
            self,
            velocity: float,
            altitude: float,
            kappa: float,
            burst_repetition_interval: float,
            num_look_min: float = -90.0,
            num_look_max: float = 90.0
    ) -> None:
        """
        Computes the look angles for a given height and velocity

        :param velocity:
        :param altitude:
        :param kappa:
        :param burst_repetition_interval:
        :param num_look_min
        :param num_look_max:

        :return:
        """
        dtheta = velocity * burst_repetition_interval / (altitude * kappa)
        theta1 = np.pi / 2. + dtheta * num_look_min
        theta2 = np.pi / 2. + dtheta * num_look_max
        self.look_angles = np.rad2deg(np.arange(theta1, theta2, dtheta))

    def _compute_doppler_frequencies(
            self,
            velocity: float,
            wavelength: float
    ) -> None:
        """
        Compute doppler frequencies

        :param velocity:
        :param wavelength:

        :return:
        """
        self.doppler_frequencies = (2 * velocity / wavelength) * np.cos(np.deg2rad(self.look_angles))

    def _compute_beam_index(
            self,
            beamsamp_factor: float,
            dfa: float
    ) -> None:
        beam_index = np.around(beamsamp_factor * self.doppler_frequencies / dfa) / beamsamp_factor
        self.span = np.where(np.diff(beam_index, axis=0) == 0)
        self.beam_index = np.delete(beam_index, self.span)


@dataclass
class WaveformModelParameters:
    """
    Data class for the background of the waveform model.

    NOTE: The parameter standard deviation are the result of
    waveform model optimization process and will remain empty
    for the forward model. These are included here because the
    waveform fitting procedure in pysiral relies on this dataclass.
    """
    epoch: float = None  # The epoch in seconds
    epoch_sdev: float = None
    significant_wave_height: float = None
    significant_wave_height_sdev: float = None
    nu: float = 0.0
    nu_sdev: float = None
    amplitude_scale: float = 1.0
    thermal_noise: float = 0.0


    @property
    def mean_square_slope(self) -> float:
        try:
            return 1. / self.nu
        except ZeroDivisionError:
            return np.inf

    @property
    def args_list(self) -> List[float]:
        return [self.epoch, self.significant_wave_height, self.nu]


@dataclass
class PlatformLocation:
    """
    """

    latitude: float = None  # latitude in degree for the waveform under iteration
    longitude: float = None  # longitude in degree between -180, 180 for the waveform under iteration
    altitude: float = None  # Orbit height in meter for the waveform under iteration
    velocity: float = None  # Satellite Velocity in m/s
    height_rate: float = None  # Orbit Height rate in m/s for the waveform under iteration
    pitch: float = None  # Altimeter Reference Frame Pitch in radian
    roll: float = None  # Altimeter Reference Frame Roll in radian
    track_sign: int = 0  # -1 for ascending & +1 for descending, set it to zero if flag_slope=False in

    @cached_property
    def earth_radius(self) -> float:
        return np.sqrt(
            CONSTANTS.R_e ** 2.0 * (np.cos(np.deg2rad(self.latitude))) ** 2. +
            CONSTANTS.b_e ** 2.0 * (np.sin(np.deg2rad(self.latitude))) ** 2.
        )

    @cached_property
    def kappa(self) -> float:
        return 1. + self.altitude / self.earth_radius

    @cached_property
    def orbit_slope(self) -> float:
        return self.track_sign * \
               ((CONSTANTS.R_e ** 2 - CONSTANTS.b_e ** 2) / (2. * self.earth_radius ** 2)) * \
               np.sin(np.deg2rad(2. * self.latitude)) - \
               (-self.height_rate / self.velocity)


@dataclass
class WaveformModelOutput:
    """
    Output of the SAMOSA waveform model
    """
    tau: np.ndarray
    power: np.ndarray
    peak_power: float
    delay_doppler_map: np.ndarray
    delay_doppler_map_masked: np.ndarray
    epoch: float
    significant_wave_height: float
    mean_square_slope: float
    gamma_0: np.ndarray
