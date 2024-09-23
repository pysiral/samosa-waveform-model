# -*- coding: utf-8 -*-

"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from samosa_waveform_model.dataclasses import WaveformModelParameters


class SurfaceTypeLead(WaveformModelParameters):

    def __init__(
        self,
        significant_wave_height=0.1,
        nu=1e6,
        epoch=0.0
    ):
        super(SurfaceTypeLead, self).__init__(
            significant_wave_height=significant_wave_height,
            nu=nu,
            epoch=epoch
        )


class SurfaceTypeSeaIce(WaveformModelParameters):

    def __init__(
        self,
        significant_wave_height=0.2,
        nu=10,
        epoch=0.0
    ):
        super(SurfaceTypeSeaIce, self).__init__(
            significant_wave_height=significant_wave_height,
            nu=nu,
            epoch=epoch
        )
