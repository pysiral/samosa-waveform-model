# -*- coding: utf-8 -*-

"""
A test/development script for the sampy sea ice conversion
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import numpy as np
import matplotlib.pyplot as plt

from samosa_waveform_model import SAMOSAWaveformModel, ScenarioData
from samosa_waveform_model.scenarios import SurfaceTypeLead, SurfaceTypeSeaIce


def main():
    """
    :return:
    """

    # Initialize the SAMOSA waveform model with sensor
    # definition, orbit parameters and SAR options
    scenario_data = ScenarioData.cryosat2_sar_example()
    waveform_model = SAMOSAWaveformModel(scenario_data)

    # Compute waveform based on epoch, swh, mss
    model_input = SurfaceTypeLead()
    model_result = waveform_model.generate_delay_doppler_waveform(model_input)


    plt.figure(dpi=150, figsize=(5, 5))
    plt.plot(model_result.tau, model_result.power)
    plt.title(f"SWH: {model_result.significant_wave_height:.2f}m, MSS={model_result.mean_square_slope}")
    plt.xlabel(r"$\tau$ (seconds)")
    plt.ylabel("Power (normed)")

    plt.figure(dpi=150)
    plt.imshow(model_result.gamma_0, aspect="auto")
    plt.title(r"$\Gamma_{k,l}(0)$ Function")
    plt.show()

    model_input = SurfaceTypeSeaIce()
    model_result = waveform_model.generate_delay_doppler_waveform(model_input)

    plt.figure(dpi=150, figsize=(5, 5))
    plt.plot(model_result.tau, model_result.power)
    plt.title(f"SWH: {model_result.significant_wave_height:.2f}m, MSS={model_result.mean_square_slope}")
    plt.xlabel(r"$\tau$ (seconds)")
    plt.ylabel("Power (normed)")

    plt.figure(dpi=150)
    plt.imshow(model_result.gamma_0, aspect="auto")
    plt.title(r"$\Gamma_{k,l}(0)$ Function")
    plt.show()


if __name__ == "__main__":
    main()
