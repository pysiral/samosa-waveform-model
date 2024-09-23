# -*- coding: utf-8 -*-

"""
A test/development script for the sampy sea ice conversion
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import timeit
import numpy as np
import matplotlib.pyplot as plt

from samosa_waveform_model import SAMOSAWaveformModel, ScenarioData
from samosa_waveform_model.scenarios import SurfaceTypeLead


def main():
    """
    :return:
    """

    # Initialize the SAMOSA waveform model
    model_scenario = ScenarioData.cryosat2_sar_example()
    waveform_model = SAMOSAWaveformModel(model_scenario)

    # Example waveform
    model_input = SurfaceTypeLead()
    #  "waveform_model.generate_delay_doppler_waveform(model_input)"
    t = timeit.Timer(lambda: timeit_forward_model(model_input, waveform_model))

    for n in np.arange(100, 501, 100):
        print(t.timeit(n)/float(n))


def timeit_forward_model(model_input, waveform_model):
    waveform_model.generate_delay_doppler_waveform(model_input)


if __name__ == "__main__":
    main()
