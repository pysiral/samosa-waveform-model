# -*- coding: utf-8 -*-

"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"


from pathlib import Path
import numpy as np


class SAMOSALookupTables(object):
    """
    Container for SAMOSA lookup tables
    """

    def __init__(self) -> None:
        """
        Load the SAMOSA lookup tables
        """

        lut_folder = Path(__file__).parent / "lut"
        kwargs = dict(dtype='float', comments='#', delimiter=None)

        self.f0 = np.genfromtxt(lut_folder / "LUT_F0.txt", **kwargs)
        self.f1 = np.genfromtxt(lut_folder / "LUT_F1.txt", **kwargs)

        kwargs = dict(dtype='float', comments='#', delimiter=',')
        self.alphap_noweight = np.genfromtxt(lut_folder / self.alphap_noweight_file, **kwargs)
        self.alphap_weight = np.genfromtxt(lut_folder / self.alphap_weight_file, **kwargs)
        self.alphapower_noweight = np.genfromtxt(lut_folder / self.alphapower_noweight_file, **kwargs)
        self.alphapower_weight = np.genfromtxt(lut_folder / self.alphapower_weight_file, **kwargs)

    @property
    def alphap_noweight_file(self) -> str:
        return 'alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_NOHAMMING).txt'

    @property
    def alphap_weight_file(self) -> str:
        return 'alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_HAMMING).txt'

    @property
    def alphapower_weight_file(self) -> str:
        return 'alphaPower_table_CONSTANT_SWH20_10_Feb_2020(CS2_NOHAMMING).txt'

    @property
    def alphapower_noweight_file(self) -> str:
        return 'alphaPower_table_CONSTANT_SWH20_10_Feb_2020(CS2_NOHAMMING).txt'
