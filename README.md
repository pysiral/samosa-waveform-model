# Samosa(+) Waveform Model

Python implementation of the SAMOSA+ waveform model for radar altimeter waveforms. 

This project does **not** include waveform fitting procedures, but the implementation 
is optimized for curve fitting operations, with pre-computation of static variables 
for a give waveform scenario.

## Limitations 

- The current implemenation only allows to compute waveform models for CryoSat-2 SAR and SARin radar modes

## Approximate Roadmap

- Add cython implementation of performance relevant functions
- Add dedicated support for CS2 LRM & pLRM
- Add support for other radar altimeters (e.g. Sentinel-3, Sentinel-6)

## Acknowledgements

SAMOSA (SAR Altimetry MOde Studies and Applications) [SAMOSA](https://www.satoc.eu/projects/samosa/) 
is an ESA-funded project to study ocean and inland water applications of Synthetic Aperture 
Radar (SAR) mode (or Delay Doppler mode) altimetry. 

The methods and procedures are described in: 

Dinardo, S. (2020). Techniques and Applications for Satellite SAR Altimetry over water, land and ice. 
Dissertation, Technische Universität Darmstadt, https://doi.org/10.25534/TUPRINTS-00011343 

This project uses code from Python implementation of the SAMOSA+ retracker developed within ESA 
Cryo-TEMPO project [(SAMPy)](https://github.com/cls-obsnadir-dev/SAMPy).