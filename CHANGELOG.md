# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2 (Unreleased)

### Added

- Capability to define multiple sensor and SAR parameter presets as yaml files (dedicated `preset` module)
- Dedicated lookup table classes for alpha power and SAMOSA model terms (f0, f1) (dedicated `lut` module)

### Changed

- larger refactoring of the code base with consistency-breaking changes
- Renamed several variables for clarity
  - Waveform model modes renamed: mode=1 -> engine='samosa', mode=2 -> engine='samosa+`.
  - Variable `alpha_p` -> `alpha_power_ptr` and `alpha_power` -> `alpha_power_ddm` to indicate their respective use cases
- Waveform model with engine=`samosa` now enforces nu=1/mss=0 (infinite diffuse surface)
- Fixed alpha_power values have been moved to the sensor settings
- Sensor and sar parameters are now stored in yaml files. 
- Added switch to compute waveform on full or masked delay doppler map (default is masked delay doppler map)
- Increased the default `beamsamp_factor` for CryoSat-2. A higher factor leads to more doppler beams and to a finer resolution in the masked delay doppler map.

### Fixed

- Moved section that were hard-coded to CryoSat-2 and prevented the use for other sensors to respective data models.
- Hamming window weighting was not applied properly in the waveform model (defaulted to False). It has now been set to True for CryoSat-2 ICE products.
- Waveform model returned the waveform instead of the peak power of the unscaled waveform model result.
- Waveform were always computed from the full delay-doppler mask and not the masked version (as in SAMPy). A switch has been
 added and the default behavior is now using the masked delay-doppler mask. 


## 0.1 

### Added
