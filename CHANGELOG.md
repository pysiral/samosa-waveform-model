# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2 (Unreleased)

### Added
- Capability to define multiple sensor and SAR parameter presets as yaml files
- Dedicated lookup table classes for alpha power and SAMOSA model terms (f0, f1)

### Changed
- larger refactoring of the code with code-breaking changes
- Waveform model modes renamed: mode=1 -> engine='samosa', mode=2 -> engine='samosa+`
- Waveform model with engine=`samosa` now enforced nu=1/mss=0 (infinite diffuse surface)
- Fixed alpha_power values have been moved to the sensor settings
- Sensor and sar parameters are now stored in yaml files. More than one sensor is now possible
- Added switch to compute waveform on full or masked delay doppler map

### Fixed
- waveform model returned the waveform instead of the peak power of the unscaled waveform model result

## 0.1 

### Added
