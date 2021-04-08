# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2021-03-24
### Added
- Add support for namedtuples in dataset `input_shape` and `output_shape`
- Add tests for test_enemble
- Expose more classes via `from ride import XXX`
- Fix import-error in hparamsearch
- Fix issues in metrics and add tests
- Remove unused cache module

### Change
- Renamed `Dataset` to `RideDataset`


## [0.3.0] - 2021-03-24
### Added
- Documentation for getting started, the Ride API, and a general API reference
- Automatic import of `SgdOptimizer`

### Change
- Renamed `Dataset` to `RideDataset`


## [0.2.0] - 2021-03-23
### Added
- Initial publicly available implementation of the library
