# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Memory profiling.

### Fixed
- Tune DeprecationWarning.


## [0.5.1] - 2021-11-16
### Added
- Add pred and target dict support in Lifecycle.

### Fixed
- Avoid detaching loss in step.


## [0.5.0] - 2021-11-12
### Added
- Add preprocess_batch method to Lifecycle.
- Add option for string type in utils.name.
- Add Metric Selector.

### Fixed
- Weight freezing during model loading.
- Fix discriminative_lr param selection for NoneType parameters.
- Fix wandb project naming during hparamsearch.
- Optimizer Schedulers take `accumulate_grad_batches` into account.

### Changed
- Key debug statements while loading models to include both missing and unexpected keys.
- Bumped PL to version 1.4. Holding back on 1.5 due to Tune integration issues.
- Bumped Tune to version 1.8.


## [0.4.6] - 2021-09-21
### Fixed
- Update profile to use model.__call__. This enable non-`forward` executions during profiling.
- Add DefaultMethods Mixin with `warm_up` to make `warm_up` overloadable by Mixins.


## [0.4.5] - 2021-09-08
### Fixed
- Fix `warm_up` function signature.
- Requirement versions.


## [0.4.4] - 2021-09-08
### Added
- `warm_up` function that is called prior to profil .

### Fixed
- Learning rate schedulers discounted steps.


## [0.4.3] - 2021-06-03
### Added
- Logging of layers that are unfrozen.

### Fixed
- Cyclic learning rate schedulers now update on step.


## [0.4.2] - 2021-06-02
### Added
- Added explicit logging of model profiling results.
- Automatic assignment of hparams.num_gpus.

### Fixed
- Finetune weight loading checks.
- Cyclic learning rate schedulers account for batch size.


## [0.4.1] - 2021-05-27
### Fixed
- Feature extraction on GPU.

### Added
- Added explicit logging of hparams.


## [0.4.0] - 2021-05-17
### Fixed
- Pass args correctly to trainer during testing.

### Changed
- CheckpointEveryNSteps now included in ModelCheckpoint c.f. pl==1.3.
- Import from torchmetrics instead of pl.metrics .
- Moved confusion matrix to RideClassificationDataset and updated plot.

### Added
- Feature extraction and visualisation.
- Lifecycle and Finetuneable mixins always included via RideModule.
- Support for pytorch-lightning==1.3.
- Additional tests: Coverage is now at 92%.

### Removed
- Support for nested inheritance of RideModule.
- Support for pytorch-lightning==1.2.


## [0.3.2] - 2021-04-15
### Fixed
- Project dependencies: removed click and added psutil to requirements.
- Logging: Save stdout and stderr to run.log.

### Changed
- Logged results names. Flattened folder structure and streamlines names.

### Added
- Docstrings to remaining core classes.
- Tests that logged results exists.


## [0.3.1] - 2021-03-24
### Added
- Add support for namedtuples in dataset `input_shape` and `output_shape`.
- Add tests for test_enemble.
- Expose more classes via `from ride import XXX`.
- Fix import-error in hparamsearch.
- Fix issues in metrics and add tests.
- Remove unused cache module.

### Change
- Renamed `Dataset` to `RideDataset`.


## [0.3.0] - 2021-03-24
### Added
- Documentation for getting started, the Ride API, and a general API reference.
- Automatic import of `SgdOptimizer`.

### Change
- Renamed `Dataset` to `RideDataset`.


## [0.2.0] - 2021-03-23
### Added
- Initial publicly available implementation of the library.
