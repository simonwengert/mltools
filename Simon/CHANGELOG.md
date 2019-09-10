# Change Log
All notable changes to this project will be documented in this file.

# 0.4 -- 2019-09-10

### Changed
- `calc_kernel_matrix_soap` method is now parallelized.


# 0.3 -- 2019-08-30

### Added
- `calc_distance_matrix()` and `calc_distance_element()` methods.

### Changed
- python2/3 related parts to have compatibility for both version.


# 0.2 -- 2019-08-20

### Added
- set with ID `other`.
- flag `job_dir` for class-instance.
- flag `outfile_teach` for class-instance.
- flag `binary_teach_sparse` for class-instance.
- flag `binary_quip` for class-instance.
- flag `append` in `read_atoms()`.
- `assign_force_atom_sigma_proportion` method.
- `eval_crossval` and `eval_grid` method for extracting a metric (e.g. RMSE) of sampled hyperparameters.
- `get_rmse` method.
- `write_dataframe` method to save dataframes either in plain-text or in (re-readable) HDF5-format.
- `read_dataframe` method.
- `get_crossval_mean` method calulates the mean of the optained hyperparameters of the individual subsets.
- `view_crossval` method shows a 3D plot of the (currently only) RMSE in dependency of two hyperparameters and highlights the corresponding mean values.
- `view_correlation` method creates a correlation plot including the corresponding RMSE.
- `run_local_optimization` method for optimizing hyperparameters (alpha-version).
- `find_furthest` method for furthest point sampling.
- `get_descriptors` mehod that can be applied to a specific set ID.
- `calc_average_kernel` method.
- `calc_kernel_matrix` method.
- `get_info()` method.

### Changed
- `_params_type_to_dir_name` method: individual key-value-pairs are separated by 3-times `_`, a key and its value are separated by 2-times `_`
- `crossvalidation` method is renamed into `run_crossval` (establish nomenclature `run_*` for methods calling QUIP)
- adjusted everything to `gap_fit` (formally `teach_sparse`)

### Fixed

### Removed

## [0.1] -- 2019-05-31
Initial release of the alpha candidate.
