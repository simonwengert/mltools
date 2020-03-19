from __future__ import print_function
import sys
import os
import copy
import subprocess
import tempfile
import shutil
import itertools as ito
import numpy as np
import scipy.optimize
import random
import pandas as pd
import sklearn.metrics
try:
    # At some hosts python3-tk are not installed
    # which would throw an error here.
    # In this way (using try) an error is only
    # thrown in case the module is called.
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    print(e)
    print('Couldn\'t load plotting packages. Corresponding functionalities will not be available!')
import time
import multiprocessing as mp
import datetime

import ase.io
import quippy.descriptors
if sys.version_info[0] == 3:
    import quippy.convert

import mltools.misc


class Gap(object):
    """
    basic usage
        >>> gap = Gap()
        >>> gap.job_dir = '/path/to/dir'
        >>> gap.outfile_gap_fit = 'gap_fit.out'
        >>> gap.params_gap_fit = {...}
        >>> gap.gaps = [{'name' : ..., ...},
        >>>             {'name' : ..., ...}]
        >>> gap.read_atoms('./path/to/train.xyz', 'train')
        >>> gap.run_gap_fit()
    """
    def __init__(self, **kwargs):
        self._set_ids = ['train', 'validate', 'test', 'other']
        self._metrics = ['RMSE']

        # defaults
        self.job_dir = kwargs.pop('job_dir', os.path.abspath(os.getcwd()))
        self.outfile_gap_fit = kwargs.pop('outfile_gap_fit', 'gap_fit.out')

        self._binary_gap_fit = kwargs.pop('binary_gap_fit', '')  # allows submission to cluster
        self._binary_quip = kwargs.pop('binary_quip', '')  # allows submission to cluster


    # cmd_* cannot be changed by user directly
    @property
    def cmd_gap_fit(self):
        "Update command string when called"
        self._build_cmd_gap_fit()
        return self._cmd_gap_fit

    @property
    def cmd_quip(self):
        "Update command string when called"
        self._build_cmd_quip()
        return self._cmd_quip

    # output
    @property
    def job_dir(self):
        return self._job_dir

    @job_dir.setter
    def job_dir(self, path):
        self._job_dir = path

    @property
    def outfile_gap_fit(self):
        return self._outfile_gap_fit

    @outfile_gap_fit.setter
    def outfile_gap_fit(self, filename):
        self._outfile_gap_fit = filename

    @property
    def errfile_gap_fit(self):
        return self.outfile_gap_fit[:-4]+'.err' if self.outfile_gap_fit.endswith('.out') else self.outfile_gap_fit+'.err'

    # atoms
    @property
    def atoms_train(self):
        return self._atoms_train

    @atoms_train.setter
    def atoms_train(self, atoms):
        self._atoms_train = atoms

    @property
    def atoms_validate(self):
        return self._atoms_validate

    @atoms_validate.setter
    def atoms_validate(self, atoms):
        self._atoms_validate = atoms

    @property
    def atoms_test(self):
        return self._atoms_test

    @atoms_test.setter
    def atoms_test(self, atoms):
        self._atoms_test = atoms

    @property
    def atoms_other(self):
        return self._atoms_other

    @atoms_other.setter
    def atoms_other(self, atoms):
        self._atoms_other = atoms

    # full and direct excess to params-dicts
    @property
    def params_gap_fit(self):
        return self._params_gap_fit

    @params_gap_fit.setter
    def params_gap_fit(self, params):
        self._params_gap_fit = params

    @property
    def gaps(self):
        for gap in self._gaps:
            self._check_key(gap, 'name')
        return self._gaps

    @gaps.setter
    def gaps(self, gaps):
        if not isinstance(gaps, list):
            gaps = [gaps]

        for gap in gaps:
            self._check_key(gap, 'name')

        self._gaps = gaps

    @property
    def params_quip(self):
        return self._params_quip

    @params_quip.setter
    def params_quip(self, params):
        self._params_quip = params

    # checks
    def _check_key(self, items, key):
        "Check if a (required) key is in a dictionary, e.g. ``name`` in ``self.params_gap_fit``"
        if not key in items:
            msg = 'Key \'{}\' not found.'.format(key)
            raise KeyError(msg)

    def _check_set_id(self, set_id):
        "Check if ``set_id`` is part of ``self._set_ids``"
        if not set_id in self._set_ids:
            msg = '\'set_id\' must be one of \'{}\''.format(' '.join(self._set_ids))
            raise ValueError(msg)

    # atoms handling
    def read_atoms(self, path, set_id, append=False):
        """
        Read geometries from file and store as attribute of the instance (self.atoms_<set_id>).

        Parameters:
        -----------
        path : string
            Location of file storing (ase-readable) geometries.
        set_id : string
            Defines to which set the geometries belong to.
            Must be one of the string stored in _set_ids.
            The atoms will be stored in self.atoms_<set_id>
        """
        self._check_set_id(set_id)
        if append:
            setattr(self, 'atoms_'+set_id, getattr(self, 'atoms_'+set_id, []) + ase.io.read(path, index=':'))
        else:
            setattr(self, 'atoms_'+set_id, ase.io.read(path, index=':'))

    def write_atoms(self, destination, set_id):
        """
        Write self.atoms_<set_id> in xyz-file.

        Parameters:
        -----------
        destination : string
            Location of file the geometries will be written to.
            Must end with ``.xyz``; will be extended if not.
        set_id : string
            Defines the geometry set that will be written.
            Must be one of the string stored in _set_ids.
        """
        suffix = '.xyz'
        if not destination.endswith(suffix):
            destination += suffix
        self._check_set_id(set_id)

        dirname = os.path.dirname(destination)
        if dirname:
            self._make_dirs(dirname)
        ase.io.write(destination, getattr(self, 'atoms_'+set_id))

    def set_lattices(self, length, set_id):
        "Purpose is to assign huge (cubic) cells to non-periodic systems in order to statisfy the fitting codes request for periodicity."
        for atoms in getattr(self, 'atoms_'+set_id):
            atoms.set_cell(np.diag([length]*3))

    def _calc_energy_sigmas_linear(self, ref_values, sigma_range):
        "Map ``ref_values`` to a range within ``sigma_range`` in a linear fashion."
        ref_values = np.asarray(ref_values)
        ref_min, ref_max = np.min(ref_values), np.max(ref_values)
        slope = (sigma_range[1]-sigma_range[0]) / float(ref_max-ref_min)
        return sigma_range[0] + slope*(ref_values-ref_min)

    def assign_energy_sigma_linear(self, ref_values, sigma_range):
        """
        Add the property ``energy_sigma`` to a set of atoms.

        Parameters:
        -----------
        ref_values : ndarray or list
            Values to be mapped to energy_sigma values and
            assigned to the atoms in self.atoms_train.
            thus, needs to have same length as self.atoms_train.
        sigma_range: ndarray or list
            Stores the minimum and maximum value of the sesired
            energy_sigma's.
        """
        if not len(self.atoms_train) == len(ref_values):
            raise ValueError('Dimension mismatch: requested atoms set and ``ref_values`` must have same dimension')

        energy_sigmas = self._calc_energy_sigmas_linear(ref_values, sigma_range)
        for atoms, energy_sigma in zip(self.atoms_train, energy_sigmas):
                atoms.info['energy_sigma'] = energy_sigma

    def assign_force_atom_sigma_proportion(self, proportion, arrays_key='force',
                                           zero_sigma=1E-5, ll=-1, ul=-1, set_id='train'):
        """
        adds an array to each atoms obj to determine the
        force_atom_sigma

        Parameters:
        -----------
        proportion : float
            This proportion of the reference data (see ``arrays_key``)
            will be used as the corresponding ``force_sigma_atom`` value.
        arrays_key : str
            Name of the key pointing to the reference values.
            The norm of it will will be used for each individual atom.
        zero_sigma : float
            The ``force_atom_sigma`` value for atoms
            with vanishing force norm.
        ll : float
            Lower limit value. If a `force_atom_sigma` falls below `ll`
            it will be set to `ll`.
        ul : float
            Upper limit value. If a `force_atom_sigma` exceeds `ul`
            it will be set to `ul`.
        set_id : string
            Defines the geometry set that will used.
            Must be one of the string stored in _set_ids.
        """
        for atoms in getattr(self, 'atoms_' + set_id):
            fas = proportion*np.linalg.norm(atoms.arrays[arrays_key], axis=1)  # force_atom_sigma
            fas[fas == 0] = zero_sigma

            if ll != -1:
                fas[fas < ll] = ll
            if ul != -1:
                fas[fas > ul] = ul

            atoms.set_array("force_atom_sigma", fas)

    # dumping parameters
    def _dict_to_string(self, items):
        keys = sorted(items)
        return 'dict(' + ',\n     '.join('{0} = {1}'.format(key, items[key]) for key in keys) + ')\n'

    def write_gap_fit_parameters(self):
        "Write gap_fit-parameters and gap-parameters to file."
        with open(os.path.join(self.job_dir, 'gap_fit.params'), 'w') as o_file:
            o_file.write('# params_gap_fit\n')
            o_file.write(self._dict_to_string(self.params_gap_fit))

            o_file.write('\n')
            o_file.write('# gaps\n')
            for gap in self.gaps:
                o_file.write(self._dict_to_string(gap))

    def write_quip_parameters(self):
        "Write quip-parameters to file."
        with open(os.path.join(self.job_dir, 'quip.params'), 'a') as o_file:
            o_file.write('# params_quip\n')
            o_file.write(self._dict_to_string(self.params_quip))

    # command handling
    def _build_cmd_gap_fit(self):
        "Builds the gap_fit command-line string"
        items_copy = copy.deepcopy(self._params_gap_fit)  # avoid changes in self.params_gap_fit
        cmd_str = '! gap_fit ' if not self._binary_gap_fit else self._binary_gap_fit+' '
        cmd_str += 'default_sigma={' + ' '.join([str(df) for df in items_copy.pop('default_sigma')]) + '}'
        cmd_str += ' '
        cmd_str += self._build_assign_str(items_copy)
        cmd_str += ' '
        cmd_str += self._build_gap_str()
        self._cmd_gap_fit = cmd_str

    def _build_assign_str(self, items):
        "Turns dictionary to a string of the form 'key=val' concatenating the items by a whitespace"
        assign_str = ''
        for key, value in items.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool): # account for int represented in scientific notation
                assign_str += '{}={:g} '.format(key, value)
            else:
                assign_str += '{}={} '.format(key, value)
        return assign_str[:-1]

    def _build_gap_str(self):
        "Builds the gap-related part of the gap_fit command-line string"
        cmd_str = 'gap={'
        cmd_str += ' :'.join([self._build_potential_str(gap) for gap in self.gaps])
        cmd_str += '}'
        return cmd_str

    def _build_potential_str(self, items):
        "Build the command-line string for a single descriptor within the gap-related part of gap_fit"
        items_copy = copy.deepcopy(items)  # avoid changes in self.gaps
        pot_str = items_copy.pop('name')
        pot_str += ' '
        pot_str += self._build_assign_str(items_copy)
        return pot_str

    def _build_cmd_quip(self):
        "Builds the quip command-line string"
        cmd_str = '! quip ' if not self._binary_quip else self._binary_quip+' '
        cmd_str += self._build_assign_str(self._params_quip)
        cmd_str += ' | grep AT | sed \'s/AT//\''
        self._cmd_quip = cmd_str

    # command execution
    def run_gap_fit(self, try_run=False):
        """
        Executes the gap_fit command based on the defined settings in
            self.params_gap_fit,
            self.gaps,
            self.job_dir,
            self.outfile_gap_fit.

        The training-set (self.atoms_train) will automatically be written to the file
        specified in self.params_gap_fit ('at_file').

        Standard output and output for error will be written into separated files.

        Parameters:
        -----------
        try_run : boolean
            Run in test-mode.
        """
        self._make_dirs(self.job_dir)
        self.write_atoms(os.path.join(self.job_dir, self.params_gap_fit['at_file']), 'train')
        self.write_gap_fit_parameters()

        cwd = os.getcwd()
        os.chdir(self.job_dir)
        print(self.cmd_gap_fit)
        if not try_run:
            os.system('{command} 1>{stdout} 2>{stderr}'.format(command=self.cmd_gap_fit, stdout=self.outfile_gap_fit, stderr=self.errfile_gap_fit))
        os.chdir(cwd)

        # NOTE: Would be more clean to have it via Popen, but Popen cannot handle this rather complex expression
        # process = subprocess.Popen(self._cmd_gap_fit.split(), stdout=subprocess.PIPE)  # stdout to file, stderr to screen
        # while True:
        #     with open(os.path.join(self.job_dir, self.outfile_gap_fit), 'a') as o_file:
        #         out = process.stdout.read(1)
        #         if out == '' and process.poll() != None:
        #             break
        #         if out != '':
        #             o_file.write(out)
        #             o_file.flush()

    def run_quip(self, set_id, try_run=False):
        """
        Executes the quip command based on the defined settings in
            self.params_quip,
            self.job_dir,

        The <set_id>-set (self.atoms_<set_id>) will automatically be written to the file
        specified in self.params_quip ('atoms_filename').
        The file containing the predictions made when running the command will be written
        to the file specified in self.params_quip ('atoms_filename') with the prefix ``quip_``.

        Standard output and output for error will be written into separated files.

        Parameters:
        -----------
        try_run : boolean
            Run in test-mode.
        set_id : string
            Defines the geometry set that will be used for predicting outcomes.
            Must be one of the string stored in _set_ids.
        """
        self._make_dirs(self.job_dir)
        self.write_atoms(os.path.join(self.job_dir, self.params_quip['atoms_filename']), set_id)
        self.write_quip_parameters()

        outfile_quip = 'quip_' + self.params_quip['atoms_filename']
        errfile_quip = outfile_quip[:-4]+'.err'

        cwd = os.getcwd()
        os.chdir(self.job_dir)
        print(self.cmd_quip)
        if not try_run:
            os.system('{command} 1>{stdout} 2>{stderr}'.format(command=self.cmd_quip, stdout=outfile_quip, stderr=errfile_quip))
        os.chdir(cwd)

    def _make_dirs(self, dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    def run_sample_grid(self, gap_fit_ranges, gaps_ranges, add_run_quip=None, add_atoms_filename=None, del_gp_file=True,
                        try_run=False, del_at_file=False, del_atoms_filename=False):
        """
        Learn and validate gap-potentials on a grid of parameters.

        Parameters:
        -----------
        gap_fit_ranges : dict
            Stores the keys and the range of values to be sampled.
        gaps_ranges : list (or dict)
            List of dictionaries (or a single dictionary in case
            only a single gap-potential is used).
            Each dictionary stores the keys and the range of values to be sampled.
        add_run_quip : list (N)
            List of lists with each sub-list containing atoms-objects.
        add_atoms_filename : list (N)
            List of filenames corresponding to the N entries of ``add_run_quip``.
            Will be used to update self.params_quip['atoms_filename'] accordingly.
        del_gp_file : boolean
            Allows to remove the (sometimes rather large) ``gp_file``.
        try_run : boolean
            Run in test-mode.
        del_at_file: boolean
            Allows to remove the large number of ``at_file`` xyz-files.
        del_atoms_filename: boolean
            Allows to remove the large number of ``atoms_filename`` xyz-files.
        """
        # TODO:
        #   - TST for add_*

        if not isinstance(gaps_ranges, list):
            gaps_ranges = [gaps_ranges]
        if not len(gaps_ranges) == len(self.gaps):
            raise ValueError('``gaps_ranges`` must have same length as ``self.gaps``')

        if add_run_quip is None:
            add_run_quip = []
        if add_atoms_filename is None:
            add_atoms_filename = []

        _job_dir = self.job_dir  # used to reset it later again to that value

        for params_tuple in self._get_params_tuples(gap_fit_ranges, gaps_ranges):
            self._set_params_tuple_values(params_tuple)
            self.job_dir = os.path.join(_job_dir, self._params_tuple_to_dir_name(params_tuple))

            # skip already completed grid-points
            if self._is_gap_fit_out_completed(os.path.join(self.job_dir, 'gap_fit.out'), del_parent_dir=True):
                print('Already completed: {}'.format(self.job_dir))
                continue

            self.run_gap_fit(try_run)
            self.run_quip('validate', try_run)

            # apply GAP on additional systems
            for atoms, atoms_filename in zip(add_run_quip, add_atoms_filename):
                _atoms_filename = self.params_quip['atoms_filename']
                self.params_quip['atoms_filename'] = atoms_filename

                self.atoms_other = atoms
                self.run_quip('other', try_run)

                self.params_quip['atoms_filename'] = _atoms_filename
            self.atoms_other = None

            if del_gp_file:
                [os.remove(os.path.join(self.job_dir, n_file)) for n_file in os.listdir(self.job_dir)
                 if self.params_gap_fit['gp_file'] in n_file]

            if del_at_file:
                os.remove(os.path.join(self.job_dir, self.params_gap_fit['at_file']))

            if del_atoms_filename:
                [os.remove(os.path.join(self.job_dir, n_file))
                 for n_file in [self.params_quip['atoms_filename']] + add_atoms_filename]

            # *.idx can always be removed
            [os.remove(os.path.join(self.job_dir, n_file)) for n_file in os.listdir(self.job_dir)
             if n_file.endswith('.idx')]

        self.job_dir = _job_dir

    @staticmethod
    def _is_gap_fit_out_completed(path_to_gap_fit_out, del_parent_dir=False):
        """Returns True if gap_fit-output-file confirms a completed run else False (and optionally remove parent-dir."""
        if not os.path.exists(path_to_gap_fit_out):
            return False
        else:
            with open(path_to_gap_fit_out, 'r') as o_file:
                last_line = o_file.readlines()[-1]
            if 'Bye-Bye!' in last_line:
                return True
            else:
                if del_parent_dir:
                    shutil.rmtree(os.path.dirname(path_to_gap_fit_out))
                return False

    @staticmethod
    def _dict_cartesian_product(items):
        """Returns the cartesian product of the values' ranges in terms of individual dictionaries."""
        if not items:
            return []
        else:
            return [dict(zip(items.keys(), values)) for values in ito.product(*items.values())]

    def _get_params_tuples(self, gap_fit_ranges, gaps_ranges):
        "Turn value ranges for the arguments keys into tuples of the form (<gap_fit-settings>, <gap_0-settings>, ...)"
        gap_fit_products = self._dict_cartesian_product(gap_fit_ranges)
        gaps_products = [self._dict_cartesian_product(gap_ranges) for gap_ranges in gaps_ranges]
        grid_dimensions = [gap_fit_products] + gaps_products
        return ito.product(*grid_dimensions)

    def _set_params_tuple_values(self, params_tuple):
        "Apply definitions in `params_tuple` to `self.params_gap_fit` and `self.gaps`"
        for key, value in params_tuple[0].items():
            self.params_gap_fit[key] = value

        for gap_idx, gap_ranges in enumerate(params_tuple[1:]):
            for key, value in gap_ranges.items():
                self.gaps[gap_idx][key] = value

    def _params_tuple_to_dir_name(self, params_tuple):
        "Turn `params_tuple` into a string with a key-value pair separated by 2*'_' individual key-value pairs by 3*'_'"
        dir_name = ''

        for key, value in params_tuple[0].items():
            if key == 'default_sigma':
                dir_name = '_'.join([dir_name, '_', key, '', '_'.join([format(ds, '.2E') for ds in value])])
            else:
                dir_name = '_'.join([dir_name, '_', key, '', str(value)])

        for gap_idx, gap_ranges in enumerate(params_tuple[1:]):
            for key, value in gap_ranges.items():
                dir_name = '_'.join([dir_name, '_', key, '', str(value)])

        return dir_name[3:]

    def _params_tuple_to_dataframe(self, params_tuple):
        "Turn `params_tuple` into a DataFrame with each key labeling a column and the row storing the values."
        df = pd.DataFrame()

        for key, value in params_tuple[0].items():
            if key == 'default_sigma':
                for suffix, val in zip(['energies', 'forces', 'virials', 'hessians'], value):
                    df[key+'_'+suffix] = [val]
            else:
                df[key] = [value]

        for gap_idx, gap_ranges in enumerate(params_tuple[1:]):
            for key, value in gap_ranges.items():
                df['_'.join(['gap', str(gap_idx), key])] = [value]

        return df

    def run_crossval(self, gap_fit_ranges, gaps_ranges, subsets, add_run_quip=None, add_atoms_filename=None,
                     del_gp_file=True, try_run=False, omnipresent=None, run_only_subsets=None, del_at_file=False,
                     del_atoms_filename=False):
        """
        Perform a Cross-validation creating training-set and validation-set based on the provided data-sets.
        Note: `self.atoms_train` and `self.atoms_validate` will be set to `None` after the Cross-validation.

        Parameters:
        -----------
        gap_fit_ranges : dict
            Stores the keys and the range of values to be sampled.
        gaps_ranges : list (or dict)
            List of dictionaries (or a single dictionary in case
            only a single gap-potential is used).
            Each dictionary stores the keys and the range of values to be sampled.
        subsets: list
            Stores lists of ase-atoms objects with each of the inner lists
            representing a subset used for the Cross-validation.
        add_run_quip : list (N)
            List of lists with each sub-list containing atoms-objects.
        add_atoms_filename : list (N)
            List of filenames corresponding to the N entries of ``add_run_quip``.
            Will be used to update self.params_quip['atoms_filename'] accordingly.
        del_gp_file : boolean
            Allows to remove the (sometimes rather large) ``gp_file``.
        try_run : boolean
            Run in test-mode.
        omnipresent : list or ase-atoms object
            Stores atoms-objects to be present in each
            of the subsets.
        run_only_subset : list or int
            If defined, runs are only performed on subsets with the specified indices/index.
        del_at_file: boolean
            Allows to remove the large number of ``at_file`` xyz-files.
        del_atoms_filename: boolean
            Allows to remove the large number of ``atoms_filename`` xyz-files.
        """
        # TODO:
        #   - TST for add_* and del_*

        # since there is no easy way to do the assignment---without using mutable argument values---we do it explicitly
        if add_run_quip is None:
            add_run_quip = []
        if add_atoms_filename is None:
            add_atoms_filename = []
        if omnipresent is None:
            omnipresent = []

        # convert to list
        omnipresent = [omnipresent] if not isinstance(omnipresent, list) else omnipresent
        if run_only_subsets is None:
            run_only_subsets = range(len(subsets))
        elif isinstance(run_only_subsets, int):
            run_only_subsets = [run_only_subsets]

        bak_job_dir = self.job_dir  # store attribute and reset later again to that value

        for idx in range(len(subsets)):
            if idx in run_only_subsets:
                # assign validation- and training-sets
                subsets_copy = copy.deepcopy(subsets)
                self.atoms_validate = subsets_copy.pop(idx) + omnipresent  # one subset for validation
                self.atoms_train = list(ito.chain(*subsets_copy)) + omnipresent  # the remaining subsets for training

                # each sub-validation of the Cross-validation gets its one directory
                self.job_dir = os.path.join(bak_job_dir, str(idx)+'_crossval')

                # perform the grid search for hyperparameters
                self.run_sample_grid(gap_fit_ranges=gap_fit_ranges,
                                     gaps_ranges=gaps_ranges,
                                     add_run_quip=add_run_quip,
                                     add_atoms_filename=add_atoms_filename,
                                     del_gp_file=del_gp_file,
                                     try_run=try_run,
                                     del_at_file=del_at_file,
                                     del_atoms_filename=del_atoms_filename)

        # set attributes to defined values
        self.job_dir = bak_job_dir
        self.atoms_train = None
        self.atoms_validate = None

    def get_subsets_random(self, data, num, seed, omnipresent=[]):
        """
        Separate a list into sub-lists by random selection.

        Parameters:
        -----------
        data : list
            Storages the entire data-set to be separated.
        num: int
            Number of subsets to be generated.
        seed : int
            Seed from the random number generator.
        omnipresent : list
            Stores entries to be present in each
            of the subsets.

        Returns:
        --------
        subsets : list
            List of lists with each of the inner ones
            representing a subset of the training-set.
        """
        subsets = []
        size = len(data)//num  # the rest will later be assigned equally to the num sets
        # populate `num` subsets, by picking out entries from `data`,
        # i.e. len(data) gets reduced in each iteration
        for idx in range(num):
            subset, data = self.separate_random_uniform(data, size, seed)

            if omnipresent:
                subset = omnipresent + subset  # `omnipresent` will appear as the first entries

            subsets.append(subset)
        # assign the remainig entries in `data` to the subsets
        for idx in range(len(data)):
            subsets[idx].append(data[idx])
        return subsets

    def separate_random_uniform(self, init_set, subset_size, seed):
        """
        Separates a list into two by selecting entries in a random-uniform manner.

        Parameters:
        -----------
        init_set : list
            Original, full list to be split.
        subset_size : int
            Number of entries to be removed from
            the original list (``init_set``)
            and stored in a separate list (``subset``).
        seed : int
            Seed from the random number generator.

        Returns:
        --------
        subset : list
            List of length ``subset_size`` storing the entries
            extracted from ``init_set``.
        init_set_red : list
            List of length len(init_set)-``subset_size`` containing
            the remaining entries.
        """
        random.seed(a=seed)
        indices_vs = random.sample(range(len(init_set)), subset_size)
        subset = [entry for idx, entry in enumerate(init_set) if idx in indices_vs]
        init_set_red = [entry for idx, entry in enumerate(init_set) if idx not in indices_vs]
        return subset, init_set_red

    def eval_grid(self, gap_fit_ranges, gaps_ranges, key_true, key_pred, info_or_arrays='info', destination='',
                  job_dir='', outfile_quip=''):
        """
        Extract metrics for the prediction-errors and the corresponding parameters of the models sampled on the grid.

        Parameters:
        -----------
        gap_fit_ranges : dict
            Stores the keys and the range of values
            for which the metrics will be extracted.
        gaps_ranges : list (or dict)
            List of dictionaries (or a single dictionary in case
            only a single gap-potential was used).
            Each dictionary stores the keys and the range of values
            for which the metrics will be extracted.
        key_true : string
            Identifier for the reference value within an ase-object.
        key_pred : string
            Identifier for the prediction value within an ase-object.
        info_or_arrays : string
            Are the values stored in ase-object's `info`- or `arrays`-dict.
        destination : string, optional
            Location of file the extracted data will be written to.
            If not specified no file will be written.
            If it ends with `.h5` it will be written in HDF5 format
            (which can be read in again).
            If it ends with `.txt` it will be written in human readable
            format (but cannot be read in again).
            If it ends with `.both` two files will be written, one for
            each of the upper suffixes.
        job_dir : string, optional
            Path to the directory containing the sub-directories created
            during `run_sample_grid()`.
            Defaults to `self.job_dir` if not specified.
        outfile_quip : string, optional
            Name of the xyz-file created by the quip command during evaluation.
            Defaults to the `atoms_filename` specified in `self.params_quip`
            with the prefix `quip_`.

        Returns:
        --------
        results : pandas DataFrame
            Stores the extracted data. Each row represents a model
            while each column represents either a model-parameter
            or an error-metric (e.g. RMSE) achieved by the model.
            The column ordering is `gap_fit`-, `gaps`- and
            then `error-metrics`.
        """
        if not isinstance(gaps_ranges, list):
            gaps_ranges = [gaps_ranges]

        results = pd.DataFrame()

        # try to assign defaults if arguments have not been specified explicitly
        job_dir = job_dir if job_dir else self.job_dir
        outfile_quip = outfile_quip if outfile_quip else 'quip_' + self.params_quip['atoms_filename']

        for params_tuple in self._get_params_tuples(gap_fit_ranges, gaps_ranges):

            # initialize dataframe with parameter settings
            result_single = self._params_tuple_to_dataframe(params_tuple)

            # add values for some metrics (e.g. RMSE) based predictions
            if info_or_arrays == 'info':
                true_n_pred = mltools.misc.get_info(
                        p_xyz_file=os.path.join(job_dir, self._params_tuple_to_dir_name(params_tuple), outfile_quip),
                        keys=[key_true, key_pred])
                for metric in self._metrics:
                    result_single[metric] = getattr(self, 'get_' + metric.lower())(true_n_pred[key_true],
                                                                                   true_n_pred[key_pred])
            elif info_or_arrays == 'arrays':
                true_n_pred = mltools.misc.get_arrays(
                    p_xyz_file=os.path.join(job_dir, self._params_tuple_to_dir_name(params_tuple), outfile_quip),
                    keys=[key_true, key_pred])
                for metric in self._metrics:
                    result_single[metric] = getattr(self, 'get_'+metric.lower())(true_n_pred[key_true].flatten(),
                                                                                 true_n_pred[key_pred].flatten())

            results = pd.concat([results, result_single], ignore_index=True)

        # clean up columns with same value everywhere (e.g. some of the 'default_sigma_*' columns)
        for column in results.columns:
            if len(np.unique(results[column])) == 1:
                results = results.drop(columns=column)

        if destination:
            self.write_dataframe(results, destination)

        return results

    def eval_crossval(self, gap_fit_ranges, gaps_ranges, num, key_true, key_pred, info_or_arrays='info', destination='',
                      job_dir='', outfile_quip=''):
        """
        Extract metrics for the prediction-errors and the corresponding parameters of the models
        sampled during Cross-validation.

        Parameters:
        -----------
        gap_fit_ranges : dict
            Stores the keys and the range of values
            for which the metrics will be extracted.
        gaps_ranges : list (or dict)
            List of dictionaries (or a single dictionary in case
            only a single gap-potential was used).
            Each dictionary stores the keys and the range of values
            for which the metrics will be extracted.
        num: int
            Number of subsets that have been generated.
        key_true : string
            Identifier for the reference value within an ase-object.
        key_pred : string
            Identifier for the prediction value within an ase-object.
        info_or_arrays : string
            Are the values stored in ase-object's `info`- or `arrays`-dict.
        destination : string, optional
            Location of file the extracted data will be written to.
            If not specified no file will be written.
            If it ends with `.h5` it will be written in HDF5 format
            (which can be read in again).
            If it ends with `.txt` it will be written in human readable
            format (but cannot be read in again).
            If it ends with `.both` two files will be written, one for
            each of the upper suffixes.
        job_dir : string, optional
            Path to the directory containing the sub-directories created
            during `run_Cross-validation()`.
            Defaults to `self.job_dir` if not specified.
        outfile_quip : string, optional
            Name of the xyz-file created by the quip command during evaluation.
            Defaults to the `atoms_filename` specified in `self.params_quip`
            with the prefix `quip_`.

        Returns:
        --------
        results : pandas DataFrame
            Stores the extracted data. Each row represents a model
            while each column represents either a model-parameter
            or an error-metric (e.g. RMSE) achieved by the model.
            The column ordering is `gap_fit`-, `gaps`- and
            then `error-metrics`.
            Each row has an additional index-level specifying
            to which subset the model belongs.
        """
        # try to assign defaults if arguments have not been specified explicitly
        job_dir = job_dir if job_dir else self.job_dir

        results_frames = []  # stores DataFrames to be merged in the end
        keys = []  # top-level index of the final, merged DataFrame

        for idx in range(num):
            job_dir_sub = str(idx)+'_crossval'
            results_frames.append(
                    self.eval_grid(
                        gap_fit_ranges=gap_fit_ranges,
                        gaps_ranges=gaps_ranges,
                        key_true=key_true,
                        key_pred=key_pred,
                        info_or_arrays=info_or_arrays,
                        job_dir=os.path.join(job_dir, job_dir_sub),
                        outfile_quip=outfile_quip)
                    )
            keys.append(job_dir_sub)

        results = pd.concat(results_frames, keys=keys)

        if destination:
            self.write_dataframe(results, destination)

        return results

    def get_rmse(self, y_true, y_pred):
        "Return the RMSE value corresonding to the two data-sets."
        return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))

    def write_dataframe(self, df, destination):
        """
        Write a DataFrame to a file.

        Parameters:
        -----------
        df : pandas DataFrame
            Stores data to be written to file.
        destination : string
            Location of the file the DataFrame will be written to.
            If it ends with `.h5` it will be written in HDF5 format
            (which can be read in again).
            If it ends with `.txt` it will be written in human readable
            format (but cannot be read in again).
            If it ends with `.both` two files will be written, one for
            each of the upper suffixes.
        """
        # if the underlying dir-tree needs to be created
        if os.path.dirname(destination):
            self._make_dirs(os.path.dirname(destination))

        # split `destination` to evaluate the file-extension
        basename, suffix = os.path.splitext(destination)
        suffixes = ['.txt', '.h5'] if suffix == '.both' else [suffix]

        for suffix in suffixes:

            # re-build `destination`
            destination = basename+suffix

            if suffix == '.txt':
                with open(destination, 'w') as o_file:
                    o_file.write(df.to_string())
            elif suffix == '.h5':
                df.to_hdf(destination, 'df', format='t', data_columns=True)
            else:
                raise ValueError('Accepted file-extensions are \'.txt\', \'.h5\' or \'.both\'')

    def read_dataframe(self, source):
        "Read in a DataFrame stored in a file with HDF5 format and return it."
        return pd.read_hdf(source, 'df')

    def get_grid_hyparams(self, df, **kwargs):
        """
        Selects hyperparameters from grid-search based on some (user-defined) criterion.

        Parameters:
        -----------
        df : pandas DataFrame
            Stores in the information from grid search
            in the form as given via the `eval_grid_info()`-function.
        criterion : string, optional
            Criterion for the hyperparameter selection. Default is
            `RMSE_min` selecting the hyperparameters yielding the lowest RMSE.
            `RMSE_min_x_max` will (in combination with `tolerance`) define
            a range of tolerable RMSE values. From this range
            the combination of hyperparameters will be selected for which
            the first one (`x`) is maximized. This is e.g. desired for
            x being the energetic `default_sigma` since high values provide
            better generalization of the model.
        tolerance : float, optional
            Factor defining the range of tolerable RMSE values (in combination
            with `criterion` = `RMSE_min_x_max`). RMSE-values up to
            `min(RMSE)*tolerance` will be used for the hyperparameter selection.
        x : string, optional
            Defining the `x` for `RMSE_min_x_max`.

        Returns:
        --------
        A list of tuples with each tuple storing the name of the hyperparameter
        and its corresponding selected value.
        """
        # NOTE: currently assumes exactly 2 hyperparameters
        # extension to infinite dimensions should be straight-forward
        criterion = kwargs.pop('criterion', 'RMSE_min')

        keys = [col for col in df.columns]  # RMSE and names of the hyperparameters

        # apply `criterion` to find hyperparameters values
        arg_min = getattr(self, '_'.join(['_idxmin', criterion]))(df, **kwargs)
        return {key: float(df[key][arg_min]) for key in keys}

    def get_crossval_hyparams(self, df, **kwargs):
        """
        Selects hyperparameters from Cross-validation based on some (user-defined) criterion.

        Parameters:
        -----------
        df : pandas DataFrame
            Stores in the information from the Cross-validation
            in the form as given via the `eval_crossval()`-function.
        criterion : string, optional
            Criterion for the hyperparameter selection. Default is
            `RMSE_min` selecting the hyperparameters yielding the lowest RMSE.
            `RMSE_min_x_max` will (in combinatin with `tolerance`) define
            a range of tolerable RMSE values. From this range
            the combination of hyperparameters will be selected for which
            the first one (`x`) is maximized. This is e.g. desired for
            x being the energetic `default_sigma` since high values provide
            better generalization of the model.
        tolerance : float, optional
            Factor defining the range of tolerable RMSE values (in combination
            with `criterion` = `RMSE_min_x_max`). RMSE-values up to
            `min(RMSE)*tolerance` will be used for the hyperparameter selection.
        x : string, optional
            Defining the `x` for `RMSE_min_x_max`.

        Returns:
        --------
        A list of tuples with each tuple storing the name of the hyperparameter
        and its corresponding selected value.
        """
        # NOTE: currently assumes exactly 2 hyperparameters
        # extension to infinite dimensions should be straight-forward
        criterion = kwargs.pop('criterion', 'RMSE_min')

        levels = np.unique(df.index.get_level_values(0))
        keys = [col for col in df.loc[levels[0]]]  # RMSE and names of the hyperparameters

        # average RMSE-hypersurface
        vals_RMSE = np.zeros((len(df.loc[levels[0]]), len(levels)))  # store RMSE-surfaces of all subsets
        for idx, level in enumerate(levels):
            vals_RMSE[:, idx] = df.loc[level]['RMSE']
        vals_RMSE_mean = np.mean(vals_RMSE, axis=1)

        # create dataframe for averaged hypersurface
        df_average = df.loc[levels[0]]
        df_average['RMSE'] = vals_RMSE_mean

        # apply `criterion` to find hyperparameters values
        arg_min = getattr(self, '_'.join(['_idxmin', criterion]))(df_average, **kwargs)
        return {key: float(df_average[key][arg_min]) for key in keys}

    @staticmethod
    def _idxmin_RMSE_min(df):
        """Return the dataframe's row-index with the lowest RMSE."""
        return df['RMSE'].idxmin()

    @staticmethod
    def _idxmin_RMSE_min_x_max(df, x, tolerance):
        """Return the dataframe's row-index with the lowest RMSE (within a tolerance) while maximizing `x`."""
        # extract data
        vals_RMSE = df['RMSE']
        vals_x = df[x]

        val_RMSE_max = vals_RMSE.min() * (1 + tolerance)  # upper limit for permitted z-values

        # filter x-y-z combinations with z-values within tolerance
        vals_in_tol_x = vals_x[vals_RMSE <= val_RMSE_max]
        vals_in_tol_RMSE = vals_RMSE[vals_RMSE <= val_RMSE_max]

        # select the x-y-z combinations with maximum x-value (there are multiple possibilities containing that x-value)
        # (x is e.g. the energetic default_sigma for which large values are desired for good generalization)
        vals_in_tol_max_x_RMSE = vals_in_tol_RMSE[vals_in_tol_x == vals_in_tol_x.max()]

        # select the x-y-z combination with the minimum RMSE
        return vals_in_tol_max_x_RMSE.idxmin()

    @staticmethod
    def _idxmin_RMSE_min_x_max_y_min(df, x, y, tolerance_RMSE, tolerance_x):
        """Return the dataframe's row-index with the lowest RMSE (within a tolerance) while maximizing `x`."""
        # extract data
        vals_RMSE = df['RMSE']
        vals_x = df[x]
        vals_y = df[y]

        val_RMSE_max = vals_RMSE.min() * tolerance_RMSE  # upper limit for permitted z-values

        # filter x-y-z combinations with z-values within tolerance
        vals_in_tol_x = vals_x[vals_RMSE <= val_RMSE_max]
        vals_in_tol_y = vals_y[vals_RMSE <= val_RMSE_max]
        vals_in_tol_RMSE = vals_RMSE[vals_RMSE <= val_RMSE_max]

        # filter x-y-z combinations with x-values within tolerance
        # (x is e.g. the energetic default_sigma for which large values are desired for good generalization)
        vals_in_tol_max_x_y = vals_in_tol_y[vals_in_tol_x >= vals_in_tol_x.max() * (1 - tolerance_x)]
        vals_in_tol_max_x_RMSE = vals_in_tol_RMSE[vals_in_tol_x >= vals_in_tol_x.max() * (1 - tolerance_x)]

        # select the x-y-z combinations with maximum x-value within tolerance and minimium y-value
        # (there are multiple possibilities containing that y-value)
        # (y is e.g. the force default_sigma for which small values can be desired for good generalization)
        vals_in_tol_max_x_min_y_RMSE = vals_in_tol_max_x_RMSE[vals_in_tol_max_x_y == vals_in_tol_max_x_y.min()]

        # select the x-y-z combination with the minimum RMSE
        return vals_in_tol_max_x_min_y_RMSE.idxmin()

            # extract data
            z_vals = df_sub.pop('RMSE')

            cols = df_sub.columns
            if len(cols) > 2:
                raise ValueError('Dimension missmatch. Found more than two possibilities for x- and y-axis.')
            x_vals = df_sub[cols[0]]
            y_vals = df_sub[cols[1]]

            arg_min = z_vals.idxmin
            x_val_min = x_vals[arg_min]
            y_val_min = y_vals[arg_min]
            z_val_min = z_vals[arg_min]

            # print some data (before showing the plot,
            # thus the user will see both simulatneously)
            msg = '\nOptimum values {0}:\n'.format(level)
            msg += '-'*(len(msg)-2) + '\n'
            msg += '{0:<30} : {1:>20}\n'.format(x_vals.name, x_val_min)
            msg += '{0:<30} : {1:>20}\n'.format(y_vals.name, y_val_min)
            msg += '{0:<30} : {1:>20}\n'.format(z_vals.name, z_val_min)
            print(msg)

            # plot data
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.set_xlabel(x_vals.name)
            ax.set_ylabel(y_vals.name)
            ax.set_zlabel(z_vals.name)

            for idx, log_i in enumerate(log):
                if idx == 0 and log_i:
                    x_vals = np.log10(x_vals)
                    x_val_min = np.log10(x_val_min)
                if idx == 1 and log_i:
                    y_vals = np.log10(y_vals)
                    y_val_min = np.log10(y_val_min)
                if idx == 2 and log_i:
                    z_vals = np.log10(z_vals)
                    z_val_min = np.log10(z_val_min)

            ax.plot_trisurf(x_vals, y_vals, z_vals, cmap=cm.jet, linewidth=0.2)

            ax.plot([x_val_min], [y_val_min], [z_val_min], 'ro')

            plt.show()

            # store for calculating mean values later
            for idx in enumerate(log):
                if idx == 0 and log[idx]:
                    x_vals = 10**x_vals
                    x_val_min = 10**x_val_min
                if idx == 1 and log[idx]:
                    y_vals = 10**y_vals
                    y_val_min = 10**y_val_min
                if idx == 2 and log[idx]:
                    z_vals = 10**z_vals
                    z_val_min = 10**z_val_min

            x_val_mins.append(x_val_min)
            y_val_mins.append(y_val_min)

        # print mean values
        x_val_mean = np.mean(x_val_mins)
        y_val_mean = np.mean(y_val_mins)
        msg = '\nMean of optimum values:\n'
        msg += '-'*(len(msg)-2) + '\n'
        msg += '{0:<20} : {1:>10}\n'.format(x_vals.name, x_val_mean)
        msg += '{0:<20} : {1:>10}\n'.format(y_vals.name, y_val_mean)
        print(msg)

    def view_correlation(self, sources, key_true, key_pred):
        # TODO: improve plot settings
        if not isinstance(sources, list):
            sources = [sources]

        fig, ax = plt.subplots()
        for source in sources:
            y_true_n_pred = mltools.misc.get_info(source, [key_true, key_pred])
            y_true, y_pred = y_true_n_pred[key_true], y_true_n_pred[key_pred]

            val_all = y_true + y_pred
            val_range = [np.min(val_all), np.max(val_all)]

            plt.plot(val_range, val_range, c='k')

            label = 'RMSE : {:>.4}'.format(self.get_rmse(y_true, y_pred))
            if len(sources) != 1:
                plt.scatter(y_true, y_pred, label=label)
                plt.legend()
            else:
                plt.scatter(y_true, y_pred)
                plt.text(0.15, 0.95,
                         s = label,
                         fontsize = 12,
                         ha = 'center', va = 'center',
                         transform = ax.transAxes)
        plt.show()

    def run_local_optimization(self, init_params_gap_fit, init_gaps, key_true, key_pred, method='l-bfgs-b', options={}, del_gp_file=True):
        # l-bfgs-b since allows for boundary conditions
        # NOTE: this method is currently in an alpha-state
        # TODO: key_true/key_pred works only for energies (or one single scalar) so far, not for a loss function with E, F, ... contributions, nor for forces alone
        if not isinstance(init_gaps, list):
            init_gaps = [init_gaps]

        bak_params_gap_fit = self.params_gap_fit
        bak_gaps = self.gaps

        decoder, hyperparams, bounds = self._init_to_hyper(init_params_gap_fit, init_gaps)  # optimization function requires a single array as input
        opt = scipy.optimize.minimize(self._run_local_optimization, hyperparams, bounds=bounds, args=(decoder, key_true, key_pred), method=method, options=options)
        opt_params_gap_fit, opt_gaps = self._hyper_decode(opt.x, decoder)

        self.params_gap_fit = bak_params_gap_fit
        self.gaps = bak_gaps

        return opt_params_gap_fit, opt_gaps

    def _run_local_optimization(self, hyperparams, decoder, key_true, key_pred, del_gp_file=True):

        self.set_hyper_values(hyperparams, decoder)
        self.run_gap_fit()
        time.sleep(2)  # sleep for 1 second to write the files required for run_quip
        self.run_quip('validate')

        outfile_quip = 'quip_' + self.params_quip['atoms_filename']
        y_true_n_pred = mltools.misc.get_info(os.path.join(self.job_dir, outfile_quip), [key_true, key_pred])
        y_true, y_pred = y_true_n_pred[key_true], y_true_n_pred[key_pred]

        # TODO: Loss-function instead of rmse
        rmse = self.get_rmse(y_true, y_pred)
        print('\nRMSE:      ', rmse, '\n')

        if del_gp_file:
            [os.remove(os.path.join(self.job_dir, n_file)) for n_file in os.listdir(self.job_dir)
             if self.params_gap_fit['gp_file'] in n_file]

        return rmse

    def _init_to_hyper(self, init_params_gap_fit, init_gaps):
        keys, vals, bounds = [], [], []

        for key, (val, boundaries) in sorted(init_params_gap_fit.items()):
            keys.append('_'.join(['gap_fit', key]))
            vals.append(val)
            bounds.append(boundaries)

        for idx, gap in enumerate(init_gaps):
            for key, (val, boundaries) in sorted(gap.items()):
                keys.append('_'.join(['gap', str(idx), key]))
                vals.append(val)
                bounds.append(boundaries)
        return keys, np.array(vals),  bounds

    def set_hyper_values(self, hyperparams, keys):
        "Apply hyperparameters to `self.params_gap_fit` and `self.gaps`"
        # TODO:
        #   - TST
        prefix_gap_fit = 'gap_fit_'
        prefix_gaps = 'gap_'
        ds_mapping = {'energies' : 0, 'forces' : 1, 'virials' : 2, 'hessians' : 3}

        for idx, (key, val) in enumerate(zip(keys, hyperparams)):
            if key.startswith(prefix_gap_fit):
                if 'default_sigma' in key:
                    specifier = key.rsplit('_', 1)[-1]
                    self.params_gap_fit['default_sigma'][ds_mapping[specifier]] = val
                else:
                    self.params_gap_fit[key.split('_', 1)[-1]] = val
            if key.startswith(prefix_gaps):
                gap_idx, _key = key.split('_', 2)[1:]
                self.gaps[int(gap_idx)][_key] = val

    def _hyper_decode(self, hyperparams, keys):
        # TODO
        prefix_gap_fit = 'gap_fit_'
        prefix_gaps = 'gap_'

        params_gap_fit = {}
        num_gaps = len(np.unique(['_'.join(key.split('_', 2)[:2]) for key in keys if key.startswith(prefix_gaps)]))
        gaps = [{} for gap in range(num_gaps)]

        for idx, (key, val) in enumerate(zip(keys, hyperparams)):
            if key.startswith(prefix_gap_fit):
                params_gap_fit[key.split('_', 1)[-1]] = val
            if key.startswith(prefix_gaps):
                gap_idx, _key = key.split('_', 2)[1:]
                gaps[int(gap_idx)][_key] = val
        return params_gap_fit, gaps

    @staticmethod
    def find_farthest(dist_matrix, seeds, number):
        """
        Find samples farthest apart from each other based on its distance matrix.

        Parameters:
        -----------
        dist_matrix : ndarray (N, N)
            Distance matrix of the underlying samples.
        seeds: int or list/ndarray of integers
            Starting point for searching further distant samples.
        number : int
            Total number of samples to be selected.

        Returns:
        --------
        samples : list
            Contains the indices that have been selected
            as farthest from each other.
        """
        if isinstance(seeds, int):
            israise = True if number <= 1 else False

            samples = [seeds, np.argmax(dist_matrix[seeds])]  # storage for farthest samples (init. with two farthest)

        elif isinstance(seeds, (list, np.ndarray)):
            israise = True if number <= len(seeds) else False

            samples = [sample for sample in seeds]

        if israise:
            raise ValueError('`number` can not be smaller than specified in `seeds`')

        for idx in range(number - len(samples)):
            samples_rem = np.delete(np.arange(len(dist_matrix)), samples)  # get indices of not selected samples

            dists = dist_matrix[samples][:, samples_rem]  # slice distances for selected samples to remaining samples

            dists_min = np.min(dists, axis=0)  # for each remaining sample find closest distance to already selected
            sample_farthest = np.argmax(dists_min)  # select the remaining sample farthest to all selected samples

            samples.append(samples_rem[sample_farthest])
        return samples

    def get_descriptors(self, set_id, desc_str):
        """
        Construct the descriptors for a given set of geometries.

        Parameters:
        -----------
        set_id : string
            Defines which set of geometries to use.
            Must be one of the string stored in _set_ids.
        desc_str : string
            String-representation of the settings
            the will be used to construct the descriptors.

        Returns:
        --------
        descs : list
            Stores the descriptors of the molecules in terms
            of numpy arrays.
        """
        descs = []  # storage for descriptors

        tmp_dir = tempfile.mkdtemp()  # python 2
        for idx, atoms in enumerate(getattr(self, 'atoms_'+set_id)):
            # python 2/3
            if sys.version_info[0] == 2:
                path = os.path.join(tmp_dir, 'atoms.xyz')
                ase.io.write(path, atoms)
                q_atoms = quippy.Atoms(path)
            elif sys.version_info[0] == 3:
                q_atoms = quippy.convert.ase_to_quip(atoms)

            desc = quippy.descriptors.Descriptor(desc_str)
            q_atoms.set_cutoff(desc.cutoff())
            q_atoms.calc_connect()

            if sys.version_info[0] == 2:
                descs.append(desc.calc(q_atoms)['descriptor'])  # calc. descriptor and append to store
                os.remove(path+'.idx')  # in order to re-use same atoms.xyz file, we need to remove the *.idx file
            elif sys.version_info[0] == 3:
                descs.append(desc.calc_descriptor(q_atoms))  # calc. descriptor and append to store

        shutil.rmtree(tmp_dir)
        return descs

    def calc_average_kernel_soap(self, desc_A, desc_B, zeta=2.0, C_AA=None, C_BB=None, local_kernel=np.dot):
        """
        Calculate the average kernel between two molecules/descriptors.

        Parameters:
        -----------
        desc_A/B : ndarray
            Descriptor representation of the molecules A/B.
        zeta : float, optional
            Specifies the sensitivity used to construct the descriptors.
        C_AA/BB : float
            Not normalized value of the average kernel.
            This value will be used for normalization of the final value.
        local_kernel : function, optional
            Specifies the operation used to calculate the kernel.
            The default builds the dot-product between individual descriptors.

        Returns:
        --------
        C_ij : float
            The value of the average kernel between molecule A and B.
        """
        # explicitely calculate values required for normalization
        if C_AA == None:
            C_AA = self._calc_average_kernel_soap_wo_C_AA_normalization(desc_A, desc_A, zeta, local_kernel=np.dot)
        if C_BB == None:
            C_BB = self._calc_average_kernel_soap_wo_C_AA_normalization(desc_B, desc_B, zeta, local_kernel=np.dot)

        # backups; might be wrong; normalization?
        # C_AB = np.einsum('im,jm->ij',desc_A, desc_B)
        # return 1.0 / (n_A* n_B) * np.sum(C_AB)
        #
        # return local_kernel(np.mean(desc_A, axis=1), np.mean(desc_B, axis=1))

        C_AB = self._calc_average_kernel_soap_wo_C_AA_normalization(desc_A, desc_B, zeta, local_kernel=np.dot)
        C_ij = 1./np.sqrt(C_AA*C_BB)*C_AB
        return C_ij

    def _calc_average_kernel_soap_wo_C_AA_normalization(self, desc_A, desc_B, zeta, local_kernel=np.dot):
        "Calculate the average kernel between two molecules/descriptors without (complete) normalization."
        C_ABs = []
        n_A = len(desc_A)
        n_B = len(desc_B)
        for comb in ito.product(desc_A, desc_B):
            k_AB = local_kernel(comb[0], comb[1])
            k_AA = local_kernel(comb[0], comb[0])
            k_BB = local_kernel(comb[1], comb[1])

            C_AB = (k_AB/np.sqrt(k_AA*k_BB))**zeta
            C_ABs.append(C_AB)
        return 1./(n_A*n_B)*np.sum(C_ABs)

    def calc_kernel_matrix_soap(self, descriptors, ncores=1, destination='', header='', zeta=2.0, local_kernel=np.dot,
                                calc_diag=False, verbose=False):
        """
        Construct the kernel-matrix for a set of molecules/descriptors.

        The SOAP-kernel for a molecule consists of individual entries for each atom.
        In order to compare entire molecules, the function construct the average kernel
        for each molecule.

         Note: Indices with capital letters (e.g. C_AA) refer to un-normalized kernel-values,
               while indices with lower letters (e.g. C_ii) refer to kernel-values normalized
               via the corresponding C_AA values.

        Parameters:
        -----------
        descriptors : list (N)
            Stores the descriptors for each of the N molecules.
        ncores : int
            Number of cores to be used for parallelization.
        destination : string, optional
            If given the kernel matrix will be written
            to the specified location.
        header : string, optional
            If given the specified header will be added
            to the file that stores the kernel matrix.
        zeta : float, optional
            Specifies the sensitivity used to construct the descriptors.
        local_kernel : function, optional
            Specifies the operation used to calculate the kernel.
            The default builds the dot-product between individual descriptors.
        calc_diag : boolean, optional
            If set to `True` the diagonal elements will be calculated
            specifically. Otherwise, the elements will be assigned to
            a value of one.
        verbose : boolean, optional
            Print the progess in generating the kernel matrix.

        Returns:
        --------
        C : ndarray (N, N)
            The kernel matrix representing the similarities
            between individual molecules.
        """
        num = len(descriptors)
        C = np.zeros((num, num))

        # required for normalization
        C_AAs = []
        for idx_A in range(num):
            C_AA = self._calc_average_kernel_soap_wo_C_AA_normalization(descriptors[idx_A], descriptors[idx_A],
                                                                        zeta=zeta, local_kernel=local_kernel)
            C_AAs.append(C_AA)

            if verbose and (idx_A+1)%10 == 0 or (idx_A+1) == num:
                msg = '{0} Kernel element (diagonal) :     {1}/{2} (for normalization)'.format(
                    str(datetime.datetime.now()), idx_A+1, num)
                print(msg)

        # fill the lower triangle (without the diagonal) with (normalized) kernel values.
        #  Note: the case ncores == 1 is treated separately without any use of multiprocessing.
        if ncores != 1:
            mp_out = mp.Queue()
            processes = []
        else:
            mp_out = None

        current = 0
        total = int((num**2 - num) / 2.)
        for i in range(1, num):

            if ncores != 1:
                processes.append(
                        mp.Process(
                            target=self.calc_kernel_matrix_soap_row,
                            args=(i, descriptors, C_AAs, mp_out, zeta, local_kernel)
                            )
                        )

                if (len(processes) == ncores) or i == num - 1:
                    # run processes
                    for p in processes:
                        p.start()

                    # exit completed processes
                    # for p in processes:
                    #     p.join()
                    # NOTE: seems like join is not necessary and even makes the processes dying.
                    #       Tested it without join and found identical results as without any parallelization

                    row_i_C_ijs = [mp_out.get() for p in processes]
                    for row_i, C_ijs in row_i_C_ijs:
                        C[row_i, :row_i] = C_ijs

                    processes = []

                    if verbose:
                        num_row_batch = (num - 1) % ncores if i == num - 1 else ncores
                        current += i * num_row_batch - sum([k for k in range(num_row_batch)])
                        msg = '{0} Kernel element (off-diagonal) : {1}/{2} ({3:.4f} %)'.format(
                            str(datetime.datetime.now()), current, total, 100 * float(current) / total)
                        print(msg)
            else:
                C[i, :i] = self.calc_kernel_matrix_soap_row(i, descriptors, C_AAs, mp_out, zeta, local_kernel)

                if verbose:
                    num_row_batch = (num - 1) % ncores if i == num - 1 else ncores
                    current += i * num_row_batch - sum([k for k in range(num_row_batch)])
                    msg = '{0} Kernel element (off-diagonal) : {1}/{2} ({3:.4f} %)'.format(
                        str(datetime.datetime.now()), current, total, 100 * float(current) / total)
                    print(msg)

        # fill the diagonal elements
        if calc_diag:
            diag = []
            for i in range(num):
                C_ii = self.calc_average_kernel_soap(descriptors[i], descriptors[i], zeta=zeta,
                                                     local_kernel=local_kernel)
                diag.append(C_ii / np.sqrt(C_ii * C_ii))

                if verbose and (i + 1) % 10 == 0:
                    msg = '{0} Kernel element (diagonal) :     {1}/{2}'.format(str(datetime.datetime.now()), i + 1, num)
                    print(msg)

            diag = np.diag(diag)
        else:
            diag = np.diag([1] * num)

        # Construct the full (symmetric) kernel matrix
        C = C + C.T + diag

        # save matrix
        if destination:
            np.savetxt(destination, C, header=header)

        return C

    def calc_kernel_matrix_soap_row(self, i, descriptors, C_AAs, mp_out=None, zeta=2.0, local_kernel=np.dot):
        """Calculates the (unique) elements for row `i` of the (SOAP) kernel matrix"""
        C_i = np.zeros(i)
        for j in range(i):
                C_i[j] = self.calc_average_kernel_soap(descriptors[i], descriptors[j], zeta=zeta, C_AA=C_AAs[i],
                                                       C_BB=C_AAs[j], local_kernel=local_kernel)
        if mp_out is not None:
            mp_out.put((i, C_i))
        else:
            return C_i

    def calc_distance_matrix(self, C, destination='', header=''):
        """
        Construct the distance-matrix from a given kernel-matrix.

        Parameters:
        -----------
        C : ndarray (N, N)
            The kernel matrix representing the similarities
            between individual elements.
        destination : string, optional
            If given the kernel matrix will be written
            to the specified location.
        header : string, optional
            If given the specified header will be added
            to the file that stores the kernel matrix.

        Returns:
        --------
        D : ndarray (N, N)
            The distance matrix between individual elements.
        """
        dim = len(C)
        D = np.zeros(C.shape)

        # fill the lower triangle (without the diagonal)
        for i in range(1, dim):
            for j in range(i):
                D[i, j] = self.calc_distance_element(C[i, i], C[j, j], C[i, j])

        # Construct the full (symmetric) distance matrix
        # Diagonal elements are zero
        D = D + D.T

        # save matrix
        if destination:
            np.savetxt(destination, D, header=header)

        return D

    @staticmethod
    def calc_distance_element(K_ii, K_jj, K_ij):
        """Distance is calculated as self-similarities minus cross-similarity."""
        radicand = K_ii+K_jj - 2*K_ij
        if radicand >= 0:
            return np.sqrt(radicand)
        elif radicand < 0 and np.isclose(radicand, 0.0):
            return 0
        else:
            msg = 'Invalid radicand \'K_ii+K_jj - 2*K_ij = {0}\'!'.format(radicand)
            raise ValueError(msg)

    def get_subsets_farthest(self, data, num, dist_matrix, omnipresent=None):
        """
        Separate a list into sub-lists by farthest-point selection.

        Parameters:
        -----------
        data : list (N)
            Stores the entire data-set to be separated.
        num: int
            Number of subsets to be generated.
        D : ndarray (N, N)
            The distance matrix between individual elements (in `data`).
        omnipresent : list
            Stores entries to be present in each
            of the subsets.

        Returns:
        --------
        subsets : list
            List of lists with each of the inner ones
            representing a subset of the training-set.
        """
        if omnipresent is None:
            omnipresent = []

        # initialize subset_samples and subsets with first `num` data-points
        subset_samples = [[idx] for idx in range(num)]
        subsets = [omnipresent + [data[idx]] for idx in range(num)]

        # continue distributing remaining samples on subsets
        sample_i = num + 1  # first `num` samples have already been assigned
        while True:
            for idx in range(num):

                # check if all samples have been distributed
                if sample_i > len(data):
                    return subsets

                # already assigned samples need to be removed from dist_matrix
                all_foreign_samples = list(ito.chain(*[subset_samples[idx_s] for idx_s in range(num)
                                                       if not idx_s == idx]))

                # mapper for indices in reduced space to indices in full space
                # red2full_mapper[i] holds the value of the corresponding index in `full`
                red2full_mapper = np.delete(np.arange(len(data)), all_foreign_samples)

                # map to reduced space
                dist_matrix_red = np.delete(dist_matrix, all_foreign_samples, axis=0)      # reduce for rows
                dist_matrix_red = np.delete(dist_matrix_red, all_foreign_samples, axis=1)  # reduced for columns
                # map samples/indices of full space to reduced space
                subset_samples_red = np.where(np.isin(red2full_mapper, subset_samples[idx]) is True)[0]

                # find farthest in reduced space
                subset_samples_red = self.find_farthest(dist_matrix_red, subset_samples_red, len(subset_samples_red)+1)

                # map to full space
                subset_samples_idx = red2full_mapper[subset_samples_red]
                if not np.array_equal(np.sort(subset_samples[idx]), np.sort(subset_samples_idx[:-1])):
                    msg = 'Debug: indices handling is wrong.'
                    raise ValueError(msg)
                subset_samples[idx] = list(subset_samples_idx)
                subsets[idx].append(data[subset_samples_idx[-1]])

                sample_i += 1  # increment for next iteration
