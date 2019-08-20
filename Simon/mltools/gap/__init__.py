import os
import copy
import subprocess
import itertools as ito
import numpy as np
import random

import ase.io


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
            setattr(self, 'atoms_'+set_id, getattr(self, 'atoms_'+set_id) + ase.io.read(path, index=':'))
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

    # dumping parameters
    def _dict_to_string(self, items):
        keys = sorted(items)
        return 'dict(' + ',\n     '.join('{0}={1}'.format(key, items[key]) for key in keys) + ')\n'

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

    def run_sample_grid(self, gap_fit_ranges, gaps_ranges, del_gp_file=True, try_run=False):
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
        del_gp_file : boolean
            Allows to remove the (sometimes rather large) ``gp_file``.
        try_run : boolean
            Run in test-mode.
        """

        if not isinstance(gaps_ranges, list):
            gaps_ranges = [gaps_ranges]
        if not len(gaps_ranges) == len(self.gaps):
            raise ValueError('``gaps_ranges`` must have same length as ``self.gaps``')

        _job_dir = self.job_dir  # used to reset it later again to that value

        teach_sparse_products = self._dict_cartesian_product(teach_sparse_ranges)
        gaps_products = [self._dict_cartesian_product(gap_ranges) for gap_ranges in gaps_ranges]
        grid_dimensions = [teach_sparse_products] + gaps_products
        for params_tuple in ito.product(*grid_dimensions):
            _job_dir_sub = ''

            for key, value in params_tuple[0].items():
                self.params_teach_sparse[key] = value
                if key == 'default_sigma':
                    _job_dir_sub = '_'.join([_job_dir_sub, '', key, '_'.join([format(ds, '.2E') for ds in value])])
                else:
                    _job_dir_sub = '_'.join([_job_dir_sub, '', key, str(value)])

            for gap_idx, gap_ranges in enumerate(params_tuple[1:]):
                for key, value in gap_ranges.items():
                    self.gaps[gap_idx][key] = value
                    _job_dir_sub = '_'.join([_job_dir_sub, '', key, str(value)])

            self.job_dir = os.path.join(_job_dir, _job_dir_sub[2:])
            self.run_teach_sparse(try_run)
            self.run_quip('validate', try_run)
            if del_gp_file:
                [os.remove(os.path.join(self.job_dir, n_file)) for n_file in os.listdir(self.job_dir)
                 if self.params_teach_sparse['gp_file'] in n_file]

        self.job_dir = _job_dir

    def _dict_cartesian_product(self, items):
        "Returns the cartesian product of the values' ranges in terms of individual dictionaries."
        return [dict(zip(items.keys(), values)) for values in ito.product(*items.values())]

    def crossvalidation(self, num, seed, teach_sparse_ranges, gaps_ranges, del_gp_file=True, try_run=False, omnipresent=[]):
        """
        Perform a cross-validation on the trainin-set data.

        Parameters:
        -----------
        num: int
            Number of sub-sets to be generated.
        seed : int
            Seed from the random number generator.
        teach_sparse_ranges : dict
            Stores the keys and the range of values to be sampled.
        gaps_ranges : list (or dict)
            List of dictionaries (or a single dictionary in case
            only a single gap-potential is used).
            Each dictionary stores the keys and the range of values to be sampled.
        del_gp_file : boolean
            Allows to remove the (sometimes rather large) ``gp_file``.
        try_run : boolean
            Run in test-mode.
        omnipresent : list
            Stores atoms-objects to be present in each
            of the sub-sets.
        """
        # storage to reset later again to these values
        _job_dir = self.job_dir
        atoms_train = copy.deepcopy(self.atoms_train)
        # self.atoms_validate might not be initiated
        if hasattr(self, 'atoms_validate'):
            atoms_validate = copy.deepcopy(self.atoms_validate)
            exists_atoms_validate = True
        else:
            exists_atoms_validate = False

        sub_sets = self._get_subsets(num, seed, omnipresent)
        for idx in range(num):
            # assing validation- and training-sets
            sub_sets_copy = copy.deepcopy(sub_sets)
            self.atoms_validate = sub_sets_copy.pop(idx)  # one sub-set for validatoin
            self.atoms_train = list(ito.chain(*sub_sets_copy))  # the remaining sub-sets for training

            # each sub-validation of the crossvalidation gets its one directory
            self.job_dir = os.path.join(_job_dir, str(idx)+'_crossval')

            # perform the grid search for hyperparameters
            self.run_sample_grid(teach_sparse_ranges, gaps_ranges, del_gp_file, try_run)

        self.job_dir = _job_dir
        self.atoms_train = atoms_train
        if exists_atoms_validate:
            self.atoms_validate = atoms_validate
        else:
            self.atoms_validate = None


    def _get_subsets(self, num, seed, omnipresent=[]):
        """
        Separates the trainin-set into sub-sets.

        Parameters:
        -----------
        num: int
            Number of sub-sets to be generated.
        seed : int
            Seed from the random number generator.
        omnipresent : list
            Stores atoms-objects to be present in each
            of the sub-sets.

        Returns:
        --------
        set_subs : list
            List of lists with each of the inner ones
            representing a sub-set of the training-set.
        """
        atoms = copy.deepcopy(self.atoms_train)  # avoid changes in self.atoms_train
        set_subs = []
        size = len(self.atoms_train)//num  # the rest will later be assigned equally to the num sets
        # populate ``num`` sub-sets, by picking out entries from ``atoms``,
        # i.e. len(atoms) gets reduced in each iteration
        for idx in range(num):
            set_sub, atoms = self._separate_random_uniform(atoms, size, seed)

            if omnipresent:
                set_sub = omnipresent + set_sub

            set_subs.append(set_sub)
        # assign the remainig entries in ``atoms`` to the sub-sets
        for idx in range(len(atoms)):
            set_subs[idx].append(atoms[idx])
        return set_subs

    def _separate_random_uniform(self, init_set, subset_size, seed):
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
