import os
import copy
import subprocess

import ase.io


class Gap(object):
    """
    basic usage
        >>> gap = Gap()
        >>> gap.job_dir = '/path/to/dir'
        >>> gap.outfile_teach = 'teach.out'
        >>> gap.params_teach_sparse = {...}
        >>> gap.gaps = [{'name' : ..., ...},
        >>>             {'name' : ..., ...}]
        >>> gap.read_atoms('./path/to/train.xyz', 'train')
        >>> gap.run_teach_sparse()
    """
    def __init__(self):
        self._set_ids = ['train', 'validate', 'holdout']

    # cmd_teach cannot be changed by user directly
    @property
    def cmd_teach(self):
        "Update command string when called"
        self._build_cmd_teach()
        return self._cmd_teach

    # output
    @property
    def job_dir(self):
        return self._job_dir

    @job_dir.setter
    def job_dir(self, path):
        self._job_dir = path

    @property
    def outfile_teach(self):
        return self._outfile_teach

    @outfile_teach.setter
    def outfile_teach(self, filename):
        self._outfile_teach = filename

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
    def atoms_holdout(self):
        return self._atoms_holdout

    @atoms_holdout.setter
    def atoms_holdout(self, atoms):
        self._atoms_holdout = atoms

    # full and direct excess to params-dicts
    @property
    def params_teach_sparse(self):
        return self._params_teach_sparse

    @params_teach_sparse.setter
    def params_teach_sparse(self, params):
        self._params_teach_sparse = params

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

    # checks
    def _check_key(self, items, key):
        "Check if a (required) key is in a dictionary, e.g. ``name`` in ``self.params_teach_sparse``"
        if not key in items:
            msg = 'Key \'{}\' not found.'.format(key)
            raise KeyError(msg)

    def _check_set_id(self, set_id):
        "Check if ``set_id`` is part of ``self._set_ids``"
        if not set_id in self._set_ids:
            msg = '\'set_id\' must be one of \'{}\''.format(' '.join(self._set_ids))
            raise ValueError(msg)

    # atoms handling
    def read_atoms(self, path, set_id):
        """
        Read geometries from file and store as attribute of the instance (self._atoms_<set_id>).

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
        setattr(self, 'atoms_'+set_id, ase.io.read(path, index=':'))

    def write_atoms(self, destination, set_id):
        """
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

        ase.io.write(destination, getattr(self, 'atoms_'+set_id))

    # command handling
    def _build_cmd_teach(self):
        "Builds the teach_sparse command-line string"
        cmd_str = '! teach_sparse '
        cmd_str += self._build_assign_str(self._params_teach_sparse)
        cmd_str += ' '
        cmd_str += self._build_gap_str()
        self._cmd_teach = cmd_str

    def _build_assign_str(self, items):
        "Turns dictionary to a string of the form 'key=val' concatenating the items by a whitespace"
        assign_str = ''
        for key, value in items.items():
            assign_str += '{}={} '.format(key, value)
        return assign_str[:-1]

    def _build_gap_str(self):
        "Builds the gap-related part of the teach_sparse command-line string"
        cmd_str = 'gap={'
        cmd_str += ' :'.join([self._build_potential_str(gap) for gap in self.gaps])
        cmd_str += '}'
        return cmd_str

    def _build_potential_str(self, items):
        "Build the command-line string for a single desciptor within the gap-related part of teach_sparse"
        items_copy = copy.deepcopy(items)  # avoid changes in self.gaps sub-dirs
        pot_str = items_copy.pop('name')
        pot_str += ' '
        pot_str += self._build_assign_str(items_copy)
        return pot_str

    # command execution
    def run_teach_sparse(self, try_run=False):
        """
        Executes the teach_sparse command based on the defined settings in
            self.params_teach_sparse,
            self.gaps,
            self.job_dir,
            self.outfile_teach.

        The training-set (self.atoms_train) will automatically be written to the file
        specified in self.params_teach_sparse ('at_file').

        Standard output and output for error will be written into separated files.
        """
        if try_run:
            print(self.cmd_teach)
        else:
            self._make_job_dir()
            self.write_atoms(os.path.join(self.job_dir, self.params_teach_sparse['at_file']), 'train')

            p_out_std = os.path.join(self.job_dir, self.outfile_teach)
            p_out_err = p_out_std[:-4]+'.err' if p_out_std.endswith('.out') else p_out_std+'.err'

            print(self.cmd_teach)
            os.system('{command} 1>{stdout} 2>{stderr}'.format(command=self.cmd_teach, stdout=p_out_std, stderr=p_out_err))

            # NOTE: Would be more clean to have it via Popen, but Popen cannot handle this rather complex expression
            # process = subprocess.Popen(self._cmd_teach.split(), stdout=subprocess.PIPE)  # stdout to file, stderr to screen
            # while True:
            #     with open(os.path.join(self.job_dir, self.outfile_teach), 'a') as o_file:
            #         out = process.stdout.read(1)
            #         if out == '' and process.poll() != None:
            #             break
            #         if out != '':
            #             o_file.write(out)
            #             o_file.flush()

    def _make_job_dir(self):
        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)
