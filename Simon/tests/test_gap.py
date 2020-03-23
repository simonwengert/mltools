# #!/usr/bin/env python
import unittest
import sys
import os
import tempfile
import shutil
import filecmp
import copy
import numpy as np
import itertools as ito
import pandas as pd

import ase.io

import mltools.gap


class TestParser(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)

    def test__build_assign_str(self):
        # Sorting for python2/3 compatibility
        items = {'key_0' : 'val_0',
                 'key_1' : 1,
                 'key_2' : 1.1,
                 'key_3' : False}
        ref_str_sorted = 'key_0=val_0 key_1=1 key_2=1.1 key_3=False'

        gap = mltools.gap.Gap()
        str_sorted = ' '.join(sorted(gap._build_assign_str(items).split(' ')))
        self.assertEqual(str_sorted, ref_str_sorted)

    def test__build_potential_str(self):
        items = {'name' : 'descriptor',
                   'key_0' : 'val_0'}
        ref_str = 'descriptor key_0=val_0'
        gap = mltools.gap.Gap()
        self.assertEqual(gap._build_potential_str(items), ref_str)

    def test__build_gap_str_single_descriptor(self):
        items = {'name' : 'descriptor',
                   'key_0' : 'val_0'}
        ref_str = 'gap={descriptor key_0=val_0}'
        gap = mltools.gap.Gap()
        gap.gaps = items
        self.assertEqual(gap._build_gap_str(), ref_str)

    def test__build_gap_str_two_descriptors(self):
        items = [{'name' : 'descriptor_0', 'key_0' : 'val_0'},
                 {'name' : 'descriptor_1', 'key_1' : 'val_1'}]
        ref_str = 'gap={descriptor_0 key_0=val_0 :descriptor_1 key_1=val_1}'
        gap = mltools.gap.Gap()
        gap.gaps = items
        self.assertEqual(gap._build_gap_str(), ref_str)

    def test__check_set_id_pass(self):
        set_ids = ['train', 'validate', 'test']
        gap = mltools.gap.Gap()
        for set_id in set_ids:
            gap._check_set_id(set_id)

    def test__check_set_id_fail(self):
        set_id = 'none'
        gap = mltools.gap.Gap()
        with self.assertRaises(ValueError):
            gap._check_set_id(set_id)

    def test_read_atoms_pass(self):
        set_ids = ['train', 'validate', 'test']
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')
        ref_atoms_all = ase.io.read(p_xyz_file, index=':')
        gap = mltools.gap.Gap()

        for set_id in set_ids:
            gap.read_atoms(p_xyz_file, set_id)

        for set_id in set_ids:
            for atoms, ref_atoms in zip(getattr(gap, 'atoms_'+set_id), ref_atoms_all):
                self.assertEqual(np.all(np.isclose(atoms.get_positions(), ref_atoms.get_positions()) == True), True)

    def test_read_atoms_fail(self):
        set_id = 'none'
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')
        gap = mltools.gap.Gap()
        with self.assertRaises(ValueError):
            gap.read_atoms(p_xyz_file, set_id)

    def test_read_atoms_append(self):
        set_ids = ['train', 'validate', 'test']
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')
        ref_atoms_all = 2*ase.io.read(p_xyz_file, index=':')
        ref_len = 2*3
        gap = mltools.gap.Gap()

        for set_id in set_ids:
            gap.read_atoms(p_xyz_file, set_id)
        for set_id in set_ids:
            gap.read_atoms(p_xyz_file, set_id, append=True)

        for set_id in set_ids:
            atoms_all = getattr(gap, 'atoms_'+set_id)
            self.assertEqual(len(atoms_all), ref_len)
            for atoms, ref_atoms in zip(atoms_all, ref_atoms_all):
                self.assertEqual(np.all(np.isclose(atoms.get_positions(), ref_atoms.get_positions()) == True), True)

    def test_write_atoms_pass(self):
        set_ids = ['train', 'validate', 'test']
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')
        atoms_ref = ase.io.read(p_xyz_file)
        gap = mltools.gap.Gap()

        for set_id in set_ids:
            gap.read_atoms(p_xyz_file, set_id)
            gap.write_atoms(set_id, set_id)
            atoms_should = ase.io.read(set_id+'.xyz')

            np.testing.assert_array_equal(atoms_should.get_positions(), atoms_ref.get_positions())

    def test_set_lattices(self):
        length = 42  # of course 42
        ref_cell = np.diag([length]*3)
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')

        gap = mltools.gap.Gap()
        gap.read_atoms(p_xyz_file, 'train')
        gap.set_lattices(length, 'train')
        for atoms in gap.atoms_train:
            np.testing.assert_array_equal(atoms.get_cell(), ref_cell)

    def test__calc_energy_sigmas_linear(self):
        ref_values  = [1, 5, 2, 3, 4]
        sigma_range = [0, 1]
        ref_sigma_energies = [0, 1, 0.25, 0.50, 0.75]

        gap = mltools.gap.Gap()
        np.testing.assert_array_equal(gap._calc_energy_sigmas_linear(ref_values, sigma_range), ref_sigma_energies)

    def test_assign_energy_sigma_linear(self):
        ref_values  = [1, 2, 3]
        sigma_range = [0, 1]
        ref_sigma_energies = [0, 0.50, 1]
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')

        gap = mltools.gap.Gap()
        gap.read_atoms(p_xyz_file, 'train')
        gap.assign_energy_sigma_linear(ref_values, sigma_range)
        for atoms, ref_sigma_energy in zip(gap.atoms_train, ref_sigma_energies):
            self.assertEqual(atoms.info['energy_sigma'], ref_sigma_energy)

    def test_write_gap_fit_parameters(self):
        ref_file = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'gap_fit.params')
        gap = mltools.gap.Gap()
        gap.params_gap_fit = {'key_0' : 'val_0', 'key_1' : 'val_1'}
        gap.gaps = {'name' : 'val_2', 'key_3' : 'val_3'}
        gap.write_gap_fit_parameters()
        self.assertTrue(filecmp.cmp('gap_fit.params', ref_file))

    def test_write_quip_parameters(self):
        ref_file = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'quip.params')
        gap = mltools.gap.Gap()
        gap.params_quip = {'key_0' : 'val_0', 'key_1' : 'val_1'}
        gap.write_quip_parameters()
        self.assertTrue(filecmp.cmp('quip.params', ref_file))

    def test__build_cmd_gap_fit(self):
        ref_cmd_gap_fit = '! gap_fit default_sigma={0 0.5 1 2} key_0=val_0 gap={gap_0 key_1=val_1}'
        gap = mltools.gap.Gap()
        gap.params_gap_fit = {'default_sigma' : [0, 0.5, 1, 2],
                                   'key_0' : 'val_0'}
        gap.gaps = {'name' : 'gap_0',
                    'key_1' : 'val_1'}
        self.assertEqual(gap.cmd_gap_fit, ref_cmd_gap_fit)

    def test__build_cmd_quip(self):
        # Sorting for python2/3 compatibility
        ref_cmd_quip_sorted = '! quip key_0=val_0 key_1=val_1 | grep AT | sed \'s/AT//\''
        gap = mltools.gap.Gap()
        gap.params_quip = {'key_0' : 'val_0',
                           'key_1' : 'val_1'}
        cmd_quip = gap.cmd_quip
        cmd_quip_prefix = ' '.join(cmd_quip.split(' ')[:2]) + ' '
        cmd_quip_inner = ' '.join(sorted(cmd_quip.split(' ')[2:4]))
        cmd_quip_suffix = ' ' + ' '.join(cmd_quip.split(' ')[4:])
        cmd_quip_sorted = cmd_quip_prefix + cmd_quip_inner + cmd_quip_suffix
        self.assertEqual(cmd_quip_sorted, ref_cmd_quip_sorted)

    def test__get_params_tuples(self):
        gap_fit_ranges = {'key_0' : [0, 1, 2], 'key_1' : [0.0, 0.1]}
        gaps_ranges = [{'key_2' : [2, 4, 6], 'key_3' : [0.2, 0.4]}, {'key_4' : [0.5, 1.0, 1.5]}]

        gap_fit_product = [{'key_0' : 0, 'key_1' : 0.0},
                           {'key_0' : 0, 'key_1' : 0.1},
                           {'key_0' : 1, 'key_1' : 0.0},
                           {'key_0' : 1, 'key_1' : 0.1},
                           {'key_0' : 2, 'key_1' : 0.0},
                           {'key_0' : 2, 'key_1' : 0.1}]
        gap_0_product = [{'key_2' : 2, 'key_3' : 0.2},
                         {'key_2' : 2, 'key_3' : 0.4},
                         {'key_2' : 4, 'key_3' : 0.2},
                         {'key_2' : 4, 'key_3' : 0.4},
                         {'key_2' : 6, 'key_3' : 0.2},
                         {'key_2' : 6, 'key_3' : 0.4}]
        gap_1_product = [{'key_4' : 0.5},
                         {'key_4' : 1.0},
                         {'key_4' : 1.5}]
        num_combs = len(gap_fit_product)*len(gap_0_product)*len(gap_1_product)

        # python2/3 requires extensive sorting here since python3 dicts cannot be sorted
        # turn each params_tuple into a string (since strings can be compared) while sorting at the same time
        ref_compr_strs = []
        for tuple_i in ito.product(*[gap_fit_product, gap_0_product, gap_1_product]):
            ref_compr_str = ''
            tuple_i_sorted = sorted(tuple_i, key=lambda x:sorted(x.keys()))
            for dict_i in tuple_i_sorted:
                ref_compr_str += ' '.join(['{} : {}'.format(key, val) for key, val in sorted(dict_i.items())])
                ref_compr_str += ' '
            ref_compr_strs.append(ref_compr_str[:-1])

        gap = mltools.gap.Gap()
        compr_strs = []
        for tuple_i in gap._get_params_tuples(gap_fit_ranges, gaps_ranges):
            compr_str = ''
            tuple_i_sorted = sorted(tuple_i, key=lambda x:sorted(x.keys()))
            for dict_i in tuple_i_sorted:
                compr_str += ' '.join(['{} : {}'.format(key, val) for key, val in sorted(dict_i.items())])
                compr_str += ' '
            compr_strs.append(compr_str[:-1])

        # check for number of combinations for the parameter ranges
        self.assertEqual(
                len([params_tuple_i for params_tuple_i in gap._get_params_tuples(gap_fit_ranges, gaps_ranges)]),
                num_combs)
        # check the combinations explicitly
        for compr_str, ref_compr_str in zip(sorted(compr_strs), sorted(ref_compr_strs)):
            self.assertEqual(compr_str, ref_compr_str)

    def test__set_params_tuple_values(self):
        # `params_tuples` below would be build applying these setting:
        # - gap_fit_ranges = {'key_0' : [0, 1], 'key_1' : [0.0, 0.1]}
        # - gaps_ranges = [{'key_2' : [2, 4], 'key_3' : [0.2, 0.4]}, {'key_4' : [0.5, 1.0]}]
        params_tuples = [({'key_0': 0, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 0, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 1.0}),
                         ({'key_0': 0, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 0, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 1.0}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 1.0}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 0, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.0}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 2, 'key_3': 0.4}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.2}, {'key_4' : 1.0}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 0.5}),
                         ({'key_0': 1, 'key_1': 0.1}, {'key_2': 4, 'key_3': 0.4}, {'key_4' : 1.0})]


        params_gap_fit = {'key_0' : 100, 'key_1' : 200, 'key_5' : 600}
        gaps = [{'name' : 'name_0', 'key_2' : 300, 'key_3' : 400, 'key_6' : 700}, {'name' : 'name_1', 'key_4' : 500}]

        gap = mltools.gap.Gap()
        gap.params_gap_fit = copy.deepcopy(params_gap_fit)
        gap.gaps = copy.deepcopy(gaps)

        for params_tuple in params_tuples:
            gap._set_params_tuple_values(params_tuple)
            params_gap_fit['key_0'] = params_tuple[0]['key_0']
            params_gap_fit['key_1'] = params_tuple[0]['key_1']
            gaps[0]['key_2'] = params_tuple[1]['key_2']
            gaps[0]['key_3'] = params_tuple[1]['key_3']
            gaps[1]['key_4'] = params_tuple[2]['key_4']
            self.assertEqual(gap.params_gap_fit, params_gap_fit)
            self.assertEqual(gap.gaps, gaps)

    def test__params_tuple_to_dir_name(self):
        # Sorting for python2/3 compatibility
        params_tuple = [{'default_sigma' : [0.1, 0, 0, 0], 'key_0' : 0}, {'key_1' : 1}, {'key_2' : 2.0}]
        ref_dir_name_sorted = 'default_sigma__1.00E-01_0.00E+00_0.00E+00_0.00E+00___key_0__0___key_1__1___key_2__2.0'

        gap = mltools.gap.Gap()
        dir_name = gap._params_tuple_to_dir_name(params_tuple)
        dir_name_sorted = '___'.join(sorted(dir_name.split('___')))
        self.assertEqual(dir_name_sorted, ref_dir_name_sorted)


    def test__params_tuple_to_dataframe(self):
        params_tuple = [{'default_sigma' : [0.1, 0.2, 0.3, 0], 'key_0' : 0}, {'key_1' : 1}, {'key_2' : 2.0}]
        df = pd.DataFrame()
        df['default_sigma_energies'] = [0.1]
        df['default_sigma_forces'] = [0.2]
        df['default_sigma_virials'] = [0.3]
        df['default_sigma_hessians'] = [0]
        df['key_0'] = 0
        df['gap_0_key_1'] = [1]
        df['gap_1_key_2'] = [2.0]

        gap = mltools.gap.Gap()
        pd.testing.assert_frame_equal(gap._params_tuple_to_dataframe(params_tuple).sort_index(axis=1), df.sort_index(axis=1))

    def test_get_subsets_random_without_omnipresent(self):
        data = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        num = 3
        seed = 42

        #python2/3; reason is random.sample()
        if sys.version_info[0] == 2:
            ref_subsets = [['a', 'e', 'g'], ['b', 'f'], ['c', 'd']]
        elif sys.version_info[0] == 3:
            ref_subsets = [['a', 'f', 'd'], ['b', 'g'], ['c', 'e']]

        gap = mltools.gap.Gap()
        self.assertEqual(gap.get_subsets_random(data, num, seed), ref_subsets)

    def test_get_subsets_random_with_omnipresent(self):
        data = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        num = 3
        seed = 42
        omnipresent = ['x']

        #python2/3; reason is random.sample()
        if sys.version_info[0] == 2:
            ref_subsets = [['x', 'a', 'e', 'g'], ['x', 'b', 'f'], ['x', 'c', 'd']]
        elif sys.version_info[0] == 3:
            ref_subsets = [['x', 'a', 'f', 'd'], ['x', 'b', 'g'], ['x', 'c', 'e']]

        gap = mltools.gap.Gap()
        self.assertEqual(gap.get_subsets_random(data, num, seed, omnipresent), ref_subsets)

    def test_separate_random_uniform(self):
        init_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        subset_size = 2
        seed = 42

        #python2/3; reason is random.sample()
        if sys.version_info[0] == 2:
            ref_subset, ref_init_set_red = ['a', 'e'], ['b', 'c', 'd', 'f', 'g']
        elif sys.version_info[0] == 3:
            ref_subset, ref_init_set_red = ['a', 'f'], ['b', 'c', 'd', 'e', 'g']

        gap = mltools.gap.Gap()
        self.assertEqual(gap.separate_random_uniform(init_set, subset_size, seed), (ref_subset, ref_init_set_red))

    def test_eval_grid(self):
        gap_fit_ranges = {'default_sigma' : [[0.1, 0, 0, 0], [0.01, 0, 0, 0], [0.001, 0, 0, 0]]}
        gaps_ranges = [{'key_0' : [0, 1, 2]}]
        key_true = 'float_0'
        key_pred = 'float_1'
        info_or_arrays = 'info'
        destination = 'eval_grid.both'
        job_dir = os.path.join(self.cwd, 'tests', 'data', 'tree', '0_crossval')
        outfile_quip = 'quip_out.xyz'

        ref_file_txt = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_grid.txt')
        #python2/3
        if sys.version_info[0] == 2:
            ref_file_h5 = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_grid.python2.h5')
        elif sys.version_info[0] == 3:
            ref_file_h5 = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_grid.python3.h5')

        gap = mltools.gap.Gap()
        gap.eval_grid(gap_fit_ranges, gaps_ranges, key_true, key_pred, info_or_arrays, destination, job_dir, outfile_quip)

        self.assertTrue(filecmp.cmp('eval_grid.txt', ref_file_txt))
        pd.testing.assert_frame_equal(pd.read_hdf('eval_grid.h5'), pd.read_hdf(ref_file_h5))

    def test_eval_crossval(self):
        gap_fit_ranges = {'default_sigma' : [[0.1, 0, 0, 0], [0.01, 0, 0, 0], [0.001, 0, 0, 0]]}
        gaps_ranges = [{'key_0' : [0, 1, 2]}]
        num = 3
        key_true = 'float_0'
        key_pred = 'float_1'
        info_or_arrays = 'info'
        destination = 'eval_crossval.both'
        job_dir = os.path.join(self.cwd, 'tests', 'data', 'tree')
        outfile_quip = 'quip_out.xyz'

        ref_file_txt = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_crossval.txt')
        #python2/3
        if sys.version_info[0] == 2:
            ref_file_h5 = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_crossval.python2.h5')
        elif sys.version_info[0] == 3:
            ref_file_h5 = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_crossval.python3.h5')

        gap = mltools.gap.Gap()
        gap.eval_crossval(gap_fit_ranges, gaps_ranges, num, key_true, key_pred, info_or_arrays, destination, job_dir, outfile_quip)

        self.assertTrue(filecmp.cmp('eval_crossval.txt', ref_file_txt))
        pd.testing.assert_frame_equal(pd.read_hdf('eval_crossval.h5'), pd.read_hdf(ref_file_h5))

    def test_get_crossval_hyparams(self):
        #python2/3
        if sys.version_info[0] == 2:
            file_h5 = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_crossval.python2.h5')
        elif sys.version_info[0] == 3:
            file_h5 = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'eval_crossval.python3.h5')
        ref_means = {'RMSE': 1.9257206443303245, 'default_sigma_energies': 0.10000000000000000, 'gap_0_key_0': 2.0}
        df = pd.read_hdf(file_h5)

        gap = mltools.gap.Gap()
        self.assertEqual(gap.get_crossval_hyparams(df), ref_means)

    def test_find_farthest(self):
        # dists_matrix resembels approximatily:
        # +--------------------+
        # | oo                 | <= sample 0, sample 1
        # |                   o| <= sample 4
        # |                    |
        # |                    |
        # |  o                 | <= sample 3
        # |  o                 | <= sample 2
        # |                    |
        # |                    |
        # +--------------------+
        dists_matrix = np.array([
            [0.0, 0.1, 0.6, 0.5, 0.7],
            [0.1, 0.0, 0.6, 0.5, 0.7],
            [0.6, 0.6, 0.0, 0.2, 0.8],
            [0.5, 0.5, 0.2, 0.0, 0.8],
            [0.7, 0.7, 0.8, 0.8, 0.0]])
        seeds = 0
        number = 5

        gap = mltools.gap.Gap()
        self.assertEqual(gap.find_farthest(dists_matrix, seeds, number), [0, 4, 2, 3, 1])

    def test_get_descriptors_soap(self):
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '1_test.xyz')
        set_id = 'other'
        desc_str = 'soap cutoff=6.0 l_max=8 n_max=8 atom_sigma=0.3 zeta=2 covariance_type=dot_product n_species=2 species_Z={8 1} n_Z=2 Z={8 1}'

        gap = mltools.gap.Gap()
        gap.read_atoms(p_xyz_file, set_id)
        descs = np.asarray(gap.get_descriptors(set_id, desc_str))
        names = ['0_first', '0_second', '0_third']
        for idx, desc_i in enumerate(descs):
            np.testing.assert_array_equal(desc_i, np.loadtxt(os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'descs_{}.txt'.format(names[idx]))))

    def test_calc_average_kernel_soap(self):
        desc_A = np.loadtxt(os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'descs_0_first.txt'))
        desc_B = np.loadtxt(os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'descs_0_second.txt'))
        zeta = 2
        local_kernel = np.dot
        ref_average_kernel_value = 0.9983700867106244

        gap = mltools.gap.Gap()
        self.assertEqual(gap.calc_average_kernel_soap(desc_A, desc_B, zeta, local_kernel=local_kernel), ref_average_kernel_value)

    def test_calc_kernel_matrix_soap(self):
        descriptors = [np.loadtxt(os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'descs_0_{}.txt'.format(name))) for name in ['first', 'second', 'third']]
        calc_diag = True
        local_kernel = np.dot
        zeta = 2
        ref_kernel_matrix = np.array(                                           #  The three H2O molecules differ only in the lenght of one H-O bond.
                [[1.0,                0.9983700867106244, 0.9961005991483849],
                 [0.9983700867106244, 1.0,                0.9994705736118292],
                 [0.9961005991483849, 0.9994705736118292, 1.0]])

        gap = mltools.gap.Gap()
        np.testing.assert_array_equal(gap.calc_kernel_matrix_soap(descriptors, calc_diag=calc_diag, local_kernel=local_kernel, zeta=zeta), ref_kernel_matrix)

    def test_calc_distance_element(self):
        kernel_matrix = np.array(                                           #  The three H2O molecules differ only in the lenght of one H-O bond.
                [[1.0,                0.9983700867106244, 0.9961005991483849],
                 [0.9983700867106244, 1.0,                0.9994705736118292],
                 [0.9961005991483849, 0.9994705736118292, 1.0]])
        ref_D_00 = 0.0
        ref_D_01 = 0.057094891003934174

        gap = mltools.gap.Gap()
        self.assertEqual(gap.calc_distance_element(kernel_matrix[0, 0], kernel_matrix[0, 0], kernel_matrix[0, 0]), ref_D_00)
        self.assertEqual(gap.calc_distance_element(kernel_matrix[0, 0], kernel_matrix[1, 1], kernel_matrix[0, 1]), ref_D_01)

    def test_calc_distance_matrix(self):
        kernel_matrix = np.array(                                           #  The three H2O molecules differ only in the lenght of one H-O bond.
                [[1.0,                0.9983700867106244, 0.9961005991483849],
                 [0.9983700867106244, 1.0,                0.9994705736118292],
                 [0.9961005991483849, 0.9994705736118292, 1.0]])
        ref_distance_matrix = np.array(
                [[0.0,                  0.057094891003934174, 0.0883108243831424],
                 [0.057094891003934174, 0.0,                  0.03254001807531068 ],
                 [0.0883108243831424,   0.03254001807531068 , 0.0]])

        gap = mltools.gap.Gap()
        np.testing.assert_array_equal(gap.calc_distance_matrix(kernel_matrix), ref_distance_matrix)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)
