# #!/usr/bin/env python
import unittest
import os
import tempfile
import shutil
import filecmp
import numpy as np

import ase.io

import mltools.gap


class TestParser(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)

    def test__build_assign_str(self):
        items = {'key_0' : 'val_0',
                 'key_1' : 1,
                 'key_2' : 1.1,
                 'key_3' : False}
        ref_str = 'key_1=1 key_0=val_0 key_3=False key_2=1.1'

        gap = mltools.gap.Gap()
        self.assertEqual(gap._build_assign_str(items), ref_str)

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
        gap = mltools.gap.Gap()

        for set_id in set_ids:
            gap.read_atoms(p_xyz_file, set_id)

    def test_read_atoms_fail(self):
        set_id = 'none'
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')
        gap = mltools.gap.Gap()
        with self.assertRaises(ValueError):
            gap.read_atoms(p_xyz_file, set_id)

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

    def test_write_teach_sparse_parameters(self):
        ref_file = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'teach.params')
        gap = mltools.gap.Gap()
        gap.params_teach_sparse = {'key_0' : 'val_0', 'key_1' : 'val_1'}
        gap.gaps = {'name' : 'val_2', 'key_3' : 'val_3'}
        gap.write_teach_sparse_parameters()
        self.assertTrue(filecmp.cmp('teach.params', ref_file))

    def test_write_quip_parameters(self):
        ref_file = os.path.join(self.cwd, 'tests', 'data', 'cmp_files', 'quip.params')
        gap = mltools.gap.Gap()
        gap.params_quip = {'key_0' : 'val_0', 'key_1' : 'val_1'}
        gap.write_quip_parameters()
        self.assertTrue(filecmp.cmp('quip.params', ref_file))

    def test__build_cmd_teach(self):
        ref_cmd_teach = '! teach_sparse default_sigma={0 0.5 1 2} key_0=val_0 gap={gap_0 key_1=val_1}'
        gap = mltools.gap.Gap()
        gap.params_teach_sparse = {'default_sigma' : [0, 0.5, 1, 2],
                                   'key_0' : 'val_0'}
        gap.gaps = {'name' : 'gap_0',
                    'key_1' : 'val_1'}
        self.assertEqual(gap.cmd_teach, ref_cmd_teach)

    def test__build_cmd_quip(self):
        ref_cmd_quip = '! quip key_1=val_1 key_0=val_0 | grep AT | sed \'s/AT//\''
        gap = mltools.gap.Gap()
        gap.params_quip = {'key_0' : 'val_0',
                           'key_1' : 'val_1'}
        self.assertEqual(gap.cmd_quip, ref_cmd_quip)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)
