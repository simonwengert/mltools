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
        set_ids = ['train', 'validate', 'holdout']
        gap = mltools.gap.Gap()
        for set_id in set_ids:
            gap._check_set_id(set_id)

    def test__check_set_id_fail(self):
        set_id = 'none'
        gap = mltools.gap.Gap()
        with self.assertRaises(ValueError):
            gap._check_set_id(set_id)

    def test_read_atoms_pass(self):
        set_ids = ['train', 'validate', 'holdout']
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
        set_ids = ['train', 'validate', 'holdout']
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')
        atoms_ref = ase.io.read(p_xyz_file)
        gap = mltools.gap.Gap()

        for set_id in set_ids:
            gap.read_atoms(p_xyz_file, set_id)
            gap.write_atoms(set_id, set_id)
            atoms_should = ase.io.read(set_id+'.xyz')

            np.testing.assert_array_equal(atoms_should.get_positions(), atoms_ref.get_positions())

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)
