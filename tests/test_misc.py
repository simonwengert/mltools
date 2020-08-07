# #!/usr/bin/env python
import unittest
import os

import mltools.misc


class TestParser(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def test_get_info(self):
        p_xyz_file = os.path.join(self.cwd, 'tests', 'data', 'xyz', '0_test.xyz')

        keys = ['float_0', 'float_1']

        dict_ref = {keys[0] : [1.0, 2.0, 3.0],
                    keys[1] : [0.10, 0.20, 0.30]}
        dict_is = mltools.misc.get_info(p_xyz_file, keys)

        for key in keys:
            self.assertListEqual(dict_is[key], dict_ref[key])

    # def tearDown(self):
    #     os.chdir(self.cwd)
    #     shutil.rmtree(self.tmpdir)
