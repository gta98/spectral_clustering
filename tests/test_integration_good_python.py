
#!/usr/bin/env python3

import unittest
from typing import List
import numpy as np
from test_integration_good_base import TestIntegrationGoodBase, random_blob
import os
import spkmeansref


class TestIntegrationGoodPython(TestIntegrationGoodBase, unittest.TestCase):

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_python()
        cls.path_to_executable = path_to_executable
        path_to_executable_parent = '/'.join(path_to_executable.split("/")[:-1])
        print(f"exec parent: {path_to_executable_parent}")
        import sys
        sys.path.insert(0, path_to_executable_parent)
        os.system(f"cp {path_to_executable_parent}/*.so .")
    
    def run_with_data(self,
            goal: str, data: List[List[float]],k=None) -> str:
        return super().run_with_data(self.path_to_executable, k if type(k) is int else int(np.random.randint(-3,5)), goal, data)
    
    def test_spk(self):
        blob = random_blob('small')
        result = self.run_with_data('spk', blob, k=0)
        print("Result as follows:")
        result = [line.strip().split(",") for line in result.split("\n") if line]
        selected_indices = result[0]
        result = result[1:]
        k = len(result[0])
        print(f"Gonna call full_spk with blob of type {type(blob)} and k=={k}")
        result_ref = spkmeansref.full_spk(blob, k=0)
        k_ref = len(result_ref[0])
        self.assertEqual(k_ref, k)
        self.assert_mat_dist(result_ref, result)


if __name__ == '__main__':
    print("Starting TestIntegrationGoodPython")
    unittest.main()
