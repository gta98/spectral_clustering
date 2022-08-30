
#!/usr/bin/env python3

import unittest
from typing import List
import numpy as np
from test_integration_good_base import TestIntegrationGoodBase, random_blob
import os
import spkmeansref
import pandas as pd


class TestIntegrationGoodPython(TestIntegrationGoodBase, unittest.TestCase):

    path_to_repo_folder: str = os.environ['HOME']+"/repos/softproj"
    path_to_writable_folder: str = os.environ['HOME']+"/repos/softproj"

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_python()
        cls.path_to_executable = path_to_executable
    
    def run_with_data(self,
            goal: str, data: List[List[float]],k=None) -> str:
        return super().run_with_data(self.path_to_executable, k if type(k) is int else int(np.random.randint(-3,5)), goal, data)
    
    def test_spk(self):
        #blob = random_blob('small')
        blob = [[y for y in x] for x in pd.read_csv(f'{self.path_to_repo_folder}/tests/resources/morefiles/spk_3.txt').to_numpy()]
        #result = spkmeansref.full_spk_1_to_5(blob,k=0)
        result = self.run_with_data('spk', blob, k=0)
        print("Result as follows:")
        result = [line.strip().split(",") for line in result.split("\n") if line]
        selected_indices = [int(x) for x in result[0]]
        result = [[float(x) for x in y] for y in result[1:]]
        k = len(result[0])
        print(f"Gonna call full_spk with blob of type {type(blob)} and k=={k}")
        T_ref = spkmeansref.full_spk_1_to_5(blob, k=0)
        k_ref = len(T_ref[0])
        import spkmeans
        result_ref = spkmeans.calc_kmeanspp(k_ref, T_ref)
        selected_indices_ref = result_ref[0]
        result_ref = result_ref[1:]
        self.assertEqual(k_ref, k)
        self.assert_mat_dist(result_ref, result)


if __name__ == '__main__':
    print("Starting TestIntegrationGoodPython")
    unittest.main()
