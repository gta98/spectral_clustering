
#!/usr/bin/env python3

import unittest
from typing import List
import numpy as np
from test_integration_good_base import TestIntegrationGoodBase


class TestIntegrationGoodPython(TestIntegrationGoodBase, unittest.TestCase):

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_python()
        cls.path_to_executable = path_to_executable
    
    def run_with_data(self,
            goal: str, data: List[List[float]]) -> str:
        return super().run_with_data(self.path_to_executable, int(np.random.randint(-3,5)), goal, data)


if __name__ == '__main__':
    print("Starting TestIntegrationGoodPython")
    unittest.main()
