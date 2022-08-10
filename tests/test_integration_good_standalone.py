
#!/usr/bin/env python3

import unittest
from typing import List
import numpy as np
from test_integration_good_base import TestIntegrationGoodBase


class TestIntegrationGoodStandalone(TestIntegrationGoodBase, unittest.TestCase):

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_standalone()
        cls.path_to_executable = path_to_executable
    
    def run_with_data(self,
            goal: str, data: List[List[float]]) -> str:
        return super().run_with_data(self.path_to_executable, None, goal, data)


if __name__ == '__main__':
    print("Starting TestIntegrationGoodStandalone")
    unittest.main()
