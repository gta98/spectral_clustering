#!/usr/bin/env python3

import unittest
from abc import abstractmethod
from typing import List, Callable, Tuple, Union, Optional, Any
NoneType = type(None)
from math import inf
import time
import subprocess
import os
import numpy as np
from test_integration_base import TestIntegrationBase, random_blob, random_blob_symmetric
import utils as spkmeansmodule_ref

class TestIntegrationStandaloneGood(TestIntegrationBase):
    """
    Testing class for good cases - valid inputs, in standalone C
    """

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_standalone()
        cls.path_to_executable = path_to_executable
    
    @classmethod
    def run_with_data(cls,
            k: Optional[int], goal: str, data: List[List[float]]) -> str:
        return super().run_with_data(cls.path_to_executable, k, goal, data)

    def test_wam(self):
        blob = random_blob('big')
        result = self.run_with_data(None, 'wam', blob)
        result_ref = spkmeansmodule_ref.calc_wam(blob)


if __name__ == '__main__':
    print("Starting TestIntegrationStandaloneGood")
    unittest.main()
