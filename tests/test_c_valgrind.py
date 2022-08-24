#!/usr/bin/env python3

import unittest
from unittest.mock import patch
import pickle
import logging
import sys
import numpy as np
import sklearn.cluster
from typing import List, Callable, Tuple
import math
import time
from sklearn.datasets import make_blobs
import spkmeansref as spkmeans_utils
from test_integration_base import TestIntegrationBase, make_compatible_blob, make_compatible_blob_symmetric
import re
import os
import pytest
import time

ROUNDING_DIGITS = 4

class TestFit(TestIntegrationBase, unittest.TestCase):

    path_to_repo_folder: str = os.environ['HOME']+"/workspace/softproj"
    path_to_writable_folder: str = os.environ['HOME']+"/workspace/softproj"
    path_to_trash_folder: str = "/tmp"

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_standalone()
        cls.path_to_executable = \
            f"valgrind --show-leak-kinds=all --leak-check=yes {path_to_executable}" \
            .split(" ")
        os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = "20"

    def run_with_data(self,
            goal: str, data: List[List[float]]) -> str:
        return super().run_with_data(self.path_to_executable, None, goal, data)

    def test_wam(self):
        goal = "wam"
        blob = make_compatible_blob(1000,10)
        result = self.run_with_data(goal, blob)
        print("done")
        #print(result)


if __name__ == '__main__':
    print("Starting tests")
    unittest.main()
