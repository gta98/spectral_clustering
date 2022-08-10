import unittest
from abc import abstractmethod, abstractclassmethod
from typing import List, Callable, Tuple, Union, Optional, Any
NoneType = type(None)
from math import inf
import time
import subprocess
import os
import numpy as np
from test_integration_base import TestIntegrationBase, random_blob, random_blob_symmetric, str_to_mat
import utils as spkmeansmodule_ref
import random
import inspect


class TestIntegrationGoodBase(TestIntegrationBase):
    """
    Testing class for good cases - valid inputs, in standalone C
    """

    path_to_repo_folder: str = "/home/fakename/repos/softproj"
    path_to_writable_folder: str = "/tmp"

    def assert_mat_dist(self, real: np.ndarray, calc: np.ndarray):
        real, calc = np.round(np.array(real),4), np.round(np.array(calc),4)
        dist = min(
            np.mean(np.abs(real-calc)) / np.mean(np.abs(real)),
            np.mean(np.square(real-calc))
        )
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        caller = calframe[1][3]
        self.assertLess(dist, 1e-5, f"workdir is {self.path_to_workdir}, caller is {caller}\nREAL:\n{real}\n\nFAKE:\n{calc}")

    def test_wam(self):
        blob = random_blob()
        result = str_to_mat(self.run_with_data('wam', blob))
        result_ref = spkmeansmodule_ref.full_wam(blob)
        self.assert_mat_dist(result_ref, result)
    
    def test_ddg(self):
        blob = random_blob()
        result = str_to_mat(self.run_with_data('ddg', blob))
        result_ref = spkmeansmodule_ref.full_ddg(blob)
        self.assert_mat_dist(result_ref, result)

    def test_lnorm(self):
        blob = random_blob()
        result = str_to_mat(self.run_with_data('lnorm', blob))
        result_ref = spkmeansmodule_ref.full_lnorm(blob)
        self.assert_mat_dist(result_ref, result)
    
    def test_jacobi(self):
        blob = random_blob_symmetric('small')
        result = str_to_mat(self.run_with_data('jacobi', blob))
        eigenvalues, eigenvectors = result[0], result[1:]
        eigenvalues_ref, eigenvectors_ref = spkmeansmodule_ref.full_jacobi(blob)
        self.assert_mat_dist(eigenvalues_ref, eigenvalues)
        self.assert_mat_dist(eigenvectors_ref, eigenvectors)