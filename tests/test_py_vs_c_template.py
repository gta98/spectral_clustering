from tkinter import ROUND
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
from test_integration_base import TestIntegrationBase
import re
import os

from test_py_vs_c import make_compatible_blob_symmetric

ROUNDING_DIGITS = 3


class TestAgainstData(TestIntegrationBase, unittest.TestCase):

    path_to_repo_folder: str = os.environ['HOME']+"/workspace/softproj"
    path_to_writable_folder: str = "/tmp"

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_python()
        cls.path_to_executable = path_to_executable

    X = [
            [-5.056, 11.011],
            [-6.409, -7.962],
            [5.694, 9.606],
            [6.606, 9.396],
            [-6.772, -5.727],
            [-4.498, 8.399],
            [-4.985, 9.076],
            [4.424, 8.819],
            [-7.595, -7.211],
            [-4.198, 8.371]
    ]

    W = [
            [0.000,0.000,0.004,0.003,0.000,0.263,0.380,0.008,0.000,0.250],
            [0.000,0.000,0.000,0.000,0.322,0.000,0.000,0.000,0.495,0.000],
            [0.004,0.000,0.000,0.626,0.000,0.006,0.005,0.474,0.000,0.007],
            [0.003,0.000,0.626,0.000,0.000,0.004,0.003,0.323,0.000,0.004],
            [0.000,0.322,0.000,0.000,0.000,0.001,0.001,0.000,0.428,0.001],
            [0.263,0.000,0.006,0.004,0.001,0.000,0.659,0.011,0.000,0.860],
            [0.380,0.000,0.005,0.003,0.001,0.659,0.000,0.009,0.000,0.590],
            [0.008,0.000,0.474,0.323,0.000,0.011,0.009,0.000,0.000,0.013],
            [0.000,0.495,0.000,0.000,0.428,0.000,0.000,0.000,0.000,0.000],
            [0.250,0.000,0.007,0.004,0.001,0.860,0.590,0.013,0.000,0.000]
    ]

    D_inv_sqrt = [
            [1.050,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
            [0.000,1.105,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
            [0.000,0.000,0.944,0.000,0.000,0.000,0.000,0.000,0.000,0.000],
            [0.000,0.000,0.000,1.018,0.000,0.000,0.000,0.000,0.000,0.000],
            [0.000,0.000,0.000,0.000,1.152,0.000,0.000,0.000,0.000,0.000],
            [0.000,0.000,0.000,0.000,0.000,0.744,0.000,0.000,0.000,0.000],
            [0.000,0.000,0.000,0.000,0.000,0.000,0.779,0.000,0.000,0.000],
            [0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.092,0.000,0.000],
            [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.040,0.000],
            [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.761]
    ]

    eigenvalues = [1.121,1.593,1.627,0.039,1.405,1.490,1.376,1.347,0.002,0.000]

    eigenvectors = [
            [0.825,0.000,0.001,-0.220,-0.000,0.043,-0.410,0.016,-0.145,0.281],
            [0.000,0.544,0.000,0.008,-0.613,-0.000,0.000,-0.000,0.507,0.267],
            [-0.001,-0.000,0.767,0.510,0.000,0.000,-0.006,-0.158,-0.172,0.312],
            [-0.001,0.000,-0.587,0.476,0.000,-0.002,-0.028,-0.564,-0.160,0.289],
            [-0.000,0.301,0.000,0.007,0.780,0.000,-0.000,0.000,0.485,0.256],
            [-0.365,-0.000,0.000,-0.314,-0.000,0.741,-0.145,0.000,-0.204,0.396],
            [0.174,-0.000,-0.000,-0.301,0.000,-0.136,0.823,-0.045,-0.195,0.378],
            [0.000,-0.000,-0.260,0.426,-0.000,0.003,0.041,0.809,-0.149,0.270],
            [0.000,-0.783,-0.000,0.008,-0.126,0.000,0.000,-0.000,0.539,0.283],
            [-0.394,-0.000,0.001,-0.306,-0.000,-0.656,-0.361,0.017,-0.199,0.387]
    ]

    # data end

    D = np.diag(np.square(1/np.diag(np.array(D_inv_sqrt))))

    def test_wam(self):
        W = self.spkmeansmodule.full_wam(self.X)
        self.assert_mat_dist(self.W, W, f"wam")
        #relative_error = relative_error_centroids(self.W, W)
        #self.assertLess(relative_error, 1e-3, f"test_wam: Relative error is too high - {relative_error}")

    #@unittest.skip('blah')
    def test_ddg(self):
        D = self.spkmeansmodule.full_ddg(self.X)
        self.assert_mat_dist(self.D, D, f"ddg")
        #relative_error = relative_error_centroids(self.D, D)
        #self.assertLess(relative_error, 1e-3, f"test_ddg: Relative error is too high - {relative_error}")

    #@unittest.skip("Very low error for eigenvalues, HIGH ERROR (0.1, 10%) for eigenVECTORS")
    def test_jacobi_template(self):
        D_inv_sqrt = np.array(self.D_inv_sqrt)
        W = np.array(self.W)
        n = W.shape[0]
        assert(n==W.shape[1]==D_inv_sqrt.shape[0]==D_inv_sqrt.shape[1])
        lnorm = self.spkmeansmodule.test_calc_L_norm(self.W, self.D_inv_sqrt)
        lnorm_ref = np.around(np.eye(n) - (D_inv_sqrt@W@D_inv_sqrt), ROUNDING_DIGITS)
        eigenvalues, eigenvectors = self.spkmeansmodule.full_jacobi(lnorm)
        eigenvalues, eigenvectors = np.around(eigenvalues,ROUNDING_DIGITS), np.around(eigenvectors,ROUNDING_DIGITS)
        eigenvalues_ref, eigenvectors_ref = np.linalg.eig(lnorm_ref)
        eigenvalues_ref, eigenvectors_ref = np.around(eigenvalues_ref,ROUNDING_DIGITS), np.around(eigenvectors_ref,ROUNDING_DIGITS)
        eigenvalues = [round(x,ROUNDING_DIGITS) for x in eigenvalues]
        eigenvectors, eigenvalues = spkmeans_utils.sort_cols_by_vector_desc(eigenvectors, eigenvalues)
        self.eigenvectors, self.eigenvalues = spkmeans_utils.sort_cols_by_vector_desc(self.eigenvectors, self.eigenvalues)
        eigenvectors_ref, eigenvalues_ref = spkmeans_utils.sort_cols_by_vector_desc(eigenvectors_ref, eigenvalues_ref)
        eigenvectors = np.array(eigenvectors)
        self.eigenvectors = np.array(self.eigenvectors)
        #print(lnorm@eigenvectors[:,0] - eigenvalues[0]*eigenvectors[:,0])
        relative_error_eigenvalues = relative_error_vectors(self.eigenvalues, eigenvalues)
        relative_error_eigenvectors = relative_error_colwise_mean(self.eigenvectors, eigenvectors)
        self.assertLess(relative_error_eigenvalues, 1e-3, f"test_jacobi_template: Eigenvalues relative error is too high - \nGot: {eigenvalues}\nReal: {self.eigenvalues}")
        self.assertLess(relative_error_eigenvectors, 1e-3, f"test_jacobi_template: Eigenvectors relative error is too high - {eigenvectors}")
    
    def test_jacobi_on_known(self):
        mat = make_compatible_blob_symmetric(3)
        eigenvalues_ref, eigenvectors_ref = np.linalg.eig(np.array(mat))
        print(eigenvectors_ref)
        print(eigenvalues_ref)
        eigenvalues, eigenvectors = spkmeans_utils.full_jacobi(mat)
        eigenvalues, eigenvectors = np.array(eigenvalues), np.array(eigenvectors)
        print("=======")
        print(np.array(eigenvectors))
        print(np.array(eigenvalues))
        eigenvectors =eigenvectors.transpose()
        print(mat@eigenvectors[:,0]-eigenvalues[0]*eigenvectors[:,0])


def relative_error_vectors(vector_real: List[float], vector_calc: List[float]) -> float:
    vector_real, vector_calc = np.array(vector_real), np.array(vector_calc)
    vector_real, vector_calc = np.round(vector_real,ROUNDING_DIGITS), np.round(vector_calc,ROUNDING_DIGITS)
    dist = np.sqrt(np.sum(np.square(vector_real-vector_calc)))
    relative_error = dist / np.sqrt(np.sum(np.square(vector_real)))
    return relative_error

def relative_error_colwise_mean(eigenvectors_real: List[List[float]], eigenvectors_calc: List[List[float]]) -> float:
    # each column is an eigenvector
    eigenvectors_real, eigenvectors_calc = np.array(eigenvectors_real), np.array(eigenvectors_calc)
    eigenvectors_real, eigenvectors_calc = np.round(eigenvectors_real,ROUNDING_DIGITS), np.round(eigenvectors_calc,ROUNDING_DIGITS)
    dist_weight = np.sqrt(np.sum(np.square(eigenvectors_real-eigenvectors_calc), axis=0)) # each column is an eigenvector
    real_weight = np.sqrt(np.sum(np.square(eigenvectors_real), axis=0))
    relative_error_vector = dist_weight/real_weight # entry i == relative error of vector i
    relative_error = np.mean(relative_error_vector)
    return relative_error


if __name__ == '__main__':
    print("Starting tests")
    unittest.main()


