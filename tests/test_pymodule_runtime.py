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
from test_integration_base import TestIntegrationBase
import re
import os
import pytest
import time

ROUNDING_DIGITS = 4


def make_compatible_blob(n=100,d=10, offset=0) -> List[List[float]]:
    x, _ = make_blobs(n_samples=n, n_features=d)
    x: np.ndarray
    return [[float(z+offset) for z in y] for y in list(x)]

def make_compatible_blob_symmetric(n=100) -> List[List[float]]:
    a = np.random.rand(n,n)
    bottom_left = np.tril(a)
    top_right = np.tril(a,-1).T
    m = bottom_left + top_right
    return [list(y) for y in list(m)]

class TestFit(TestIntegrationBase, unittest.TestCase):

    path_to_repo_folder: str = os.environ['HOME']+"/repos/softproj"
    path_to_writable_folder: str = os.environ['HOME']+"/repos/softproj"
    path_to_trash_folder: str = "/tmp"

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_python()
        cls.path_to_executable = path_to_executable

    #@unittest.skip("----------------")
    def test_wam(self):
        start = time.time()
        self.spkmeansmodule.full_wam(make_compatible_blob(1000,10))
        diff = int(time.time() - start)
        print(f"Time elapsed: {diff} seconds (WAM)")

    #@unittest.skip("----------------")
    def test_ddg(self):
        start = time.time()
        self.spkmeansmodule.full_ddg(make_compatible_blob(1000,10))
        diff = int(time.time() - start)
        print(f"Time elapsed: {diff} seconds (DDG)")
    
    #@unittest.skip("----------------")
    def test_lnorm(self):
        start = time.time()
        self.spkmeansmodule.full_lnorm(make_compatible_blob(1000,10))
        diff = int(time.time() - start)
        print(f"Time elapsed: {diff} seconds (LNORM)")

    #@unittest.skip("Does not work")
    def test_jacobi(self):
        start = time.time()
        self.spkmeansmodule.full_jacobi(make_compatible_blob_symmetric(10000))
        diff = int(time.time() - start)
        print(f"Time elapsed: {diff} seconds (JACOBI)")
    
    def test_jacobi_sorted(self):
        start = time.time()
        self.spkmeansmodule.full_jacobi_sorted(make_compatible_blob_symmetric(10000))
        diff = int(time.time() - start)
        print(f"Time elapsed: {diff} seconds (JACOBI SORTED)")


def randomize_fit_params(k=None, max_iter=None, eps=None, point_count=None, dims_count=None):
    #np.random.seed(4)
    k = k or np.random.randint(2, 5)
    max_iter = max_iter or np.random.randint(100, 300)
    eps = eps or float(np.random.rand(1,1))/100
    point_count = point_count or np.random.randint(50, 200)
    dims_count = dims_count or np.random.randint(3,5)
    datapoints_list = np.random.rand(point_count, dims_count)
    indices_column = np.array(range(datapoints_list.shape[0])).reshape(datapoints_list.shape[0],1)
    datapoints_list = np.hstack((indices_column, datapoints_list))
    initial_centroids_list = self.kmeans_pp.KMeansPlusPlus(k, datapoints_list)
    initial_centroids_indices_as_written = [int(initial_centroids_list[i][0]) for i in range(len(initial_centroids_list))]
    #print(','.join([str(x) for x in initial_centroids_indices_as_written]))
    initial_centroids_indices_actual = self.kmeans_pp.select_actual_centroids(datapoints_list, initial_centroids_list)
    datapoints_list = datapoints_list[:,1:]
    return (
        initial_centroids_indices_actual,
        datapoints_list,
        dims_count,
        k,
        point_count,
        max_iter,
        eps
    )


def dist_between_centroid_lists(list_1: np.ndarray, list_2: np.ndarray) -> float:
    if type(list_1) == list and type(list_2) == list:
        list_1 = np.array(list_1)
        list_2 = np.array(list_2)
    list_1 = np.around(list_1, ROUNDING_DIGITS)
    list_2 = np.around(list_2, ROUNDING_DIGITS)
    return np.linalg.norm((list_1)-(list_2))

def relative_error_centroids(centroids_real: List[List[float]], centroids_calc: List[List[float]]) -> float:
    centroids_real_round = [[round(y,ROUNDING_DIGITS) for y in x] for x in centroids_real]
    centroids_calc_round = [[round(y,ROUNDING_DIGITS) for y in x] for x in centroids_calc]
    return np.linalg.norm(np.sort(centroids_real_round)-np.sort(centroids_calc_round))/np.linalg.norm(centroids_real_round)

def relative_error_matrices(centroids_real: List[List[float]], centroids_calc: List[List[float]]) -> float:
    real = np.array([[round(y,ROUNDING_DIGITS) for y in x] for x in centroids_real])
    calc = np.array([[round(y,ROUNDING_DIGITS) for y in x] for x in centroids_calc])
    relerr_mat = np.abs(calc-real)/np.abs(real)
    relerr_arr = []
    for i in range(len(centroids_real)):
        for j in range(len(centroids_real[i])):
            if relerr_mat[i,j] != np.nan:
                relerr_arr.append(relerr_mat[i,j])
    relerr_arr = np.array(relerr_arr)
    print(relerr_arr)
    return np.mean(relerr_arr)

if __name__ == '__main__':
    print("Starting tests")
    unittest.main()
