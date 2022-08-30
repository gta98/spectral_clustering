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

ROUNDING_DIGITS = 4
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(0)


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

    path_to_repo_folder: str = os.environ['HOME']+"/workspace/softproj"
    path_to_writable_folder: str = os.environ['HOME']+"/workspace/softproj"
    path_to_trash_folder: str = "/tmp"

    @classmethod
    def compile(cls):
        path_to_executable = cls.compile_c_python()
        cls.path_to_executable = path_to_executable

    def test_make_compatible_blob_symmetric(self):
        x = make_compatible_blob_symmetric(100)
        for i in range(100):
            for j in range(100):
                self.assertEqual(x[i][j], x[j][i])

    #@unittest.skip("This is no longer relevant")
    def test_numpy_to_numpy(self):
        # this tests: read_data, Mat_to_PyListListFloat, PyListListFloat_to_Mat, wrap__ndarray_to_list_of_lists
        n, d = 100, 10
        A = np.round(np.random.rand(n*d)*10,ROUNDING_DIGITS).reshape((n,d))
        A_list = [[float(y) for y in x] for x in A]
        save_path = f"{self.path_to_trash_folder}/tmp_mat.txt"
        with open(f'{save_path}.orig', 'w') as f:
            for line in A_list:
                f.write(','.join([str(x) for x in line]) + '\n')
        B_list = spkmeans_utils.numpy_to_numpy(A_list, save_path)
        B_list = [[round(y,ROUNDING_DIGITS) for y in x] for x in B_list]
        os.system(f"rm {save_path}*")
        self.assertEqual(A_list, B_list)
    
    def _compare_c_and_py(self,
            name: str,
            datapoints: List[List[float]],
            ptr_py: Callable, ptr_c: Callable,
            ptr_comparator: Callable):
        result_py = ptr_py(datapoints)
        result_c = ptr_c(datapoints)
        ptr_comparator(name, result_py, result_c)
    
    def _comparator_mat(self, name: str, result_py: List[List[float]], result_c: List[List[float]]):
        #dist = dist_between_centroid_lists(result_c, result_py)
        #relative_error = relative_error_centroids(result_py, result_c)
        #self.assertEqual(type(relative_error), np.float64, f"Failed test for {name}: could not determine relative error")
        #self.assertLess(relative_error, 1e-3, f"Failed test for {name}: relative error = {relative_error} is too high")
        self.assert_mat_dist(np.array(result_py),np.array(result_c), f"Failed test for {name}")

    def _comparator_jacobi(self,
            name: str,
            result_py: Tuple[List[float], List[List[float]]], result_c: Tuple[List[float], List[List[float]]]):
        vector_py, mat_py = np.around(result_py[0],5), np.around(result_py[1],5)
        vector_c, mat_c = np.around(result_c[0],5), np.around(result_c[1],5)
        self.assertTrue(np.allclose(vector_py, vector_c, rtol=1e-3), f"diff is {np.abs(vector_py-vector_c)}")
        self.assertTrue(np.allclose(mat_py, mat_c, rtol=1e-3), f"diff is {np.abs(mat_py-mat_c)}")

    def _comparator_calc_k(self, name: str, result_py: int, result_c: int):
        self.assertEqual(result_c, result_py, f"Failed test for {name}: Calculated k's are not equal - k(py)={result_py}, k(c)={result_c}")

    #@unittest.skip("----------------")
    def test_wam(self):
        blob = make_compatible_blob()
        self._compare_c_and_py('wam', blob, spkmeans_utils.full_wam, self.spkmeansmodule.full_wam, self._comparator_mat)
    
    #@unittest.skip("----------------")
    def test_ddg(self):
        self._compare_c_and_py('ddg', make_compatible_blob(), spkmeans_utils.full_ddg, self.spkmeansmodule.full_ddg, self._comparator_mat)
    
    #@unittest.skip("----------------")
    def test_lnorm(self):
        for i in range(5):
            print(f"Running lnorm #{i}")
            blob = make_compatible_blob()
            self._compare_c_and_py('lnorm', blob, spkmeans_utils.full_lnorm, self.spkmeansmodule.full_lnorm, self._comparator_mat)
            self.assert_mat_symmetric(spkmeans_utils.full_lnorm(blob))

    #@unittest.skip("----------------")
    def test_jacobi(self):
        for i in range(1):
            print(f"Running Jacobi #{i}")
            X=make_compatible_blob_symmetric(100)
            A_real, V_real = spkmeans_utils.full_jacobi(X)
            A_fake, V_fake = self.spkmeansmodule.full_jacobi(X)
            A_real, V_real = np.array(A_real), np.array(V_real)
            A_fake, V_fake = np.array(A_fake), np.array(V_fake)
            self.assertTrue(np.allclose(A_real, A_fake, rtol=1e-3))
            self.assertTrue(np.allclose(V_real, V_fake, rtol=1e-3))

    def test_jacobi_sorted(self):
        X = make_compatible_blob_symmetric(10)
        self._compare_c_and_py('jacobi_sorted', X,
            spkmeans_utils.full_jacobi_sorted, self.spkmeansmodule.full_jacobi_sorted, self._comparator_jacobi)
    
    #@unittest.skip("Skip for now")
    def test_full_spk_1_to_5(self):
        spkmeansmodule = self.spkmeansmodule
        X = [
            [0.010,0.010],
            [0.011,0.011],
            [0.012,0.012],
            [0.013,0.013],
            [0.210,0.210],
            [0.211,0.211],
            [0.212,0.212],
            [0.213,0.213],
            [0.810,0.810],
            [0.811,0.811],
            [0.812,0.812],
            [0.813,0.813],
        ]
        k=0
        L_norm = np.around(spkmeansmodule.full_lnorm(X),4).tolist()
        L_norm_ref = np.around(spkmeans_utils.full_lnorm(X),4).tolist()
        eigenvalues, eigenvectors = spkmeansmodule.full_jacobi_sorted(L_norm)
        eigenvalues_ref, eigenvectors_ref = spkmeans_utils.full_jacobi_sorted(L_norm_ref)
        print(np.around(eigenvectors_ref,4)-np.around(eigenvectors,4))
        if k==0:
            k = spkmeansmodule.full_calc_k(eigenvalues)
            k_ref = spkmeans_utils.calc_k(eigenvalues)
        U = [x[:k] for x in eigenvectors]
        U_ref = [x[:k] for x in eigenvectors_ref]
        T = spkmeansmodule.normalize_matrix_by_rows(U)
        T = np.array(spkmeansmodule.full_spk_1_to_5(X,0))
        T_ref = np.array(spkmeans_utils.normalize_matrix_by_rows(U_ref))
        diff_T = np.sum(np.abs(T-T_ref))
        print(f"results diff {diff_T}")
        sys.path.insert(0, self.path_to_repo_folder)
        import spkmeans
        results=spkmeans.calc_kmeanspp(k,T)
        print("bye T")
    
    def test_PTAP(self):
        n = 20
        A = make_compatible_blob(n,n)
        P = make_compatible_blob(n,n)
        PTAP = np.array(P).transpose()@np.array(A)@np.array(P)
        #C_PTAP = np.array(self.spkmeansmodule.test_PTAP(A,P))
        #relative_error = relative_error_matrices(PTAP, C_PTAP)
        #print(f"Relative error is {relative_error}")
        #print(PTAP)
        #self.assertLess(relative_error, 1e-4)
    
    def test_calc_k(self):
        def test_calc_k_length_n(n:int):
            eigenvalues = [float(x) for x in np.random.rand(n)]
            eigenvalues_asc = sorted(eigenvalues)
            eigenvalues_des = eigenvalues_asc[::-1]
            k_c = self.spkmeansmodule.full_calc_k(eigenvalues_des)
            k_py = spkmeans_utils.calc_k(np.array(eigenvalues_des))
            self.assertEqual(k_c, k_py, f"k value calculated by C and Python not equal - k(c)=={k_c}, k(py)=={k_py}, n is {n}")
        test_calc_k_length_n(2)
        test_calc_k_length_n(3)
        test_calc_k_length_n(4)
        test_calc_k_length_n(998)
        test_calc_k_length_n(999)
        test_calc_k_length_n(1000)

    def test_normalize_rows(self):
        n,k = 1000,100
        U = make_compatible_blob(n,k,offset=+0.1)
        T_py = spkmeans_utils.normalize_matrix_by_rows(U)
        T_c = self.spkmeansmodule.normalize_matrix_by_rows(U)
        relative_error = relative_error_centroids(T_py, T_c)
        self.assertLess(relative_error, 1e-4)

    def test_mat_cellwise_add(self):
        A = make_compatible_blob(14,91,offset=+1.0)
        B = make_compatible_blob(14,91,offset=+1.0)
        C = spkmeans_utils.convert__np_matrix__to__list_of_lists(spkmeans_utils.convert__list_of_lists__to__np_matrix(A)  \
                                                                 +                                                        \
                                                                 spkmeans_utils.convert__list_of_lists__to__np_matrix(B))
        C_module = self.spkmeansmodule.mat_cellwise_add(A, B)
        relative_error = relative_error_centroids(C, C_module)
        self.assertLess(relative_error, 1e-3)

    def test_mat_cellwise_mul(self):
        A = make_compatible_blob(14,91,offset=+1.0)
        B = make_compatible_blob(14,91,offset=+1.0)
        C = spkmeans_utils.convert__np_matrix__to__list_of_lists(spkmeans_utils.convert__list_of_lists__to__np_matrix(A)  \
                                                                 *                                                        \
                                                                 spkmeans_utils.convert__list_of_lists__to__np_matrix(B))
        C_module = self.spkmeansmodule.mat_cellwise_mul(A, B)
        relative_error = relative_error_centroids(C, C_module)
        self.assertLess(relative_error, 1e-3)

    def test_matmul(self):
        A = make_compatible_blob(14,44,offset=+1.0)
        B = make_compatible_blob(44,32,offset=+1.0)
        C = spkmeans_utils.convert__np_matrix__to__list_of_lists(spkmeans_utils.convert__list_of_lists__to__np_matrix(A)  \
                                                                 @                                                        \
                                                                 spkmeans_utils.convert__list_of_lists__to__np_matrix(B))
        C_module = self.spkmeansmodule.matmul(A, B)
        relative_error = relative_error_centroids(C, C_module)
        self.assertLess(relative_error, 1e-3)
    
    def test_mat_swap_rows(self):
        A = make_compatible_blob(14,44,offset=+1.0)
        indices = np.arange(44)
        np.random.shuffle(indices)
        indices = [int(x) for x in indices]
        B = spkmeans_utils.reorder_mat_cols_by_indices(A, indices)
        #print("Swapped rows on:")
        #print(B)
        B_module = self.spkmeansmodule.reorder_mat_cols_by_indices(A, indices)
        relative_error = relative_error_centroids(B, B_module)
        self.assertLess(relative_error, 1e-5)
    
    #@unittest.skip("LALALALALLA")
    def test_sort_cols_by_vector_desc(self):
        A = make_compatible_blob(44,44,offset=+1.0)
        indices = np.arange(44)
        np.random.shuffle(indices)
        indices = [int(x) for x in indices]
        B = spkmeans_utils.sort_cols_by_vector_desc(A, indices)[0]
        B_module = self.spkmeansmodule.sort_cols_by_vector_desc(A, indices)
        relative_error = relative_error_centroids(B, B_module)
        self.assertLess(relative_error, 1e-5)


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
    #print(relerr_arr)
    return np.mean(relerr_arr)

if __name__ == '__main__':
    print("Starting tests")
    unittest.main()
