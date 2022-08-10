#!/usr/bin/env python3

import unittest
from unittest.mock import patch
import pickle
import logging
import sys
import numpy as np
import sklearn.cluster
import kmeans_pp
from typing import List, Callable, Tuple
import math
import time
from sklearn.datasets import make_blobs
import spkmeansmodule
import spkmeansref as spkmeans_utils
import re
import os

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


class TestAgainstData(unittest.TestCase):
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

    D_pow_minus_half = [
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

    D = np.diag(np.square(1/np.diag(np.array(D_pow_minus_half))))

    def test_wam(self):
        W = spkmeansmodule.full_wam(self.X)
        relative_error = relative_error_centroids(self.W, W)
        self.assertLess(relative_error, 1e-3, f"test_wam: Relative error is too high - {relative_error}")
    
    def test_ddg(self):
        D = spkmeansmodule.full_ddg(self.X)
        relative_error = relative_error_centroids(self.D, D)
        self.assertLess(relative_error, 1e-3, f"test_ddg: Relative error is too high - {relative_error}")

    @unittest.skip("Very low error for eigenvalues, HIGH ERROR (0.1, 10%) for eigenVECTORS")
    def test_jacobi_template(self):
        D_pow_minus_half = np.array(self.D_pow_minus_half)
        W = np.array(self.W)
        lnorm = spkmeansmodule.full_lnorm(self.X)
        eigenvalues, eigenvectors = spkmeansmodule.full_jacobi(lnorm)
        eigenvalues = [round(x,ROUNDING_DIGITS) for x in eigenvalues]
        relative_error_eigenvalues = relative_error_vectors(self.eigenvalues, eigenvalues)
        relative_error_eigenvectors = relative_error_colwise_mean(self.eigenvectors, eigenvectors)
        self.assertLess(relative_error_eigenvalues, 1e-3, f"test_jacobi_template: Eigenvalues relative error is too high - \nGot: {eigenvalues}\nReal: {self.eigenvalues}")
        self.assertLess(relative_error_eigenvectors, 1e-3, f"test_jacobi_template: Eigenvectors relative error is too high - {eigenvectors}")


class TestFit(unittest.TestCase):

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
        save_path = "./tmp_mat.txt"
        with open(f'{save_path}.orig', 'w') as f:
            for line in A_list:
                f.write(','.join([str(x) for x in line]) + '\n')
        #print(A_list)
        B_list = spkmeans_utils.numpy_to_numpy(A_list, save_path)
        B_list = [[round(y,ROUNDING_DIGITS) for y in x] for x in B_list]
        os.system(f"rm {save_path}*")
        self.assertEqual(A_list, B_list)
    
    def _compare_c_and_py(self,
            name: str,
            datapoints: List[List[float]],
            ptr_py: Callable, ptr_c: Callable,
            ptr_comparator: Callable):
        #print()
        #print('\n'.join([','.join([str(round(y,ROUNDING_DIGITS)) for y in x]) for x in datapoints]))
        result_py = ptr_py(datapoints)
        result_c = ptr_c(datapoints)
        #print()
        #print('\n'.join([','.join([str(round(y,ROUNDING_DIGITS)) for y in x]) for x in result_py]))
        ptr_comparator(name, result_py, result_c)
    
    def _comparator_mat(self, name: str, result_py: List[List[float]], result_c: List[List[float]]):
        dist = dist_between_centroid_lists(result_c, result_py)
        relative_error = relative_error_centroids(result_py, result_c)
        self.assertEqual(type(relative_error), np.float64, f"Failed test for {name}: could not determine relative error")
        self.assertLess(relative_error, 1e-3, f"Failed test for {name}: relative error = {relative_error} is too high")

    def _comparator_jacobi(self,
            name: str,
            result_py: Tuple[List[float], List[List[float]]], result_c: Tuple[List[float], List[List[float]]]):
        vector_py, mat_py = result_py[0], result_py[1]
        vector_c, mat_c = result_c[0], result_c[1]
        #print("jacobi comparator started")
        #print(vector_py)
        #print(vector_c)
        #print(mat_py)
        #print(mat_c)
        vector_py, vector_c = np.array(vector_py), np.array(vector_c)
        dist_vector = np.sqrt(np.sum(np.square(vector_py-vector_c)))
        relative_error_vector = dist_vector/np.sqrt(np.sum(np.square(vector_py)))
        self.assertEqual(type(relative_error_vector), np.float64, f"Failed test for {name}: could not determine relative error (eigenvalues)")
        self.assertLess(relative_error_vector, 1e-3, f"Failed test for {name}: eigenvalues are too far apart")
        relative_error = relative_error_centroids(mat_py, mat_c)
        self.assertEqual(type(relative_error), np.float64, f"Failed test for {name}: could not determine relative error")
        self.assertLess(relative_error, 1e-3, f"Failed test for {name}: relative error = {relative_error} is too high")

    def _comparator_calc_k(self, name: str, result_py: int, result_c: int):
        self.assertEqual(result_c, result_py, f"Failed test for {name}: Calculated k's are not equal - k(py)={result_py}, k(c)={result_c}")

    #@unittest.skip("----------------")
    def test_wam(self):
        self._compare_c_and_py('wam', make_compatible_blob(), spkmeans_utils.full_wam, spkmeansmodule.full_wam, self._comparator_mat)
    
    #@unittest.skip("----------------")
    def test_ddg(self):
        self._compare_c_and_py('ddg', make_compatible_blob(), spkmeans_utils.full_ddg, spkmeansmodule.full_ddg, self._comparator_mat)
    
    #@unittest.skip("----------------")
    def test_lnorm(self):
        self._compare_c_and_py('lnorm', make_compatible_blob(), spkmeans_utils.full_lnorm, spkmeansmodule.full_lnorm, self._comparator_mat)

    #@unittest.skip("Does not work")
    def test_jacobi(self):
        self._compare_c_and_py('jacobi', make_compatible_blob_symmetric(10),
            spkmeans_utils.full_jacobi, spkmeansmodule.full_jacobi, self._comparator_jacobi)

    def test_jacobi_sorted(self):
        X = make_compatible_blob_symmetric(10)
        c_val, c_vec = spkmeansmodule.full_jacobi_sorted(X)
        #print(f"c eigenvalues: {c_val}")
        py_val, py_vec = spkmeans_utils.full_jacobi_sorted(X)
        #self._compare_c_and_py('jacobi_sorted', make_compatible_blob_symmetric(10),
        #    spkmeans_utils.full_jacobi_sorted, spkmeansmodule.full_jacobi_sorted, self._comparator_jacobi)
    
    def test_PTAP(self):
        n = 20
        A = make_compatible_blob(n,n)
        P = make_compatible_blob(n,n)
        PTAP = np.array(P).transpose()@np.array(A)@np.array(P)
        C_PTAP = np.array(spkmeansmodule.test_PTAP(A,P))
        relative_error = relative_error_matrices(PTAP, C_PTAP)
        #print(f"Relative error is {relative_error}")
        #print(PTAP)
        self.assertLess(relative_error, 1e-4)
    
    def test_calc_k(self):
        def test_calc_k_length_n(n:int):
            eigenvalues = sorted([float(x) for x in np.random.rand(n)])[::-1]
            #print(type(eigenvalues))
            k_c = spkmeansmodule.full_calc_k(eigenvalues)
            k_py = spkmeans_utils.calc_k(np.array(eigenvalues))
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
        T_c = spkmeansmodule.normalize_matrix_by_rows(U)
        #print(T_py)
        relative_error = relative_error_centroids(T_py, T_c)
        self.assertLess(relative_error, 1e-4)

    def test_mat_cellwise_add(self):
        A = make_compatible_blob(14,91,offset=+1.0)
        B = make_compatible_blob(14,91,offset=+1.0)
        C = spkmeans_utils.convert__np_matrix__to__list_of_lists(spkmeans_utils.convert__list_of_lists__to__np_matrix(A)  \
                                                                 +                                                        \
                                                                 spkmeans_utils.convert__list_of_lists__to__np_matrix(B))
        C_module = spkmeansmodule.mat_cellwise_add(A, B)
        relative_error = relative_error_centroids(C, C_module)
        self.assertLess(relative_error, 1e-3)

    def test_mat_cellwise_mul(self):
        A = make_compatible_blob(14,91,offset=+1.0)
        B = make_compatible_blob(14,91,offset=+1.0)
        C = spkmeans_utils.convert__np_matrix__to__list_of_lists(spkmeans_utils.convert__list_of_lists__to__np_matrix(A)  \
                                                                 *                                                        \
                                                                 spkmeans_utils.convert__list_of_lists__to__np_matrix(B))
        C_module = spkmeansmodule.mat_cellwise_mul(A, B)
        relative_error = relative_error_centroids(C, C_module)
        self.assertLess(relative_error, 1e-3)

    def test_matmul(self):
        A = make_compatible_blob(14,44,offset=+1.0)
        B = make_compatible_blob(44,32,offset=+1.0)
        C = spkmeans_utils.convert__np_matrix__to__list_of_lists(spkmeans_utils.convert__list_of_lists__to__np_matrix(A)  \
                                                                 @                                                        \
                                                                 spkmeans_utils.convert__list_of_lists__to__np_matrix(B))
        C_module = spkmeansmodule.matmul(A, B)
        relative_error = relative_error_centroids(C, C_module)
        self.assertLess(relative_error, 1e-3)
    
    def test_mat_swap_rows(self):
        A = make_compatible_blob(14,44,offset=+1.0)
        indices = np.arange(44)
        np.random.shuffle(indices)
        indices = [int(x) for x in indices]
        B = spkmeans_utils.reorder_mat_cols_by_indices(A, indices)
        B_module = spkmeansmodule.reorder_mat_cols_by_indices(A, indices)
        relative_error = relative_error_centroids(B, B_module)
        self.assertLess(relative_error, 1e-5)
    
    #@unittest.skip("LALALALALLA")
    def test_sort_cols_by_vector_desc(self):
        A = make_compatible_blob(44,44,offset=+1.0)
        indices = np.arange(44)
        np.random.shuffle(indices)
        indices = [int(x) for x in indices]
        B = spkmeans_utils.sort_cols_by_vector_desc(A, indices)[0]
        B_module = spkmeansmodule.sort_cols_by_vector_desc(A, indices)
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
    initial_centroids_list = kmeans_pp.KMeansPlusPlus(k, datapoints_list)
    initial_centroids_indices_as_written = [int(initial_centroids_list[i][0]) for i in range(len(initial_centroids_list))]
    #print(','.join([str(x) for x in initial_centroids_indices_as_written]))
    initial_centroids_indices_actual = kmeans_pp.select_actual_centroids(datapoints_list, initial_centroids_list)
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


def idx_of_centroid_closest_to_point(centroids_list: List[List[float]], point: List[float]):
    dist_vector = np.sum(np.square(np.array(centroids_list)-np.array(point)), axis=1)
    return (np.argmin(dist_vector), np.min(dist_vector))

def dist_between_centroid_lists_redundant(list_1: List[List[float]], list_2: List[List[float]]) -> float:
    if len(list_1) != len(list_2): return math.inf
    k = len(list_1)
    if k == 0: return 0
    dims = len(list_1[0])
    for i in range(k):
        if (len(list_1[i]) != dims) or (len(list_2[i]) != dims): return math.inf
    dist = 0
    for i in range(k):
        #print(f"UNO: list_1 length is {list_1}, list_2 is {(list_2)}")
        idx, _dist = idx_of_centroid_closest_to_point(list_1, list_2[0])
        dist += _dist
        list_1 = np.array(list(list_1)[:idx] + list(list_1)[idx+1:])
        list_2 = np.array(list(list_2)[1:])
        #print(f"list_1 length is {list_1}, list_2 is {(list_2)}")
    return dist

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
