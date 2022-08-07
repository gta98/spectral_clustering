#!/usr/bin/env python3

import unittest
from unittest.mock import patch
import pickle
import logging
import sys
import numpy as np
import sklearn.cluster
import kmeans_pp
import kmeans_sk
from typing import List, Callable, Tuple
import math
import time
from sklearn.datasets import make_blobs
import spkmeansmodule
import utils as spkmeans_utils
import re
import os


def make_compatible_blob(n=100,d=10, offset=0) -> List[List[float]]:
    x, _ = make_blobs(n_samples=n, n_features=d)
    x: np.ndarray
    return [[z+offset for z in y] for y in list(x)]

def make_compatible_blob_symmetric(n=100) -> List[List[float]]:
    a = np.random.rand(n,n)
    bottom_left = np.tril(a)
    top_right = np.tril(a,-1).T
    m = bottom_left + top_right
    return [list(y) for y in list(m)]


class TestFit(unittest.TestCase):

    def test_make_compatible_blob_symmetric(self):
        x = make_compatible_blob_symmetric(100)
        for i in range(100):
            for j in range(100):
                self.assertEqual(x[i][j], x[j][i])

    @unittest.skip("This is no longer relevant")
    def test_numpy_to_numpy(self):
        # this tests: read_data, Mat_to_PyListListFloat, PyListListFloat_to_Mat, wrap__ndarray_to_list_of_lists
        n, d = 100, 10
        A = np.round(np.random.rand(n*d)*10,4).reshape((n,d))
        A_list = [[float(y) for y in x] for x in A]
        save_path = "./tmp_mat.txt"
        with open(f'{save_path}.orig', 'w') as f:
            for line in A_list:
                f.write(','.join([str(x) for x in line]) + '\n')
        #print(A_list)
        B_list = spkmeans_utils.numpy_to_numpy(A_list, save_path)
        B_list = [[round(y,4) for y in x] for x in B_list]
        os.system(f"rm {save_path}*")
        self.assertEqual(A_list, B_list)
    
    def _compare_c_and_py(self,
            name: str,
            datapoints: List[List[float]],
            ptr_py: Callable, ptr_c: Callable,
            ptr_comparator: Callable):
        #print()
        #print('\n'.join([','.join([str(round(y,4)) for y in x]) for x in datapoints]))
        result_py = ptr_py(datapoints)
        result_c = ptr_c(datapoints)
        #print()
        #print('\n'.join([','.join([str(round(y,4)) for y in x]) for x in result_py]))
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
        print("jacobi comparator started")
        print(vector_py)
        print(vector_c)
        print(mat_py)
        print(mat_c)
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
        print("\n\n\n\n\n\n\n\n")
        self._compare_c_and_py('jacobi', make_compatible_blob_symmetric(3),
            spkmeans_utils.full_jacobi, spkmeansmodule.full_jacobi, self._comparator_jacobi)
    
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
        A = make_compatible_blob(14,44,offset=+1.0)
        indices = np.arange(44)
        np.random.shuffle(indices)
        indices = [int(x) for x in indices]
        B = spkmeans_utils.sort_cols_by_vector_desc(A, indices)[0]
        B_module = spkmeansmodule.sort_cols_by_vector_desc(A, indices)
        relative_error = relative_error_centroids(B, B_module)
        self.assertLess(relative_error, 1e-5)

    @unittest.skip("Disable if too heavy")
    def test_c_and_sklearn_over_and_over(self):
        make_blobs()
        np.random.seed(0)
        for i in range(100):
            self.test_c_and_sklearn_equal()
            #self.test_py_and_mine_equal()
            pass


    @unittest.skip("We only care about correctness for now")
    def test_my_py_runtime_vs_sklearn(self):
        print("test_my_py_runtime_vs_sklearn() - START")
        for i in range(1000):
            np.random.seed(i)
            #print(f"hai {i}")
            fit_params = list(randomize_fit_params(k=1000, max_iter=1000, eps=None, point_count=150, dims_count=7))
            time_1, time_2, time_3 = time.time(), time.time(), time.time()
            time_1 = time.time()
            centroids_list_py = kmeans_pp.KmeansAlgorithm(*fit_params)
            time_2 = time.time()
            centroids_list_sk = kmeans_sk.KmeansAlgorithm(*fit_params)
            time_3 = time.time()
            delta_py = time_2-time_1
            delta_sk = time_3-time_2
            relative_sk_py = relative_error_centroids(centroids_list_sk, centroids_list_py)
            #print(f"delta_py/delta_sk={delta_py/delta_sk}")
            print(f"relative err = {relative_sk_py}")
            pass
        print("test_my_py_runtime_vs_sklearn() - END")


    @unittest.skip("Only needed once in a while")
    def test_equal_to_templates(self):
        def test_equal_to_template_idx(*args):
            print(f"test_equal_to_template_idx{args} - start")
            with patch('sys.argv', ["python3","blah.py"]+list([str(x) for x in args])):
                fit_params = list(kmeans_pp.extract_fit_params())
                initial_centroids_list = fit_params[0]

                centroids_list_desired = None
                file_expected = args[-1].replace("input","output")
                file_expected = re.sub('_db_\d', '', file_expected)
                with open(file_expected, 'r') as f:
                    s = f.read().split("\n")[:-1]
                    s = [x.split(",") for x in s]
                    s[0] = [int(y) for y in s[0]]
                    initial_centroids_list_desired = s[0]
                    centroids_list_desired = np.array([[float(y) for y in x] for x in s[1:]])

                self.assertTrue(np.all(np.array(initial_centroids_list)==np.array(initial_centroids_list_desired)))

                centroids_list_py = kmeans_pp.KmeansAlgorithm(*fit_params)
                centroids_list_sk = kmeans_sk.KmeansAlgorithm(*fit_params)

                centroids_list_py = np.array(centroids_list_py)
                centroids_list_sk = np.array(centroids_list_sk)

                dist_desired_py = dist_between_centroid_lists(centroids_list_desired, centroids_list_py)
                dist_py_sk      = dist_between_centroid_lists(centroids_list_py, centroids_list_sk)
                
                print(dist_desired_py)
                print(dist_py_sk)

                self.assertTrue(dist_desired_py == 0)
                #self.assertTrue(dist_desired_sk == 0)
    
        print("test_equal_to_templates() - start")
        pathloc = f"../resources/test_data_2"
        test_equal_to_template_idx(3,  333,                       0, f"{pathloc}/input_1_db_1.txt", f"{pathloc}/input_1_db_2.txt")
        test_equal_to_template_idx(7,  kmeans_pp.MAX_ITER_UNSPEC, 0, f"{pathloc}/input_2_db_1.txt", f"{pathloc}/input_2_db_2.txt")
        test_equal_to_template_idx(15, 750,                       0, f"{pathloc}/input_3_db_1.txt", f"{pathloc}/input_3_db_2.txt")


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
    list_1 = np.around(list_1, 4)
    list_2 = np.around(list_2, 4)
    return np.linalg.norm((list_1)-(list_2))

def relative_error_centroids(centroids_real: List[List[float]], centroids_calc: List[List[float]]) -> float:
    return np.linalg.norm(np.sort(centroids_real)-np.sort(centroids_calc))/np.linalg.norm(centroids_real)

if __name__ == '__main__':
    print("Starting tests")
    unittest.main()
