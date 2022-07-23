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
from typing import List
import math
import time
from sklearn.datasets import make_blobs
import re


class TestFit(unittest.TestCase):

    @unittest.skip("Disable if too heavy")
    def test_c_and_sklearn_over_and_over(self):
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
            with open('/home/ubuntu/lala2.bin','wb') as f:
                f.write(pickle.dumps(fit_params))
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


    #@unittest.skip("Only needed once in a while")
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
    return np.linalg.norm((list_1)-(list_2))

def relative_error_centroids(centroids_real: List[List[float]], centroids_calc: List[List[float]]) -> float:
    return np.linalg.norm(np.sort(centroids_real)-np.sort(centroids_calc))/np.linalg.norm(centroids_real)

if __name__ == '__main__':
    print("Starting tests")
    unittest.main()
