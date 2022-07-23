import os
import sys
from typing import List
import numpy as np
import pandas as pd
import mykmeanssp
from definitions import *


def main_kmeanspp(k, datapoints):
    np.random.seed(0)
    fit_params = extract_fit_params(k, KMEANS_MAX_ITER, KMEANS_EPSILON, datapoints)
    results = KmeansAlgorithm(*fit_params)
    return results


def extract_fit_params(*data_from_cmd, should_print=True):
    k, max_iter, eps, datapoints_list = data_from_cmd
    initial_centroids_list = KMeansPlusPlus(k, datapoints_list)
    initial_centroids_indices_as_written = [int(initial_centroids_list[i][0]) for i in range(len(initial_centroids_list))]
    if should_print: print(','.join([str(x) for x in initial_centroids_indices_as_written]))
    initial_centroids_indices_actual = select_actual_centroids(datapoints_list, initial_centroids_list)
    datapoints_list = [list(x) for x in list(datapoints_list[:,1:])] # remove index, convert to List[List[float]] for C
    dims_count = len(datapoints_list[0])
    point_count = len(datapoints_list)
    return (
        initial_centroids_indices_actual,
        datapoints_list,
        dims_count,
        k,
        point_count,
        max_iter,
        eps
    )


def KMeansPlusPlus(k: int, x: np.array) -> List[int]:
    np.random.seed(0)
    x = np.array(x)
    N, d = x.shape
    u = [None for _ in range(k)]
    u_idx = [-1 for _ in range(N)]
    P = [0 for _ in range(N)]
    D = [float('inf') for _ in range(N)]

    i = 0
    selection = np.random.choice(x[:,0])
    u[0] = x[np.where(x[:,0]==selection)]

    while (i+1) < k:
        for l in range(N):
            x_l = x[l] # remove index
            min_square_dist = float('inf')
            for j in range(0,i+1):
                u_j = u[j][0,:] # u.shape = (1,u.shape[0]) -> (u.shape[0],)
                square_dist = np.sum((x_l[1:] - u_j[1:])**2) # first item is an index
                min_square_dist = min(square_dist, min_square_dist)
            D[l] = min_square_dist
        D_sum = sum(D)
        if D_sum <= 0:
            raise Exception(f"Somehow reached D_sum = {D_sum}, but P values must be nonnegative and finite")
        P = D/D_sum

        i += 1
        selection = np.random.choice(x[:,0], p=P)
        u[i] = x[np.where(x[:,0]==selection)]
        continue

    centroids_without_padding = [a[0] for a in u]
    return centroids_without_padding


def select_actual_centroids(data: List[List[float]], initial_centroids_list: List[List[float]]) -> List[int]:
    # incase we have duplicates, etc...
    initial_centroids_indices_actual = [None for centroid in initial_centroids_list]
    for i, centroid in enumerate(initial_centroids_list):
        loc = np.where(np.all(data==centroid,axis=1))[0] #[0] because this returns a tuple
        if len(loc) == 0: # or len(loc)>=2?
            raise GenericErrorTrigger(f"There should only be one match among the datapoints for every initial centroid, but one got {len(loc)} matches")
        initial_centroids_indices_actual[i] = loc[0]
    return initial_centroids_indices_actual


def KmeansAlgorithm(*fit_params) -> List[List[float]]:
    return mykmeanssp.fit(*fit_params)