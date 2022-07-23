
from ctypes.wintypes import DWORD
from typing import List, Tuple
import numpy as np
import os
import sys
import pandas as pd
import mykmeanssp
from definitions import *


def transpose_1d(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        return x.reshape((x.shape[0],1))
    elif len(x.shape) == 2:
        assert(1 in x.shape)
        return x.reshape((x.shape[1],x.shape[2]))
    else:
        raise ValueError("Unsupported!")


def identity_matrix_like(A: np.ndarray) -> np.ndarray:
    assert(A.ndim == 2)
    return np.diag(np.ones(np.min(A.shape)))


def is_1d_vector(x: np.ndarray) -> bool:
    return (x.ndim == 1)


def is_square_matrix(A: np.ndarray) -> bool:
    return (A.ndim == 2) and (A.shape[0] == A.shape[1])


def is_diagonal_matrix(A: np.ndarray) -> bool:
    return (np.count_nonzero(x - np.diag(np.diagonal(x))) == 0)


def calc_wam(datapoints: np.ndarray) -> np.ndarray:
    # returns weighted adjacency matrix
    assert(datapoints.ndim == 2)
    n, d = datapoints.shape
    datapoints_horizontal = datapoints.reshape((1,n,d)) #datapoints_horizontal = datapoints.squeeze()
    datapoints_vertical = datapoints_horizontal.swapaxes(0,1)
    datapoints_delta = datapoints_vertical - datapoints_horizontal
    datapoints_dist_squared = np.sum(datapoints_delta**2, axis=2)
    W_with_diagonal = np.exp(-1 * (datapoints_dist_squared / 2))
    W = W_with_diagonal - np.diag(W_with_diagonal.diagonal())
    return W


def calc_ddg(W: np.ndarray) -> np.ndarray:
    W_sum_along_horizontal_axis = W.sum(axis=1)
    D = np.diag(W_sum_along_horizontal_axis)
    return D


def calc_L_norm(W: np.ndarray, D: np.ndarray) -> np.ndarray:
    D_pow_minus_half = D**(-0.5)
    DWD = (D_pow_minus_half @ W @ D_pow_minus_half)
    I = np.diag(np.ones_like(DWD))
    L_norm = I-DWD
    return L_norm


def sign(num: int) -> int:
    if num == 0:
        return 1
    else:
        return np.sign(num)


def calc_P_ij(A: np.ndarray, i: int, j: int) -> np.ndarray:
    assert(len(A.ndim)==2)
    P = identity_matrix_like(A)
    theta = (A[j,j] - A[i,i]) / (2*A[i,j])
    t = sign(theta) / (np.abs(theta) + np.sqrt(1 + (theta**2)))
    c = 1 / np.sqrt(1 + (t**2))
    s = t*c
    P[i,i] = P[j,j] = c
    P[i,j] = s
    P[j,i] = -s
    return P


def get_indices_of_max_element(A: np.ndarray) -> Tuple:
    return np.unravel_index(np.argmax(A, axis=None), A.shape)


def calc_P(A: np.ndarray) -> np.ndarray:
    assert(len(A.ndim)==2)
    A_abs = np.abs(A)
    largest_abs_element_location: Tuple = get_indices_of_max_element(A_abs)
    assert(len(largest_abs_element_location) == 2)
    P = calc_P_ij(A, *largest_abs_element_location)
    return P


def calc_off_squared(A: np.ndarray) -> np.ndarray:
    assert(is_square_matrix(A))
    A_without_diagonal = A - np.diag(A.diagonal())
    A_without_diagonal__square = A_without_diagonal**2
    A_without_diagonal__square__sum = A_without_diagonal__square.sum()
    return A_without_diagonal__square__sum


def jacobi_algorithm(A_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert(is_square_matrix(A_original))
    V = identity_matrix_like(A_original)
    A = np.array(A_original)
    rotations = 0
    while True:
        P = calc_P(A)
        assert(P.shape == A.shape)
        rotations += 1
        V = V @ P
        assert(V.shape == A.shape)
        A_tag = V.transpose() @ A_original @ V
        assert(A_tag.shape == A.shape)
        dist_between_offs = calc_off_squared(A)-calc_off_squared(A_tag)
        assert(dist_between_offs >= 0) # see forum: https://moodle.tau.ac.il/mod/forum/discuss.php?d=125232
        if (dist_between_offs <= JACOBI_EPSILON) or (rotations == JACOBI_MAX_ROTATIONS):
            break
        A = A_tag
    eigenvalues = A_tag.diagonal()
    eigenvectors = V
    return eigenvalues, eigenvectors


def sort_cols_by_vector_desc(A: np.ndarray, v: np.array) -> Tuple[np.ndarray, np.array]:
    sorting_indices = np.argsort(v)[::-1]
    v_sorted = v[sorting_indices]
    A_sorted = A[:, sorting_indices]
    return A_sorted, v_sorted


def calc_eigengap(eigenvalues_sort_dec: np.ndarray) -> np.ndarray:
    eigenvalues_i_plus_1 = eigenvalues_sort_dec[1:]
    eigenvalues_i = eigenvalues_sort_dec[:-1]
    delta_abs = np.abs(eigenvalues_i - eigenvalues_i_plus_1)
    return delta_abs


def calc_k(datapoints: np.ndarray) -> np.ndarray:
    n = len(datapoints)
    L_norm = calc_L_norm(datapoints)
    eigenvalues, eigenvectors = jacobi_algorithm(L_norm)
    eigenvectors_sort_dec, eigenvalues_sort_dec = sort_cols_by_vector_desc(eigenvectors, eigenvalues)
    assert(n == eigenvalues_sort_dec.shape[0])
    assert(n >= 2) # eigengap is undefined for n in {0,1}
    half_n = int(np.floor(n/2))
    delta_abs = calc_eigengap(eigenvalues_sort_dec)
    delta_max = np.max(delta_abs[:half_n])
    for i in range(half_n):
        if delta_abs[i] == delta_max:
            return i
    raise Exception("We were supposed to return")