
from ctypes.wintypes import DWORD
from typing import List, Tuple, NoReturn, Union
import numpy as np
import os
import sys
import pandas as pd
import mykmeanssp
from definitions import *

def assertd(condition:bool) -> Union[None, NoReturn]:
    assert(condition)

def sign(num: int) -> int:
    if num == 0:
        return 1
    else:
        return np.sign(num)

def can_convert_to_list_of_list(A: np.ndarray) -> bool:
    if type(A) != np.ndarray:
        return False
    if A.ndim != 2:
        return False
    return True

def can_convert_to__np_matrix(A: List[List[float]]):
    if type(A) != list:
        return False
    if any([(type(x)!=list) or (len(x)!=len(A[0])) for x in A]):
        return False
    return True

def convert__np_matrix__to__list_of_lists(A: np.ndarray) -> List[List[float]]:
    if not can_convert_to_list_of_list(A):
        raise ValueError("A must be a 2d matrix")
    return [list(x) for x in list(A)]

def convert__list_of_lists__to__np_matrix(A: List[List[float]]) -> np.ndarray:
    if not can_convert_to__np_matrix(A):
        raise ValueError("Cannot convert A to np matrix")
    B = np.array(A)
    assertd(B.ndim == 2)
    return B

def wrap__ndarray_to_list_of_lists(func):
    def inner(*args, **kwargs):
        if np.ndarray in [type(x) for x in args]+[type(x) for x in kwargs.values()]:
            # working in numpy mode
            return func(*args, **kwargs)
        _args = [(convert__list_of_lists__to__np_matrix(v) if can_convert_to__np_matrix(v) else v) \
                for v in args]
        _kwargs = {k: (convert__list_of_lists__to__np_matrix(v) if can_convert_to__np_matrix(v) else v) \
                for k,v in kwargs.items()}
        retval = func(*_args, **_kwargs)
        if can_convert_to_list_of_list(retval):
            return convert__np_matrix__to__list_of_lists(retval)
        elif (type(retval) is tuple) and all([type(x) is np.ndarray for x in retval]):
            return tuple([convert__np_matrix__to__list_of_lists(x) for x in retval])
        else:
            return retval
    return inner
            

@wrap__ndarray_to_list_of_lists
def transpose_1d(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        return x.reshape((x.shape[0],1))
    elif len(x.shape) == 2:
        assertd(1 in x.shape)
        return x.reshape((x.shape[1],x.shape[2]))
    else:
        raise ValueError("Unsupported!")


@wrap__ndarray_to_list_of_lists
def identity_matrix_like(A: np.ndarray) -> np.ndarray:
    assertd(A.ndim == 2)
    return np.diag(np.ones(np.min(A.shape)))


@wrap__ndarray_to_list_of_lists
def is_1d_vector(x: np.ndarray) -> bool:
    return (x.ndim == 1)


@wrap__ndarray_to_list_of_lists
def is_square_matrix(A: np.ndarray) -> bool:
    return (A.ndim == 2) and (A.shape[0] == A.shape[1])


@wrap__ndarray_to_list_of_lists
def is_diagonal_matrix(A: np.ndarray) -> bool:
    return (np.count_nonzero(A - np.diag(np.diagonal(A))) == 0)


@wrap__ndarray_to_list_of_lists
def calc_wam(datapoints: np.ndarray) -> np.ndarray:
    # returns weighted adjacency matrix
    assertd(datapoints.ndim == 2)
    n, d = datapoints.shape
    datapoints_horizontal = datapoints.reshape((1,n,d)) #datapoints_horizontal = datapoints.squeeze()
    datapoints_vertical = datapoints_horizontal.swapaxes(0,1)
    datapoints_delta = datapoints_vertical - datapoints_horizontal
    datapoints_dist_squared = np.sqrt(np.sum(datapoints_delta**2, axis=2))
    W_with_diagonal = np.exp(-1 * (datapoints_dist_squared / 2))
    W = W_with_diagonal - np.diag(W_with_diagonal.diagonal())
    return W


@wrap__ndarray_to_list_of_lists
def calc_ddg(W: np.ndarray) -> np.ndarray:
    W_sum_along_horizontal_axis = W.sum(axis=1)
    D = np.diag(W_sum_along_horizontal_axis)
    return D


@wrap__ndarray_to_list_of_lists
def calc_L_norm(W: np.ndarray, D: np.ndarray) -> np.ndarray:
    D_pow_minus_half = D**(-0.5)
    DWD = (D_pow_minus_half @ W @ D_pow_minus_half)
    I = np.diag(np.ones_like(DWD))
    L_norm = I-DWD
    return L_norm


@wrap__ndarray_to_list_of_lists
def calc_P_ij(A: np.ndarray, i: int, j: int) -> np.ndarray:
    assertd(len(A.ndim)==2)
    P = identity_matrix_like(A)
    theta = (A[j,j] - A[i,i]) / (2*A[i,j])
    t = sign(theta) / (np.abs(theta) + np.sqrt(1 + (theta**2)))
    c = 1 / np.sqrt(1 + (t**2))
    s = t*c
    P[i,i] = P[j,j] = c
    P[i,j] = s
    P[j,i] = -s
    return P


@wrap__ndarray_to_list_of_lists
def get_indices_of_max_element(A: np.ndarray) -> Tuple:
    return np.unravel_index(np.argmax(A, axis=None), A.shape)


@wrap__ndarray_to_list_of_lists
def calc_P(A: np.ndarray) -> np.ndarray:
    assertd(len(A.ndim)==2)
    A_abs = np.abs(A)
    largest_abs_element_location: Tuple = get_indices_of_max_element(A_abs)
    assertd(len(largest_abs_element_location) == 2)
    P = calc_P_ij(A, *largest_abs_element_location)
    return P


@wrap__ndarray_to_list_of_lists
def calc_off_squared(A: np.ndarray) -> np.ndarray:
    assertd(is_square_matrix(A))
    A_without_diagonal = A - np.diag(A.diagonal())
    A_without_diagonal__square = A_without_diagonal**2
    A_without_diagonal__square__sum = A_without_diagonal__square.sum()
    return A_without_diagonal__square__sum


@wrap__ndarray_to_list_of_lists
def jacobi_algorithm(A_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assertd(is_square_matrix(A_original))
    V = identity_matrix_like(A_original)
    A = np.array(A_original)
    rotations = 0
    while True:
        P = calc_P(A)
        assertd(P.shape == A.shape)
        rotations += 1
        V = V @ P
        assertd(V.shape == A.shape)
        A_tag = V.transpose() @ A_original @ V
        assertd(A_tag.shape == A.shape)
        dist_between_offs = calc_off_squared(A)-calc_off_squared(A_tag)
        assertd(dist_between_offs >= 0) # see forum: https://moodle.tau.ac.il/mod/forum/discuss.php?d=125232
        if (dist_between_offs <= JACOBI_EPSILON) or (rotations == JACOBI_MAX_ROTATIONS):
            break
        A = A_tag
    eigenvalues = A_tag.diagonal()
    eigenvectors = V
    return eigenvalues, eigenvectors


@wrap__ndarray_to_list_of_lists
def sort_cols_by_vector_desc(A: np.ndarray, v: np.array) -> Tuple[np.ndarray, np.array]:
    sorting_indices = np.argsort(v)[::-1]
    v_sorted = v[sorting_indices]
    A_sorted = A[:, sorting_indices]
    return A_sorted, v_sorted


@wrap__ndarray_to_list_of_lists
def calc_eigengap(eigenvalues_sort_dec: np.ndarray) -> np.ndarray:
    eigenvalues_i_plus_1 = eigenvalues_sort_dec[1:]
    eigenvalues_i = eigenvalues_sort_dec[:-1]
    delta_abs = np.abs(eigenvalues_i - eigenvalues_i_plus_1)
    return delta_abs


@wrap__ndarray_to_list_of_lists
def calc_k(datapoints: np.ndarray) -> np.ndarray:
    n = len(datapoints)
    L_norm = calc_L_norm(datapoints)
    eigenvalues, eigenvectors = jacobi_algorithm(L_norm)
    eigenvectors_sort_desc, eigenvalues_sort_desc = sort_cols_by_vector_desc(eigenvectors, eigenvalues)
    assertd(n == eigenvalues_sort_desc.shape[0])
    assertd(n >= 2) # eigengap is undefined for n in {0,1}
    half_n = int(np.floor(n/2))
    delta_abs = calc_eigengap(eigenvalues_sort_desc)
    delta_max = np.max(delta_abs[:half_n])
    for i in range(half_n):
        if delta_abs[i] == delta_max:
            return i
    raise Exception("We were supposed to return")


@wrap__ndarray_to_list_of_lists
def numpy_to_numpy(datapoints: np.ndarray, save_path: str) -> np.ndarray:
    # takes datapoints as np array, returns datapoints as np array
    # 1. Py converts numpy into List[List[float]]
    # 2. Py sends to C as List[List[float]] and save_path
    # 3. C converts List[List[float]] to double**
    # 4. C writes double** into save_path
    # 5. Py asks C to read matrix from save_path
    # 6. C parses double** from save_path using matrix_reader
    # 7. C converts double** to List[List[float]]
    # 8. Py converts List[List[float]] to numpy
    # this also tests the wrapper
    # if this works, we confirm read_data, Mat_to_PyListListFloat, PyListListFloat_to_Mat, wrap__ndarray_to_list_of_lists
    import spkmeansmodule
    datapoints = [list(x) for x in datapoints] # 1
    print("blargh")
    spkmeansmodule.test_write_data(datapoints, save_path) # 2, 3, 4
    print("bloorgh")
    datapoints_tag = spkmeansmodule.test_read_data(save_path) # 5, 6, 7
    #datapoints_tag = datapoints
    return datapoints_tag # 8


def full_wam(datapoints: List[List[float]]) -> List[List[float]]:
    W = calc_wam(datapoints)
    return W


def full_ddg(datapoints: List[List[float]]) -> List[List[float]]:
    W = full_wam(datapoints)
    D = calc_ddg(W)
    return D


def full_lnorm(datapoints: List[List[float]]) -> List[List[float]]:
    W = full_wam(datapoints)
    D = calc_ddg(W)
    Lnorm = calc_L_norm(W, D)
    return Lnorm


def full_jacobi(datapoints: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
    eigenvalues, eigenvectors = jacobi_algorithm(datapoints)
    return eigenvalues, eigenvectors