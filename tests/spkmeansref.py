"""
This module provides reference methods that are meant to replicate spkmeansmodule
"""

from cmath import inf
from ctypes.wintypes import DWORD
from typing import List, Tuple, NoReturn, Union
import numpy as np
import os
import sys
#from definitions import *

INFINITY = float('inf')
JACOBI_MAX_ROTATIONS = 100
JACOBI_EPSILON = 1e-5
KMEANS_EPSILON = 1e-5 # FIXME - what should this be? 0?
KMEANS_MAX_ITER = 300 # this was verified to be 300

def assertd(condition:bool) -> Union[None, NoReturn]:
    assert(condition)

def sign(num: int) -> int:
    assertd(num != np.nan)
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
    if type(A) == np.ndarray:
        return True
    if type(A) != list:
        raise ValueError(f"NOT A LIST BUT A {type(A)}")
        return False
    if type(A[0]) == list:
        for i in range(len(A)):
            if type(A[i]) != list:
                raise ValueError(f"A in index {i} is not a list but a {type(A[i])}, A is {A}")
                return False
            if len(A[i]) != len(A[0]):
                raise ValueError("Length is not identical")
                return False
    elif type(A[0]) in {float,int}:
        for i in range(len(A)):
            if type(A[i]) != type(A[0]):
                raise ValueError(f'Not a {type(A[0])} in index {i}: but a {type(A[i])}')
                return False
    else:
        raise ValueError(f'Unrecognized type {type(A[0])}')
        return False
    return True

def convert__np_matrix__to__list_of_lists(A: np.ndarray) -> List[List[float]]:
    #if not can_convert_to_list_of_list(A):
    #    raise ValueError("A must be a 2d matrix")
    return A.tolist()#[[float(y) for y in x] for x in list(A)]

def convert__list_of_lists__to__np_matrix(A: List[List[float]]) -> np.ndarray:
    #if not can_convert_to__np_matrix(A):
    #    raise ValueError(f"Cannot convert A to np matrix because: ")
    B = np.array(A)
    return B

def wrap__ndarray_to_list_of_lists(func):
    def inner(*args, **kwargs):
        if np.ndarray in [type(x) for x in args]+[type(x) for x in kwargs.values()]:
            # working in numpy mode
            return func(*args, **kwargs)
        _args = [(convert__list_of_lists__to__np_matrix(v)) \
                for v in args]
        _kwargs = {k: (convert__list_of_lists__to__np_matrix(v)) \
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
    assertd(is_square_matrix(A))
    return np.eye(*A.shape)


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
def is_symmetric(A: np.ndarray) -> bool:
    return (np.count_nonzero(A - A.transpose()) == 0)


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

# simple edition
@wrap__ndarray_to_list_of_lists
def calc_wam_2(datapoints: np.ndarray) -> np.ndarray:
    # returns weighted adjacency matrix
    import math
    assertd(datapoints.ndim == 2)
    n, d = datapoints.shape
    W = np.zeros((n,n))
    for i in range(n):
        x_i = datapoints[i,:]
        for j in range(n):
            x_j = datapoints[j,:]
            wij = 0
            for k in range(d):
                wij += round((x_i[k] - x_j[k])**2,3)
            wij = round(math.sqrt(wij),3)
            wij = round(wij*(-0.5),3)
            wij = round(math.exp(wij),3)
            W[i,j] = round(wij,3)
    for i in range(n):
        W[i,i] = 0
    return W


@wrap__ndarray_to_list_of_lists
def calc_ddg(W: np.ndarray) -> np.ndarray:
    W_sum_along_horizontal_axis = W.sum(axis=1)
    D = np.diag(W_sum_along_horizontal_axis)
    return D

@wrap__ndarray_to_list_of_lists
def calc_ddg_inv_sqrt(W: np.ndarray) -> np.ndarray:
    D = calc_ddg(W)
    D_inv_sqrt = np.diag(np.power(np.diag(D), -0.5))
    return D_inv_sqrt


@wrap__ndarray_to_list_of_lists
def calc_L_norm(W: np.ndarray, D_pow_minus_half: np.ndarray) -> np.ndarray:
    #D_pow_minus_half = np.diag(np.diag(D)**(-0.5)) # removes inf's off diag and replaces with zeros
    DWD = (D_pow_minus_half @ W @ D_pow_minus_half)
    assertd(DWD.shape[0] == DWD.shape[1])
    I = identity_matrix_like(DWD)
    L_norm = I-DWD
    return L_norm


@wrap__ndarray_to_list_of_lists
def calc_P_ij(A: np.ndarray, i: int, j: int) -> np.ndarray:
    assertd(A.ndim==2)
    assertd(i<j)
    P = identity_matrix_like(A)
    c,s = calc_c_s(A,i,j)
    #print(f"Py: c={c}, s={s}")
    P[i,i] = P[j,j] = c
    P[i,j] = s
    P[j,i] = -s
    return P


@wrap__ndarray_to_list_of_lists
def get_indices_of_max_element(A: np.ndarray) -> Tuple:
    #assertd(is_symmetric(A))
    i, j = -1, -1
    max_val = -1*inf
    for k in range(A.shape[0]):
        for l in range(k+1, A.shape[1]):
            val = np.abs(A[k,l])
            if val > max_val:
                i, j = k, l
                max_val = val
    assert i<j
    return min(i,j), max(i,j)


@wrap__ndarray_to_list_of_lists
def calc_P(A: np.ndarray) -> np.ndarray:
    assertd(A.ndim==2)
    #assertd(is_symmetric(A))
    largest_abs_element_location: Tuple = get_indices_of_max_element(A)
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

def calc_dist_between_offs(A_tag: np.ndarray, A: np.ndarray) -> float:
    return calc_off_squared(A)-calc_off_squared(A_tag)

def is_jacobi_convergence(A_tag: np.ndarray, A: np.ndarray, rotations: int) -> bool:
    dist_between_offs = calc_dist_between_offs(A_tag, A)
    assertd(dist_between_offs >= 0) # see forum: https://moodle.tau.ac.il/mod/forum/discuss.php?d=125232
    #print(f"dist_between_offs={dist_between_offs}, rotations={rotations}")
    return (dist_between_offs <= JACOBI_EPSILON) or (rotations >= JACOBI_MAX_ROTATIONS)

def calc_c_s(A: np.ndarray, i:int, j:int) -> Tuple[float,float]:
    A_j_j, A_i_i, A_i_j = A[j,j], A[i,i], A[i,j]
    theta_sign = 1
    if A_j_j-A_i_i < 0:
        theta_sign *= -1
    if A_i_j < 0:
        theta_sign *= -1
    if A_j_j-A_i_i == 0 and A_i_j != 0:
        theta_sign = 1
    if (A_i_j != 0):
        theta = (A_j_j - A_i_i) / (2*A_i_j)
        t = theta_sign / (np.abs(theta) + np.sqrt(1+(theta*theta)))
        c = 1/np.sqrt(1+(t*t))
        s = t*c
    else:
        if (A_j_j == A_i_i):
            theta = 1
            t = theta_sign / (np.abs(theta) + np.sqrt(1+(theta*theta)))
            c = 1/np.sqrt(1+(t*t))
            s = t*c
        else:
            # theta = inf
            t = 0
            c = 1
            s = 0
    return c, s

def calc_A_tag(A: np.ndarray) -> np.ndarray:
    A_tag = np.copy(A)
    i, j = get_indices_of_max_element(A_tag)
    c, s = calc_c_s(A,i,j)

    for r in range(A_tag.shape[0]):
        if r not in {i,j}:
            A_tag[r,i]=(c*A[r,i])-(s*A[r,j])
            A_tag[i,r]=A_tag[r,i]
        if r == i:
            A_tag[i,i]=(c*c*A[i,i])+(s*s*A[j,j])-(2*s*c*A[i,j])
    for r in range(A_tag.shape[0]):
        if r not in {i,j}:
            A_tag[r,j]=(c*A[r,j])+(s*A[r,i])
            A_tag[j,r]=A_tag[r,j]
        if r == j:
            A_tag[j,j]=(s*s*A[i,i])+(c*c*A[j,j])+(2*s*c*A[i,j])
    A_tag[i,j]=0
    A_tag[j,i]=0
    return A_tag

#@wrap__ndarray_to_list_of_lists
def jacobi_algorithm(A_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assertd(is_square_matrix(A_original))
    assertd(is_symmetric(A_original))
    V = identity_matrix_like(A_original)
    A = np.copy(A_original)
    rotations = 0
    while True:
        P = calc_P(A)
        V = V @ P
        A_tag = P.T @ A @ P
        rotations += 1
        if is_jacobi_convergence(A_tag, A, rotations): break
        A = A_tag
    eigenvalues = A_tag.diagonal()
    eigenvectors = V
    return eigenvalues, eigenvectors

@wrap__ndarray_to_list_of_lists
def reorder_mat_cols_by_indices(A: np.ndarray, v: np.array) -> np.ndarray:
    return A[:, np.array(v)]


#@wrap__ndarray_to_list_of_lists
def sort_cols_by_vector_desc(A: np.ndarray, v: np.array) -> Tuple[np.ndarray, np.array]:
    sorting_indices = np.argsort(v)[::-1]
    v_sorted = np.array(v)[sorting_indices]
    A_sorted = np.array(A)[:, sorting_indices]
    A_sorted = [[float(y) for y in x] for x in A_sorted]
    v_sorted = [float(y) for y in v_sorted]
    return A_sorted, v_sorted


@wrap__ndarray_to_list_of_lists
def calc_k(eigenvalues: np.array) -> np.ndarray:
    n = len(eigenvalues)
    if n <= 1: return n
    half_n = int(np.floor(n/2))
    delta = -np.diff(eigenvalues) #np.diff(A)[i]==A[i+1]-A[i]
    assertd(np.all(delta>=0))
    return 1+np.argmax(delta[:half_n])


#@wrap__ndarray_to_list_of_lists
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
    spkmeansmodule.test_write_data(datapoints, save_path) # 2, 3, 4
    datapoints_tag = spkmeansmodule.test_read_data(save_path) # 5, 6, 7
    #datapoints_tag = datapoints
    return np.array(datapoints) # 8


def full_wam(datapoints: List[List[float]]) -> List[List[float]]:
    W = calc_wam(datapoints)
    return W


def full_ddg(datapoints: List[List[float]]) -> List[List[float]]:
    W = full_wam(datapoints)
    D = calc_ddg(W)
    return D


def full_lnorm(datapoints: List[List[float]]) -> List[List[float]]:
    W = full_wam(datapoints)
    D_pow_minus_half = calc_ddg_inv_sqrt(W)
    Lnorm = calc_L_norm(W, D_pow_minus_half)
    return Lnorm


def full_jacobi(datapoints: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
    datapoints = np.array(datapoints)
    eigenvalues, eigenvectors = jacobi_algorithm(datapoints)
    return eigenvalues.tolist(), eigenvectors.tolist()


def full_jacobi_sorted(datapoints: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
    datapoints = convert__list_of_lists__to__np_matrix(datapoints)
    eigenvalues, eigenvectors = jacobi_algorithm(datapoints)
    eigenvectors, eigenvalues = sort_cols_by_vector_desc(eigenvectors, eigenvalues)
    eigenvalues = [float(x) for x in eigenvalues]
    return eigenvalues, eigenvectors


@wrap__ndarray_to_list_of_lists
def normalize_matrix_by_rows(U: np.ndarray) -> np.ndarray:
    U_square_sum = np.sqrt(np.sum(np.square(U), axis=1))
    normalized = (U.transpose()/U_square_sum).transpose()
    normalized = np.nan_to_num(normalized,0)
    return normalized


#@wrap__ndarray_to_list_of_lists
def full_spk_1_to_5(datapoints: List[List[float]], k: int) -> List[List[float]]:
    datapoints = np.array(datapoints)
    L_norm = np.around(full_lnorm(datapoints),4)
    eigenvalues, eigenvectors = full_jacobi_sorted(L_norm)
    if k==0: k = calc_k(eigenvalues)
    U = [x[:k] for x in eigenvectors]
    T = np.array(normalize_matrix_by_rows(U))
    return T.tolist()

def full_spk_1_to_5_ref(datapoints: List[List[float]], k: int) -> List[List[float]]:
    import spkmeansmodule
    datapoints = [[y for y in x] for x in datapoints]
    L_norm = spkmeansmodule.full_lnorm(datapoints)
    eigenvalues, eigenvectors = spkmeansmodule.full_jacobi_sorted(L_norm)
    if k==0: k = spkmeansmodule.full_calc_k(eigenvalues)
    U = [x[:k] for x in eigenvectors]
    T = spkmeansmodule.normalize_matrix_by_rows(U)