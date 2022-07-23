
from typing import List, Tuple
import numpy as np

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

def calc_wam(datapoints: np.ndarray) -> np.ndarray:
    # returns weighted adjacency matrix
    assert(datapoints.ndim == 1)
    datapoints_horizontal = datapoints #datapoints_horizontal = datapoints.squeeze()
    datapoints_vertical = transpose_1d(datapoints_horizontal)
    x_dist = datapoints_vertical - datapoints_horizontal
    w_with_diagonal = np.exp(-1 * ((x_dist ** 2) / 2))
    w = w_with_diagonal - np.diag(w_with_diagonal.diagonal())
    return w

def calc_ddg(datapoints: np.ndarray) -> np.ndarray:
    w = calc_wam(datapoints)
    w_sum_along_horizontal_axis = w.sum(axis=1)
    d = np.diag(w_sum_along_horizontal_axis)
    return d

def calc_lnorm(datapoints: np.ndarray) -> np.ndarray:
    I = np.diag(np.ones_like(datapoints))
    W = calc_wam(datapoints)
    D = calc_ddg(datapoints) # could make this run faster by reusing W
    D_pow_minus_half = D**(-0.5)
    L_norm = (I - (D_pow_minus_half @ W @ D_pow_minus_half))
    return L_norm

def sign(num: int) -> int:
    if num == 0:
        return 1
    else:
        return np.sign(num)
    
def calc_P_ij(A: np.ndarray, i: int, j: int) -> np.ndarray:
    assert(len(A.ndim)==2)
    I = identity_matrix_like(A)
    theta = (A[j,j] - A[i,i]) / (2*A[i,j])
    t = sign(theta) / (np.abs(theta) + np.sqrt(1 + (theta**2)))
    c = 1 / np.sqrt(1 + (t**2))
    s = t*c
    P = np.array(I)
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
    A_without_diagonal = A - np.diag(A.diagonal())
    A_without_diagonal__square = A_without_diagonal**2
    A_without_diagonal__square__sum = A_without_diagonal__square.sum()
    return A_without_diagonal__square__sum

def calc_dist_between_offs(A: np.ndarray, A_tag: np.ndarray) -> bool:
    assert(is_square_matrix(A))
    assert(is_square_matrix(A_tag))
    assert(A.shape == A_tag.shape)
    off_delta = calc_off_squared(A)-calc_off_squared(A_tag) # FIXME - can off(A) be less than off(A')?
    return off_delta

def calc_A_tag(A: np.ndarray) -> np.ndarray:
    P = calc_P(A)
    P_T = P.transpose()
    A_tag = P_T @ A @ P
    return A_tag


if __name__ == "__main__":
    x = np.round((np.random.rand(10)*10)+1)
    calc_wam(x)
    pass