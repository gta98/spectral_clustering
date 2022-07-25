#include "spkmeans.h"


mat_t* calc_wam(const mat_t* data) {
    mat_t* W;
    uint i, j, k, n, d;
    real sum;

    n = data->h;
    d = data->w;
    W = mat_init(n, n);
    if (!W) return NULL;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            sum = 0;
            for (k=0; k<d; k++) {
                sum += pow((mat_get(data, i, k) - mat_get(data, j, k)), 2);
            }
            sum = exp(-1 * (pow(sum, 0.5) / 2));
            mat_set(W, i, j, sum);
        }
    }

    for (i=0; i<n; i++) mat_set(W, i, i, 0);

    return W;
}

mat_t* calc_ddg(const mat_t* W) {
    mat_t* D;
    uint i, j, k, n, d;
    real sum;

    n = W->h;
    assert(n == W->w);

    D = mat_init_full(n, n, 0);
    for (i=0; i<n; i++) {
        sum = 0;
        for (j=0; j<n; j++) {
            sum += mat_get(W, i, j);
        }
        mat_set(D, i, i, sum);
    }
}

mat_t* calc_ddg_inv_sqrt(const mat_t* W) {
    mat_t* D;
    uint i, n;
    D = calc_ddg(W);
    if (!D) return NULL;
    assert(D->h == D->w);
    for (i=0; i<D->h; i++) {
        mat_set(D, i, i, 1/sqrt(mat_get(D, i, i)));
    }
    return D;
}

mat_t* calc_lnorm(const mat_t* W, const mat_t* D_inv_sqrt) {
    mat_t* L;
    uint i, n;
    n = W->h;
    assert((n == W->h) && (n == W->w));
    assert((n == D_inv_sqrt->h) && (n == D_inv_sqrt->w));
    L = mat_init(n,n);
    if (!L) return NULL;
    mat_mul(L, W, D_inv_sqrt);
    mat_mul(L, D_inv_sqrt, L);
    mat_mul_scalar(L, L, -1);
    for (i=0; i<n; i++) {
        mat_set(L, i, i, 1 + mat_get(L, i, i));
    }
    return L;
}

mat_t* calc_full_wam(const mat_t* data) {
    mat_t* W;
    W = calc_wam(data);
    return W;
}

mat_t* calc_full_ddg(const mat_t* data) {
    mat_t* W;
    mat_t* D;
    W = calc_wam(data);
    if (!W) return NULL;
    D = calc_ddg(W);
    mat_free(&W);
    return D;
}

mat_t* calc_full_lnorm(const mat_t* data) {
    mat_t* W;
    mat_t* D_inv_sqrt;
    mat_t* L_norm;
    W = calc_wam(data);
    if (!W) return NULL;
    D_inv_sqrt = calc_ddg_inv_sqrt(W);
    if (!D_inv_sqrt) {
        mat_free(&W);
        return NULL;
    }
    L_norm = calc_lnorm(W, D_inv_sqrt);
    mat_free(&W);
    mat_free(&D_inv_sqrt);
    return L_norm;
}


/*
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
*/

mat_t* calc_P_ij(mat_t* A, uint i, uint j) {
    mat_t* P;
    real theta, t, c, s;
    assert((A->h) == (A->w));
    P = mat_init_identity(A->h);
    if (!P) return NULL;
    theta = (mat_get(A,j,j) - mat_get(A,i,i)) / (2*mat_get(A,i,j));
    t = real_sign(theta) / (real_abs(theta) + sqrt(1 + pow(theta, 2)));
    c = 1 / sqrt(1 + pow(t, 2));
    s = t*c;
    mat_set(P, i, i, c);
    mat_set(P, j, j, c);
    mat_set(P, i, j, s);
    mat_set(P, j, i, -s);
    return P;
}

/* V = V @ P_ij with c, s */
void perform_V_iteration_ij_cs(mat_t* V, uint i, uint j, real c, real s) {
    real new_value;
    uint k, n;
    assert(V && (V->h == V->w));
    n = V->h;
    /* deal with i'th col in V */
    for (k=0; k<n; k++) {
        new_value = (c*mat_get(V,k,i))-(s*mat_get(V,k,j));
        mat_set(V, k, i, new_value);
    }

    /* deal with j'th col in V */
    for (k=0; k<n; k++) {
        new_value = (s*mat_get(V,k,i))+(c*mat_get(V,k,j));
        mat_set(V, k, j, new_value);
    }
}

/* V = V @ P_ij with A */
void perform_V_iteration_ij(mat_t* V, uint i, uint j, const mat_t* A) {
    real theta, t, c, s;
    assert((A->h) == (A->w));
    theta = (mat_get(A,j,j) - mat_get(A,i,i)) / (2*mat_get(A,i,j));
    t = real_sign(theta) / (real_abs(theta) + sqrt(1 + pow(theta, 2)));
    c = 1 / sqrt(1 + pow(t, 2));
    s = t*c;
    perform_V_iteration_ij_cs(V, i, j, c, s);
}

void get_indices_of_max_element(mat_t* A, uint* i, uint* j) {
    uint k, l;
    real val, max_val;
    max_val = NEG_INF;
    for (k=0; k<A->h; k++) {
        for (l=0; l<A->w; l++) {
            val = mat_get(A, k, l);
            if (val > max_val) {
                max_val = val;
                *i = k;
                *j = l;
            }
        }
    }
}

void perform_V_iteration(mat_t* V, mat_t* A) {
    uint i=-1, j=-1;
    get_indices_of_max_element(A, &i, &j);
    assert(i>=0);
    assert(j>=0);
    perform_V_iteration_ij(V, i, j, A);
}

void perform_A_V_iteration(mat_t* A_tag, mat_t* A, mat_t* V) {
    assert(A_tag && V && A);
    assert(A_tag->h == A_tag->w);
    assert(A_tag->w == V->h);
    assert(V->h == V->w);
    assert(V->w == A->h);
    assert(A->h == A->w);
    perform_V_iteration(V, A);
    mat_mul(A_tag, A, V);
    mat_transpose(V);
    mat_mul(A_tag, V, A_tag);
    mat_transpose(V);
}

void calc_jacobi(mat_t* A, mat_t** eigenvectors, mat_t** eigenvalues) {
    mat_t* A_tag;
    mat_t* V;
    uint n, rotations;
    assert(A->h == A->w);
    n = A->h;
    A_tag = mat_init_like(A);
    V = mat_init_identity(n);
    if (!A_tag || !V) {
        if (A_tag) mat_free(&A_tag);
        if (V) mat_free(&V);
        *eigenvectors = NULL;
        *eigenvalues = NULL;
        return;
    }
    *eigenvectors = V;
    *eigenvalues = A_tag;

    do {
        perform_A_V_iteration(A_tag, A, V);
        A = A_tag;
        rotations++;
    } while (
        !is_jacobi_convergence(A, A_tag, rotations)
    );
}

// TODO - finish sort_cols_by_vector_desc, calc_eigengap, calc_k

real calc_off_squared(mat_t* A) {
    uint i, j, n;
    real off;
    assert((A->h) == (A->w));
    n = A->h;
    off = 0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (i!=j)
                off += pow(mat_get(A,i,j),2);
        }
    }
    return off;
}

real calc_dist(mat_t* A, mat_t* A_tag) {
    return calc_off_squared(A) - calc_off_squared(A_tag);
}

bool is_jacobi_convergence(mat_t* A, mat_t* A_tag, uint rotations) {
    if (rotations >= JACOBI_MAX_ROTATIONS) {
        return true;
    } else {
        if (calc_dist(A, A_tag) < JACOBI_EPSILON) {
            return true;
        } else {
            return false;
        }
    }
}



mat_t* calc_P(mat_t* A) {
    mat_t* P;
    assert((A->h) == (A->w));
}



int main(int argc, char* argv[]) {
    
    return 0;
}