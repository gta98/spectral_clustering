#include "jacobi.h"

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

/* TODO - finish sort_cols_by_vector_desc, calc_eigengap, calc_k */

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