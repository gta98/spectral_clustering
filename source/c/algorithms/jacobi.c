#include "jacobi.h"

/*mat_t* calc_P_ij(mat_t* A, uint i, uint j) {
    mat_t* P;
    real theta, t, c, s;
    assertd((A->h) == (A->w));
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
}*/

void calc_c_s(mat_t* A, uint i, uint j, real* c, real* s) {
    real theta, t;
    theta = (mat_get(A,j,j) - mat_get(A,i,i)) / (((double)2)*mat_get(A,i,j));
    t = real_sign(theta) / (real_abs(theta) + sqrt((double)1 + (double)pow(theta, 2)));
    *c = ((double)1) / sqrt((double)1 + (double)pow(t, 2));
    *s = t*(*c);
}

void calc_P_ij(mat_t* P, mat_t* A, uint i, uint j) {
    uint k, l;
    real c, s;
    assertd_is_square(A);
    assertd_same_dims(A, P);

    for (k=0; k<P->h; k++) {
        for (l=0; l<P->w; l++) {
            mat_set(P, k, l, 0);
        }
    }

    for (k=0; k<P->h; k++) {
        mat_set(P, k, k, 1);
    }

    calc_c_s(A, i, j, &c, &s);

    mat_set(P, i, i, c);
    mat_set(P, j, j, c);
    mat_set(P, i, j, s);
    mat_set(P, j, i, -s);
}

void get_indices_of_max_element(mat_t* A, uint* i, uint* j) {
    uint k, l;
    real val, max_val;
    assertd((A->h >= 2) || (A->w >= 2)); /* we want the max OFF DIAGONAL element */
    if (A->h >= 2) {
        max_val = mat_get(A,1,0);
        *i = 1, *j = 0;
    } else if (A->w >= 2) {
        max_val = mat_get(A,0,1);
        *i = 0, *j = 1;
    } else {
        printd("This is impossible - assertion failed\n");
        exitd(1);
    }
    for (k=0; k<A->h; k++) {
        for (l=0; l<A->w; l++) {
            if (k==l) continue;
            val = mat_get(A, k, l);
            if (val < 0) val *= (real) -1;
            if (val > max_val) {
                max_val = val;
                *i = k;
                *j = l;
            }
        }
    }
}

void calc_P_inplace(mat_t* P, mat_t* A) {
    uint i, j;
    assertd(P); assertd(A);
    assertd_is_square(P);
    assertd_same_dims(P, A);

    i=0, j=0;
    get_indices_of_max_element(A, &i, &j);

    calc_P_ij(P, A, i, j);
}

mat_t* calc_P(mat_t* A) {
    mat_t* P;
    P = mat_init_like(A);
    if (!P) return NULL;
    calc_P_inplace(P, A);
    return P;
}

/* V = V @ P_ij with c, s */
void perform_V_iteration_ij_cs(mat_t* V, uint i, uint j, real c, real s) {
    real new_value;
    uint k, n;
    assertd(V); assertd_is_square(V);
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
void perform_V_iteration_ij(mat_t* V, uint i, uint j, mat_t* A) {
    real theta, t, c, s;
    assertd_is_square(A);
    theta = (mat_get(A,j,j) - mat_get(A,i,i)) / (2*mat_get(A,i,j));
    t = real_sign(theta) / (real_abs(theta) + sqrt(1 + pow(theta, 2)));
    c = 1 / sqrt(1 + pow(t, 2));
    s = t*c;
    perform_V_iteration_ij_cs(V, i, j, c, s);
}

void perform_V_iteration(mat_t* V, mat_t* A) {
    uint i, j;
    i=0, j=0;
    get_indices_of_max_element(A, &i, &j);
    perform_V_iteration_ij(V, i, j, A);
}

void calc_A_tag(mat_t* A_tag, mat_t* A) {
    uint i, j, r;
    real c, s;
    i=0, j=0;
    get_indices_of_max_element(A, &i, &j);
    calc_c_s(A, i, j, &c, &s);

    mat_copy_to(A_tag, A);
    for (r=0; r<A->h; r++) {
        if ((r != i) && (r != j)) {
            mat_set(A_tag, r, i, (c*mat_get(A,r,i))-(s*mat_get(A,r,j)));
            mat_set(A_tag, r, j, (c*mat_get(A,r,j))+(s*mat_get(A,r,i)));
        }
    }
    mat_set(A_tag, i, i,
            ((c*c)*mat_get(A,i,i)) + ((s*s)*mat_get(A,j,j)) - (((real)2)*s*c*mat_get(A,i,j)));
    mat_set(A_tag, j, j,
            ((s*s)*mat_get(A,i,i)) + ((c*c)*mat_get(A,j,j)) + (((real)2)*s*c*mat_get(A,i,j)));
    mat_set(A_tag, i, j, 0);
}

void perform_A_V_iteration(mat_t* A_tag, mat_t* A, mat_t* A_original, mat_t* V, mat_t* P, mat_t* tmp) {
    assertd(A_original); assertd(V); assertd(A); assertd(A_tag); assertd(tmp);
    assertd_is_square(A_original);
    assertd_same_dims(A_original, V); assertd_same_dims(V, A); assertd_same_dims(A, A_tag);

    calc_P_inplace(P, A);

    /* V = V @ P */
    mat_mul(tmp, V, P);
    mat_copy_to(V, tmp);

    /* A_tag = P.transpose() @ A @ P */
    /*mat_transpose(P);
    mat_mul(A_tag, P, A);
    mat_transpose(P);
    mat_mul(A_tag, A_tag, P);*/
    mat_mul(A_tag, A_original, V);
    mat_transpose(V);
    mat_mul(A_tag, V, A_original);
    mat_transpose(V);
    /*calc_A_tag(A_tag, A);*/
}

/* TODO - finish sort_cols_by_vector_desc, calc_eigengap, calc_k */


real calc_dist_between_offs(mat_t* A_tag, mat_t* A) {
    return calc_off_squared(A) - calc_off_squared(A_tag);
}

bool is_jacobi_convergence(mat_t* A_tag, mat_t* A, uint rotations) {
    if (rotations >= JACOBI_MAX_ROTATIONS) {
        return true;
    } else {
        if (calc_dist_between_offs(A_tag, A) <= JACOBI_EPSILON) {
            return true;
        } else {
            return false;
        }
    }
}


void calc_jacobi(mat_t* A_original, mat_t** eigenvectors, mat_t** eigenvalues) {
    mat_t* A_tag;
    mat_t* A;
    mat_t* V;
    mat_t* P;
    mat_t* tmp;
    uint n, rotations;

    A_tag = NULL;
    A = NULL;
    V = NULL;
    P = NULL;

    *eigenvectors = NULL;
    *eigenvalues = NULL;

    assertd(is_square(A_original));
    n = A_original->h;

    P = mat_init_copy(A_original);
    A_tag = mat_init_copy(A_original);
    A = mat_init_copy(A_original);
    V = mat_init_identity(n);
    tmp = mat_init(n,n);
    if (!A_tag || !V || !A || !P || !tmp) {
        if (A_tag) mat_free(&A_tag);
        if (V) mat_free(&V);
        if (A) mat_free(&A);
        if (P) mat_free(&P);
        if (tmp) mat_free(&tmp);
        *eigenvectors = NULL;
        *eigenvalues = NULL;
        return;
    }
    *eigenvectors = V;
    *eigenvalues = A_tag;

    rotations = 0;
    while (true) {
        perform_A_V_iteration(A_tag, A, A_original, V, P, tmp);

        rotations += 1;

        if (is_jacobi_convergence(A_tag, A, rotations)) break;

        mat_copy_to(A, A_tag);
    }

    /* these are no longer relevant */
    if (P) mat_free(&P);
    if (A) mat_free(&A);
    if (tmp) mat_free(&tmp);
}