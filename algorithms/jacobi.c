#include "jacobi.h"

void calc_c_s(mat_t* A, uint i, uint j, real* c, real* s) {
    real theta, t;
    real theta_sign, A_i_i, A_i_j, A_j_j;
    A_i_i=mat_get(A,i,i); A_i_j=mat_get(A,i,j); A_j_j=mat_get(A,j,j);
    theta_sign = real_sign(A_j_j-A_i_i)*real_sign(A_i_j);
    if (A_i_j != 0) {
        theta = (A_j_j-A_i_i) / (((real)2.0)*A_i_j);
        t = theta_sign / (real_abs(theta) + sqrt((real)1 + (theta*theta)));
    } else {
        if (A_j_j == A_i_i) {
            theta = 1;
            t = theta_sign / (real_abs(theta) + sqrt((real)1 + (theta*theta)));
        } else {
            /* theta = inf */
            t = 0;
        }
    }
    *c = real_pow((real)1 + (t*t), (real)-0.5);
    *s = t*(*c);
}

void calc_P_ij(mat_t* P, mat_t* A, uint i, uint j) {
    uint k, l;
    real c, s;
    assertd_is_square(A);
    assertd_same_dims(A, P);
    assertd(i<j);

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
    assertd_is_square(A);
    *i = 0; *j = 1;
    max_val = real_abs(mat_get(A, *i, *j));
    for (k=0; k<A->h-1; k++) {
        for (l=(k+1); l<A->w; l++) {
            val = real_abs(mat_get(A, k, l));
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

void calc_A_tag(mat_t* A_tag, mat_t* A, uint i, uint j, real c, real s) {
    uint r;
    real A_r_i, A_r_j;
    /*real A_i_r, A_j_r;*/
    assertd(i<j);

    mat_copy_to(A_tag, A);

    for (r=0; r<A->h; r++) {
        if ((r == i) || (r == j)) continue;
        A_r_i = mat_get(A,r,i), A_r_j = mat_get(A,r,j);
        mat_set(A_tag, r, i, (c*A_r_i)-(s*A_r_j));
        mat_set(A_tag, r, j, (s*A_r_i)+(c*A_r_j));
        mat_set(A_tag, i, r, mat_get(A_tag,r,i));
        mat_set(A_tag, j, r, mat_get(A_tag,r,j));
    }

    mat_set(A_tag, i, i, (c*c*mat_get(A,i,i)+(s*s*mat_get(A,j,j))-(2*s*c*mat_get(A,i,j))));
    mat_set(A_tag, j, j, (s*s*mat_get(A,i,i)+(c*c*mat_get(A,j,j))+(2*s*c*mat_get(A,i,j))));
    mat_set(A_tag, i, j, 0);
    mat_set(A_tag, j, i, 0);

}

void calc_V(mat_t* V, uint i, uint j, real c, real s) {
    uint r;
    real V_r_i, V_r_j;
    assertd(i<j);

    for (r=0; r<V->h; r++) {
        V_r_i = mat_get(V, r, i), V_r_j = mat_get(V, r, j);
        mat_set(V, r, i, (c*V_r_i)-(s*V_r_j));
        mat_set(V, r, j, (s*V_r_i)+(c*V_r_j));
    }
}

void perform_A_V_iteration(mat_t* A_tag, mat_t* A, mat_t* V) {
    uint i, j;
    real c, s;
    assertd(V); assertd(A); assertd(A_tag);
    assertd_same_dims(A_tag, A); assertd_same_dims(A, V);

    get_indices_of_max_element(A, &i, &j);
    calc_c_s(A, i, j, &c, &s);

    /* V = V @ P */
    calc_V(V, i, j, c, s);

    /* A_tag = P.transpose() @ A @ P */
    calc_A_tag(A_tag, A, i, j, c, s);
}

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
    uint n, rotations;

    A_tag = NULL;
    A = NULL;
    V = NULL;

    *eigenvectors = NULL;
    *eigenvalues = NULL;

    assertd(is_square(A_original));
    n = A_original->h;

    A_tag = mat_init_copy(A_original);
    A = mat_init_copy(A_original);
    V = mat_init_identity(n);
    if (!A_tag || !V || !A) {
        if (A_tag) mat_free(&A_tag);
        if (V) mat_free(&V);
        if (A) mat_free(&A);
        *eigenvectors = NULL;
        *eigenvalues = NULL;
        return;
    }
    *eigenvectors = V;
    *eigenvalues = A_tag;

    rotations = 0;
    while (true) {
        perform_A_V_iteration(A_tag, A, V);

        rotations += 1;

        if (is_jacobi_convergence(A_tag, A, rotations)) break;

        mat_copy_to(A, A_tag);
    }

    if (A) mat_free(&A);
}