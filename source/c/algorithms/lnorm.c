#include "lnorm.h"

mat_t* calc_lnorm(mat_t* W, mat_t* D_inv_sqrt) {
    mat_t* L;
    mat_t* tmp;
    uint i, j, n;
    real Lij;
    assertd_is_square(W);
    assertd_same_dims(W, D_inv_sqrt);
    n = W->h;
    L = mat_init(n,n);
    if (!L) return NULL;
    
    tmp = mat_init(n,n);
    if (!tmp) {
        mat_free(&L);
        return NULL;
    }

    mat_mul(tmp, W, D_inv_sqrt);
    mat_mul(L, D_inv_sqrt, tmp);
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            Lij = mat_get(L,i,j);
            mat_set(L,i,j,((real)-1)*Lij);
        }
    }
    for (i=0; i<n; i++) {
        Lij = mat_get(L, i, i);
        mat_set(L, i, i, Lij + 1);
    }

    if (tmp) mat_free(&tmp);
    return L;
}