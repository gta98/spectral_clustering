#include "lnorm.h"

mat_t* calc_lnorm(mat_t* W, mat_t* D_inv_sqrt) {
    mat_t* L;
    uint i, j, n;
    assertd_is_square(W);
    assertd_same_dims(W, D_inv_sqrt);
    n = W->h;
    L = mat_init(n,n);
    if (!L) return NULL;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            mat_set(L,i,j, mat_get(D_inv_sqrt,i,i)*mat_get(W,i,j)*mat_get(D_inv_sqrt,j,j));
        }
    }
    
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            mat_set(L, i, j, ((real)-1)*mat_get(L,i,j));
        }
    }

    for (i=0; i<n; i++) {
        mat_set(L, i, i, 1.0);
    }
    
    return L;
}