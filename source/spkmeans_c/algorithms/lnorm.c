#include "lnorm.h"

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