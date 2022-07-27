#include "ddg.h"

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