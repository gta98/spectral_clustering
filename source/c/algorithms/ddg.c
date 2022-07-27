#include "ddg.h"

mat_t* calc_ddg(mat_t* W) {
    mat_t* D;
    uint i, j, n;
    real sum;

    n = W->h;
    assertd(n == W->w);

    D = mat_init_full(n, n, 0);
    for (i=0; i<n; i++) {
        sum = 0;
        for (j=0; j<n; j++) {
            sum += mat_get(W, i, j);
        }
        mat_set(D, i, i, sum);
    }

    return D;
}

mat_t* calc_ddg_inv_sqrt(mat_t* W) {
    mat_t* D;
    uint i;
    D = calc_ddg(W);
    if (!D) return NULL;
    assertd_is_square(D);
    for (i=0; i<D->h; i++) {
        mat_set(D, i, i, 1/sqrt(mat_get(D, i, i)));
    }
    return D;
}