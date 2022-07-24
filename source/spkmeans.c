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






int main(int argc, char* argv[]) {
    
    return 0;
}