#include "wam.h"

real dist_between_rows(mat_t* A, uint i, uint j) {
    uint x;
    real sum;
    real diff;

    if (i==j) return 0.0;
    
    sum = 0;

    for (x = 0; x < A->w; x++) {
        diff = mat_get(A,i,x)-mat_get(A,j,x);
        sum += diff*diff;
    }

    return real_sqrt(sum);
}

mat_t* calc_wam(mat_t* data) {
    mat_t* W;
    uint i, j, n;
    real sum;

    n = data->h;
    W = mat_init(n, n);
    if (!W) return NULL;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            sum = dist_between_rows(data,i,j);
            sum = sum / ((real)-2.0);
            sum = exp(sum);
            mat_set(W, i, j, sum);
        }
    }

    for (i=0; i<n; i++) mat_set(W, i, i, 0);

    return W;
}