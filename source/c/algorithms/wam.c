#include "wam.h"

mat_t* calc_wam(mat_t* data) {
    mat_t* W;
    uint i, j, k, n, d;
    real sum;

    n = data->h;
    d = data->w;
    W = mat_init(n, n);
    if (!W) return NULL;
    printd("n is %d, d is %d\n",n,d);
    printd("mat is:\n");
    mat_print(data);

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            sum = 0;
            for (k=0; k<d; k++) {
                sum += pow((mat_get(data, i, k) - mat_get(data, j, k)), 2);
            }
            sum = sqrt(sum);
            sum = sum / 2;
            printd("cursum3: %f\n",sum);
            sum = exp(-1 * sum);
            mat_set(W, i, j, sum);
        }
    }

    for (i=0; i<n; i++) mat_set(W, i, i, 0);

    return W;
}