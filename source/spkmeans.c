#include "spkmeans.h"


mat_t* calc_weighted_adjacency_matrix(const mat_t* data) {
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
            sum = pow(sum, 0.5);
            mat_set(W, i, j, sum);
        }
    }
    return W;
}




int main(int argc, char* argv[]) {
    
    return 0;
}