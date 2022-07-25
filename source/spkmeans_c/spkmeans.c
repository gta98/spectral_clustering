#include "spkmeans.h"

mat_t* calc_full_wam(const mat_t* data) {
    mat_t* W;
    W = calc_wam(data);
    return W;
}

mat_t* calc_full_ddg(const mat_t* data) {
    mat_t* W;
    mat_t* D;
    W = calc_wam(data);
    if (!W) return NULL;
    D = calc_ddg(W);
    mat_free(&W);
    return D;
}

mat_t* calc_full_lnorm(const mat_t* data) {
    mat_t* W;
    mat_t* D_inv_sqrt;
    mat_t* L_norm;
    W = calc_wam(data);
    if (!W) return NULL;
    D_inv_sqrt = calc_ddg_inv_sqrt(W);
    if (!D_inv_sqrt) {
        mat_free(&W);
        return NULL;
    }
    L_norm = calc_lnorm(W, D_inv_sqrt);
    mat_free(&W);
    mat_free(&D_inv_sqrt);
    return L_norm;
}

mat_t* calc_full_jacobi(const mat_t* data) {
    
}



int main(int argc, char* argv[]) {
    
    return 0;
}