#include "spkmeans.h"

int main(int argc, char* argv[]) {
    mat_t* data;
    goal_t goal;
    status_t status;

    if (argc != 3) {
        return get_code_print_msg(1, MSG_ERR_INVALID_INPUT);
    }

    goal = get_selected_routine(argv[1]);
    if (goal == INVALID_GOAL) {
        return get_code_print_msg(1, MSG_ERR_INVALID_INPUT);
    }

    data = NULL;
    status = read_data(&data, argv[2]);
    if (status != SUCCESS) {
        if (data) free(data);
        switch(status) {
            case ERROR_FOPEN: {
                return get_code_print_msg(1, MSG_ERR_INVALID_INPUT);
            }
            case ERROR_MALLOC: {
                return get_code_print_msg(1, MSG_ERR_GENERIC);
            }
            case ERROR_FORMAT: {
                return get_code_print_msg(1, MSG_ERR_GENERIC);
            }
            default: {
                break;
            }
        }
        assertd(false); /* this is not supposed to happen */
        return get_code_print_msg(1, MSG_ERR_GENERIC);
    }

    assertd(status == SUCCESS);
    assertd(data);

    if ((goal == JACOBI) && (data->h != data->w)) {
        if (data) free(data);
        return get_code_print_msg(1, MSG_ERR_GENERIC);
    }

    /* goal, data is validated */

    switch(goal) {
        case WAM: {
            status = print_wam(data);
            break;
        }
        case DDG: {
            status = print_ddg(data);
            break;
        }
        case LNORM: {
            status = print_lnorm(data);
            break;
        }
        case JACOBI: {
            status = print_jacobi(data);
            break;
        }
        default: {
            status = INVALID_GOAL;
            break;
        }
    }

    free(data);

    if (status != SUCCESS) {
        return get_code_print_msg(1, MSG_ERR_GENERIC);
    }
    
    return 0;
}

int get_code_print_msg(int code, const char* msg) {
    printf("%s", msg);
    return code;
}

goal_t get_selected_routine(const char* goal_str) {
    uint i;
    for (i=0; i<GOALS_COUNT; i++) {
        if (streq_insensitive(goal_str, GOALS[i])) {
            return i;
        }
    }
    return INVALID_GOAL;
}

status_t print_wam(mat_t* data) {
    mat_t* w = calc_full_wam(data);
    if (!w) return ERROR;
    mat_print(w);
    mat_free(&w);
    return SUCCESS;
}

status_t print_ddg(mat_t* data) {
    mat_t* d = calc_full_ddg(data);
    if (!d) return ERROR;
    mat_print(d);
    mat_free(&d);
    return SUCCESS;
}

status_t print_lnorm(mat_t* data) {
    mat_t* lnorm = calc_full_lnorm(data);
    if (!lnorm) return ERROR;
    mat_print(lnorm);
    mat_free(&lnorm);
    return SUCCESS;
}

status_t print_jacobi(mat_t* data) {
    mat_t* eigenvectors;
    mat_t* eigenvalues;
    eigenvectors = NULL;
    eigenvalues = NULL;

    calc_full_jacobi(data, &eigenvectors, &eigenvalues);
    if (!eigenvalues || !eigenvectors) {
        if (eigenvectors) mat_free(&eigenvectors);
        if (eigenvalues) mat_free(&eigenvalues);
        return ERROR;
    }

    mat_print_diagonal(eigenvalues);
    mat_print(eigenvectors);

    mat_free(&eigenvectors);
    mat_free(&eigenvalues);
    return SUCCESS;
}

mat_t* calc_full_wam(mat_t* data) {
    mat_t* W;
    W = calc_wam(data);
    return W;
}

mat_t* calc_full_ddg(mat_t* data) {
    mat_t* W;
    mat_t* D;
    W = calc_wam(data);
    if (!W) return NULL;
    D = calc_ddg(W);
    mat_free(&W);
    return D;
}

mat_t* calc_full_lnorm(mat_t* data) {
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

void calc_full_jacobi(mat_t* data, mat_t** eigenvectors, mat_t** eigenvalues) {
    calc_jacobi(data, eigenvectors, eigenvalues);
}