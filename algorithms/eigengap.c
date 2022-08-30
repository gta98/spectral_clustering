#include "eigengap.h"

void reverse_list(uint* l, uint n) {
    uint i;
    for (i=0; i<((uint)(n/2)); i++) {
        l[i]=l[n-1-i];
    }
}

status_t sort_cols_by_vector_desc(mat_t* A, mat_t* v) {
    real* eigenvalues;
    status_t status;
    uint* sorting_indices;
    uint n, i;

    assertd_is_square(A);
    assertd_same_dims(A, v);

    sorting_indices = NULL;
    eigenvalues = NULL;
    n = v->h;
    eigenvalues = malloc(sizeof(real)*n);
    if (!eigenvalues) {
        status = ERROR_MALLOC;
        goto sort_cols_by_vector_desc_finish;
    }
    for (i=0; i<n; i++) eigenvalues[i] = mat_get(v,i,i);
    
    sorting_indices = argsort(eigenvalues, n);
    if (!sorting_indices) {
        status = ERROR_MALLOC;
        goto sort_cols_by_vector_desc_finish;
    }

    #ifdef FLAG_DEBUG_REVERSE_SORTING_INDICES
    reverse_list(sorting_indices, n);
    #endif

    for (i=0; i<n; i++) {
        mat_set(v, i, sorting_indices[i],
                mat_get(v, sorting_indices[i], sorting_indices[i]));
    }
    
    status = reorder_mat_cols_by_indices(v, sorting_indices);
    if (status != SUCCESS) goto sort_cols_by_vector_desc_finish;

    status = reorder_mat_cols_by_indices(A, sorting_indices);
    if (status != SUCCESS) goto sort_cols_by_vector_desc_finish;
    
    sort_cols_by_vector_desc_finish:
    if (sorting_indices) free(sorting_indices);
    if (eigenvalues) free(eigenvalues);
    return status;
}

uint calc_k(mat_t* eigenvalues) {
    uint i, n, half_n;
    uint max_eigengap_idx;
    real max_eigengap_val, eigengap_val;
    /*assertd(is_diagonal(eigenvalues));*/
    /*assertd(eigenvalues->h == 1);*/
    n = eigenvalues->w;
    half_n = floor(((double)n)/2);
    assertd(half_n >= 1);
    max_eigengap_idx = -1;
    max_eigengap_val = -1;
    if (eigenvalues->h > 1) {
        assertd_is_square(eigenvalues);
        for (i=0; i<half_n; i++) {
            eigengap_val = mat_get(eigenvalues,i,i)-mat_get(eigenvalues,i+1,i+1);
            #ifndef FLAG_DEBUG_REVERSE_SORTING_INDICES
            assertd(eigengap_val>=0); 
            #endif
            eigengap_val = real_abs(eigengap_val);
            if (eigengap_val > max_eigengap_val) {
                max_eigengap_idx = i;
                max_eigengap_val = eigengap_val;
            }
        }
    } else if (eigenvalues->h == 1) {
        for (i=0; i<half_n; i++) {
            eigengap_val = mat_get(eigenvalues,0,i)-mat_get(eigenvalues,0,i+1);
            #ifndef FLAG_DEBUG_REVERSE_SORTING_INDICES
            assertd(eigengap_val>=0); 
            #endif
            eigengap_val = real_abs(eigengap_val);
            if (eigengap_val > max_eigengap_val) {
                max_eigengap_idx = i;
                max_eigengap_val = eigengap_val;
            }
        }
    }
    return max_eigengap_idx+1;
}