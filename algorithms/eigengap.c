#include "eigengap.h"

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
    /*printd("sorting indices: ");
    for (i=0; i<n; i++) {
        printd("%d, ", sorting_indices[i]);
    }
    printd("\n");*/

    /* before reordering eigenvalues:
    ** v'[*,i] will get v[*, sorting_indices[i]]
    ** we wanna make sure v'[i,i] = eigenval
    ** meaning v[i, sorting_indices[i]] = v[sorting_indices[i],sorting_indices[i]]
    */
    /*printd("lala 1:\n");
    mat_print(v);*/
    for (i=0; i<n; i++) {
        mat_set(v, i, sorting_indices[i],
                mat_get(v, sorting_indices[i], sorting_indices[i]));
    }
    /*printd("\n");
    printd("lala 2:\n");
    mat_print(v);*/
    
    status = reorder_mat_cols_by_indices(v, sorting_indices);
    if (status != SUCCESS) goto sort_cols_by_vector_desc_finish;
    
    /*printd("lala 3:\n");
    mat_print(v);

    printd("HOLD THE LINE!");
    mat_print_diagonal(v);
    printd("\nLove isnt always on time\n");*/

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
    assertd(is_square(eigenvalues));
    /*assertd(is_diagonal(eigenvalues));*/
    /*assertd(eigenvalues->h == 1);
    assertd(eigenvalues->w >= 2);*/
    n = eigenvalues->w;
    half_n = floor(((double)n)/2);
    assertd(half_n >= 1);
    max_eigengap_idx = -1;
    max_eigengap_val = -1;
    for (i=0; i<half_n; i++) {
        eigengap_val = mat_get(eigenvalues,i,i)-mat_get(eigenvalues,i+1,i+1);
        assertd(eigengap_val>=0);
        if (eigengap_val > max_eigengap_val) {
            max_eigengap_idx = i;
            max_eigengap_val = eigengap_val;
        }
    }
    return max_eigengap_idx+1;
}