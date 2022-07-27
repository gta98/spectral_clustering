#include "generics/common_utils.h"

real real_add(real lhs, real rhs) { return lhs + rhs; }
real real_sub(real lhs, real rhs) { return lhs - rhs; }
real real_mul(real lhs, real rhs) { return lhs * rhs; }
real real_div(real lhs, real rhs) { return lhs / rhs; }

/* lhs ^ rhs */
real real_pow(real lhs, real rhs) { return pow(lhs, rhs); }

/* rhs ^ lhs */
real real_pow_rev(real lhs, real rhs) { return pow(rhs, lhs); }

real real_abs(real x) { return (x>=0) ? x : -x; }

real real_sign(real x) {
    return (x >= 0) ? 1 : -1;
}

int isanum(char* s) {
    int i;
    for (i = 0; s[i] != 0; i++) {
        if (!(('0' <= s[i]) && (s[i] <= '9')))
            return 0;
    }
    return 1;
}

/* argsort copied from: https://stackoverflow.com/questions/36714030/c-sort-float-array-while-keeping-track-of-indices */

typedef struct {
    real value;
    int index;
} validx_t;

int cmp(const void *a, const void *b)
{
    validx_t* a1;
    validx_t* a2;
    a1 = (validx_t*) a;
    a2 = (validx_t*) b;
    if ((*a1).value > (*a2).value)
        return -1;
    else if ((*a1).value < (*a2).value)
        return 1;
    else
        return 0;
}

uint* argsort(const real* v, const uint n) {
    validx_t* indices;
    uint* indices_uint;
    uint i;
    assertd(v);

    indices = NULL;
    indices_uint = NULL;
    indices = malloc(sizeof(validx_t)*n);
    indices_uint = malloc(sizeof(uint)*n);
    if (!indices || !indices_uint) {
        if (indices) free(indices);
        if (indices_uint) free(indices_uint);
        return NULL;
    }

    for (i=0; i<n; i++) {
        indices[i].index = i;
        indices[i].value = v[i];
    }

    qsort(indices, n, sizeof(validx_t), cmp);
    
    for (i=0; i<n; i++) {
        indices_uint[i] = indices[i].index;
    }
    free(indices);
    return indices_uint;
}

uint* argsort_desc(const real* v, const uint n) {
    uint i, max_i;
    uint* indices;
    indices = argsort(v, n);
    if (!indices) return NULL;
    for (i=0, max_i=floor(n/2); i<max_i; i++) {
        swap_uint(&(indices[i]), &(indices[n-i]));
    }
    return indices;
}

void swap(real* x, real* y) {
    real z;
    z = *x;
    *x = *y;
    *y = z;
}

void swap_uint(uint* x, uint* y) {
    uint z;
    z = *x;
    *x = *y;
    *y = z;
}

/* inspired by https://www.geeksforgeeks.org/reorder-a-array-according-to-given-indexes/ */
status_t reorder_real_vector_by_indices(real* v, uint* indices, uint n) {
    /*
    for (int i = 0; i < n; i++) {
        // While index[i] and arr[i] are not fixed
        while (index_arr[i] != i) {
            swap(arr[i], arr[index_arr[i]]);
            swap(index_arr[i], index_arr[index_arr[i]]);
        }
    }
    */
    uint i;
    uint* indices_copy;
    indices_copy = malloc(sizeof(uint)*n);
    if (!indices_copy) return ERROR_MALLOC;
    for (i=0; i<n; i++) indices_copy[i] = indices[i];
    for (i=0; i<n; i++) {
        while (indices[i] != i) {
            swap(&(v[i]), &(v[indices[i]]));
            swap_uint(&(indices[i]), &(indices[indices[i]]));
        }
    }
    free(indices_copy);
    return SUCCESS;
}

char lowercase(const char c) {
    if (('A' <= c) && (c <= 'Z')) return c-'A'+'a';
    else return c;
}

bool streq_insensitive(const char* s1, const char* s2) {
    uint i, n;
    n = strlen(s1);
    if (n != strlen(s2)) return false;
    for (i=0; i<n; i++) {
        if (lowercase(s1[i]) != lowercase(s2[i])) {
            return false;
        }
    }
    return true;
}