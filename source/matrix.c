
#include "linalg.h"


/* initialize mat(h,w) with zeros and return */
mat_t* mat_init(const uint h, const uint w) {
    mat_t* mat;
    int i, mat_size;
    mat_size = h*w;
    mat = malloc(sizeof(real)*mat_size);
    if (!mat) return NULL;
    for (i=0; i<mat_size; mat->__data[i]=0, i++);
    mat->__swap_axes = false;
    mat->h = h;
    mat->w = w;
}

void mat_free(mat_t** mat) {
    free((*mat)->__data);
    free(*mat);
}

/* transpose mat */
void mat_transpose(mat_t* mat) {
    mat->h ^= mat->w;
    mat->w ^= mat->h;
    mat->h ^= mat->w;
    mat->__swap_axes = !mat->__swap_axes;
}

/* return mat[i][j] if __transposed==0, else mat[j][i] */
real mat_get(mat_t* mat, uint i, uint j) {
    if (mat->__swap_axes) {
        i ^= j;
        j ^= i;
        i ^= j;
    }
    if ((i < 0) || (j < 0) || (i >= mat->h) || (j >= mat->w)) {
        perror("Attempted to access invalid matrix indices");
    }
    return mat->__data[(i*mat->w)+j];
}

/* mat[i][j] = new_value if __transposed==0, else mat[j][i] */
void mat_set(mat_t* mat, uint i, uint j, const real new_value) {
    if (mat->__swap_axes) {
        i ^= j;
        j ^= i;
        i ^= j;
    }
    assert((i >= 0) && (j >= 0) && (i < mat->h) && (j < mat->w));
    mat->__data[(i*mat->w)+j] = new_value;
}

/* return a copy of mat */
void mat_copy_to(mat_t* dst, mat_t* src) {
    int i, j, h, w;
    assert(dst && src);
    assert(dst->h == src->h);
    assert(dst->w == src->w);
    h = dst->h;
    w = dst->w;
    for (i=0; i<h; i++) {
        for (j=0; j<w; j++) {
            mat_set(dst, i, j, mat_get(src, i, j));
        }
    }
}


/* return a copy of mat */
mat_t* mat_copy(mat_t* mat) {
    mat_t* copy = mat_init(mat->h, mat->w);
    if (!copy) return NULL;
    mat_copy_to(copy, mat);
    return copy;
}

void __mat_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2, real (*op)(real, real)) {
    int i, j;
    assert((mat_1->h == mat_2->h) && (mat_2->h == dst->h));
    assert((mat_1->w == mat_2->w) && (mat_2->w == dst->w));
    for (i=0; i<dst->h; i++) {
        for (j=0; j<dst->w; j++) {
            mat_set(dst, i, j, op(mat_get(mat_1, i, j),mat_get(mat_2, i, j)));
        }
    }
}

void __mat_cellwise_scalar(mat_t* dst, const mat_t* mat, const real alpha, real (*op)(real, real)) {
    int i, j;
    assert(mat->h == dst->h);
    assert(mat->w == dst->w);
    for (i=0; i<dst->h; i++) {
        for (j=0; j<dst->w; j++) {
            mat_set(dst, i, j, op(mat_get(mat, i, j), alpha));
        }
    }
}

/* dst = mat_1 + mat_2 */
void mat_add_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2) {
    __mat_cellwise(dst, mat_1, mat_2, real_add);
}

/* dst = mat_1 - mat_2 */
void mat_sub_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2) {
    __mat_cellwise(dst, mat_1, mat_2, real_sub);
}

/* dst = mat_1 * mat_2 */
void mat_mul_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2) {
    __mat_cellwise(dst, mat_1, mat_2, real_mul);
}

/* dst = mat_1 / mat_2 */
void mat_div_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2) {
    __mat_cellwise(dst, mat_1, mat_2, real_div);
}

/* dst = mat_1 ^ mat_2 */
void mat_pow_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2) {
    __mat_cellwise(dst, mat_1, mat_2, real_pow);
}

/* dst = mat + alpha */
void mat_add_scalar(mat_t* dst, const mat_t* mat, const real alpha) {
    __mat_cellwise_scalar(dst, mat, alpha, real_add);
}

/* dst = mat - alpha */
void mat_sub_scalar(mat_t* dst, const mat_t* mat, const real alpha) {
    __mat_cellwise_scalar(dst, mat, alpha, real_sub);
}

/* dst = mat * alpha */
void mat_mul_scalar(mat_t* dst, const mat_t* mat, const real alpha) {
    __mat_cellwise_scalar(dst, mat, alpha, real_mul);
}

/* dst = mat / alpha */
void mat_div_scalar(mat_t* dst, const mat_t* mat, const real alpha) {
    __mat_cellwise_scalar(dst, mat, alpha, real_div);
}

/* dst = mat ^ alpha */
void mat_pow_scalar(mat_t* dst, const mat_t* mat, const real alpha) {
    __mat_cellwise_scalar(dst, mat, alpha, real_pow);
}

/* dst = dst @ src */
void mat_mul(mat_t* dst, const mat_t* mat_lhs, const mat_t* mat_rhs) {
    uint i, j, k;
    real value;
    assert(mat_lhs->w == mat_rhs->h);
    assert(dst->w == mat_rhs->w);
    assert(dst->h == mat_lhs->h);
    for (i=0; i<dst->h; i++) {
        for (j=0; j<dst->w; j++) {
            value = 0;
            for (k=0; k<dst->w; k++) {
                value += mat_get(mat_lhs, i, k) * mat_get(mat_rhs, k, j);
            }
            mat_set(dst, i, j, value);
        }
    }
}

mat_t* matmul(const mat_t* mat_lhs, const mat_t* mat_rhs) {
    mat_t* dst;
    dst = mat_init(mat_lhs->h, mat_rhs->w);
    mat_mul(dst, mat_lhs, mat_rhs);
    return dst;
}