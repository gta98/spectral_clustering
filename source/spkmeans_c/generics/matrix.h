#ifndef H_MATRIX
#define H_MATRIX

#include "common_utils.h"
#include "common_types.h"
#include "common_includes.h"

typedef struct {
    real*       __data;
    bool        __swap_axes;
    uint        h;
    uint        w;
} mat_t;

/* initialize mat(h,w) with junk and return */
mat_t* mat_init(const uint h, const uint w);

/* initialize mat(h,w) and fill with value */
mat_t* mat_init_full(const uint h, const uint w, const real value);

/* return a copy of mat */
mat_t* mat_init_like(mat_t* mat);

/* initialize mat(n,n) with zeros, fill diagonal with 1 */
mat_t* mat_init_identity(const uint n);

/* free mat memory */
void mat_free(mat_t** mat);

/* transpose mat */
void mat_transpose(mat_t* mat);

/* return mat[i][j] if __transposed==0, else mat[j][i] */
real mat_get(mat_t* mat, uint i, uint j);

/* mat[i][j] = new_value if __transposed==0, else mat[j][i] */
void mat_set(mat_t* mat, uint i, uint j, const real new_value);

/* return a copy of mat */
void mat_copy_to(mat_t* dst, mat_t* src);

/* dst = mat_1 + mat_2 */
void mat_add_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2);

/* dst = mat_1 - mat_2 */
void mat_sub_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2);

/* dst = mat_1 * mat_2 */
void mat_mul_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2);

/* dst = mat_1 / mat_2 */
void mat_div_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2);

/* dst = mat_1 ^ mat_2 */
void mat_pow_cellwise(mat_t* dst, const mat_t* mat_1, const mat_t* mat_2);

/* dst = mat + alpha */
void mat_add_scalar(mat_t* dst, const mat_t* mat, const real alpha);

/* dst = mat - alpha */
void mat_sub_scalar(mat_t* dst, const mat_t* mat, const real alpha);

/* dst = mat * alpha */
void mat_mul_scalar(mat_t* dst, const mat_t* mat, const real alpha);

/* dst = mat / alpha */
void mat_div_scalar(mat_t* dst, const mat_t* mat, const real alpha);

/* dst = mat ^ alpha */
void mat_pow_scalar(mat_t* dst, const mat_t* mat, const real alpha);

/* dst = mat_lhs @ mat_rhs */
void mat_mul(mat_t* dst, const mat_t* mat_lhs, const mat_t* mat_rhs);

/* returns mat_lhs @ mat_rhs */
mat_t* matmul(const mat_t* mat_lhs, const mat_t* mat_rhs);

/* prints mat to stdout */
void mat_print(const mat_t* mat);

status_t reorder_mat_cols_by_indices(mat_t* v, uint* indices);


#endif