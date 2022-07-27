#ifndef H_EIGENGAP
#define H_EIGENGAP

#include "generics/common_includes.h"
#include "generics/common_utils.h"
#include "generics/matrix.h"

status_t sort_cols_by_vector_desc(mat_t* A, mat_t* v);
uint calc_k(mat_t* eigenvalues);

#endif