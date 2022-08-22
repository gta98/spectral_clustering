#ifndef H_EIGENGAP
#define H_EIGENGAP

#include "generics/common_includes.h"
#include "generics/common_utils.h"
#include "generics/matrix.h"

/* this sorts A and v by v, which can be either flat or square */
status_t sort_cols_by_vector_desc(mat_t* A, mat_t* v);

/* calculates k from eigenvalues according to eigengap heuristic */
uint calc_k(mat_t* eigenvalues);

#endif