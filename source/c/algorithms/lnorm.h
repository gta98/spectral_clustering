#ifndef H_LNORM
#define H_LNORM

#include "generics/common_includes.h"
#include "generics/matrix.h"

mat_t* calc_lnorm(const mat_t* W, const mat_t* D_inv_sqrt);

#endif