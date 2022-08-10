#ifndef H_DDG
#define H_DDG

#include "generics/common_includes.h"
#include "generics/matrix.h"

mat_t* calc_ddg(mat_t* W);
mat_t* calc_ddg_inv_sqrt(mat_t* W);

#endif