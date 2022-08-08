#ifndef H_JACOBI
#define H_JACOBI

#include "generics/common_includes.h"
#include "generics/matrix.h"

void calc_jacobi(mat_t* A, mat_t** eigenvectors, mat_t** eigenvalues);

#ifdef FLAG_DEBUG
void transform_A_tag(mat_t* A_tag, mat_t* A, mat_t* P, mat_t* tmp);
#endif

#endif