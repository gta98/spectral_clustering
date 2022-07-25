#ifndef H_COMMON_UTILS
#define H_COMMON_UTILS

#include "common_types.h"

real real_add(real lhs, real rhs);
real real_sub(real lhs, real rhs);
real real_mul(real lhs, real rhs);
real real_div(real lhs, real rhs);
real real_pow(real lhs, real rhs);
real real_pow_rev(real lhs, real rhs);
real real_abs(real x);
real real_sign(real x);

int isanum(char* s);
int sign(real x);

void swap(real* x, real* y);

uint* argsort(const real* v, const uint n);
uint* argsort_desc(const real* v, const uint n);

#endif