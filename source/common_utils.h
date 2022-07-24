#ifndef H_COMMON_UTILS
#define H_COMMON_UTILS

#include "common_types.h"

real real_add(real lhs, real rhs);
real real_sub(real lhs, real rhs);
real real_mul(real lhs, real rhs);
real real_div(real lhs, real rhs);
real real_pow(real lhs, real rhs);
real real_pow_rev(real lhs, real rhs);

int isanum(char* s);

#endif