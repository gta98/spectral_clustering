#ifndef H_SPKMEANS_TYPES
#define H_SPKMEANS_TYPES

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define MAX_DATA_POINTS         1000
#define MAX_DATA_DIMENSIONS     10

typedef float           real;
typedef uint32_t        uint;
typedef 

typedef struct {
    real at[MAX_DATA_DIMENSIONS];
    uint dims;
} point_t;

real real_add(real lhs, real rhs);
real real_sub(real lhs, real rhs);
real real_mul(real lhs, real rhs);
real real_div(real lhs, real rhs);
real real_pow(real lhs, real rhs);

#endif