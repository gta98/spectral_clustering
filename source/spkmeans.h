#ifndef H_SPKMEANS
#define H_SPKMEANS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define MAX_DATA_POINTS         1000
#define MAX_DATA_DIMENSIONS     10

typedef float           real;
typedef uint32_t        uint;

typedef struct {
    real at[MAX_DATA_DIMENSIONS];
    uint dims;
} point_t;

#endif