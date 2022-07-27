#ifndef H_SPKMEANS
#define H_SPKMEANS

typedef enum Goal {
    SPK,
    WAM,
    DDG,
    LNORM,
    JACOBI,
    INVALID_GOAL
} goal_t;
const char* GOALS[] = {"spk", "wam", "ddg", "lnorm", "jacobi"};
const int GOALS_COUNT = 5;

#include "generics/common.h"
#include "generics/matrix.h"
#include "generics/matrix_reader.h"
#include "algorithms/ddg.h"
#include "algorithms/eigengap.h"
#include "algorithms/jacobi.h"
#include "algorithms/lnorm.h"
#include "algorithms/wam.h"

mat_t* calc_wam(const mat_t* data);

#endif