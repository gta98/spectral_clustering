#ifndef H_SPKMEANS
#define H_SPKMEANS

#include "generics/common.h"
#include "generics/matrix.h"
#include "generics/matrix_reader.h"
#include "algorithms/ddg.h"
#include "algorithms/eigengap.h"
#include "algorithms/jacobi.h"
#include "algorithms/lnorm.h"
#include "algorithms/wam.h"

typedef enum Goal {
    SPK,
    WAM,
    DDG,
    LNORM,
    JACOBI,
    INVALID_GOAL
} goal_t;
const char* GOALS[] = {"spk", "wam", "ddg", "lnorm", "jacobi"};
const uint GOALS_COUNT = 5;

#endif