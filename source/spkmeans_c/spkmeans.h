#ifndef H_SPKMEANS
#define H_SPKMEANS

#include "common.h"
#include "algorithms/ddg.h"
#include "algorithms/eigengap.h"
#include "algorithms/jacobi.h"
#include "algorithms/lnorm.h"
#include "algorithms/wam.h"

mat_t* calc_wam(const mat_t* data);

#endif