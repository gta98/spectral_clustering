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
/*#define GOALS_COUNT 5
extern char GOALS[GOALS_COUNT][7];*/


mat_t* calc_full_wam(mat_t* data);
mat_t* calc_full_ddg(mat_t* data);
mat_t* calc_full_lnorm(mat_t* data);
void   calc_full_jacobi(mat_t* generic_symm_mat, mat_t** eigenvectors, mat_t** eigenvalues);

status_t print_wam(mat_t* data);
status_t print_ddg(mat_t* data);
status_t print_lnorm(mat_t* data);
status_t print_jacobi(mat_t* data);

goal_t get_selected_routine(const char* goal_str);
int get_code_print_msg(int code, const char* msg);

#endif