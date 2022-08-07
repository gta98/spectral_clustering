#ifndef H_SPKMEANS_TYPES
#define H_SPKMEANS_TYPES


#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>

typedef float           real;
typedef uint32_t        uint;

typedef struct Point {
    struct Point* next;
    double* coord;
    int cluster;
} point_t;

typedef enum Status {
    SUCCESS,
    ERROR,
    ERROR_MALLOC,
    ERROR_FOPEN,
    ERROR_FORMAT,
    INVALID
} status_t;

#define MAX_DATA_POINTS         1000
#define MAX_DATA_DIMENSIONS     10

#define MAX_ITER_UNSPEC 200

#define MSG_ERR_INVALID_INPUT "Invalid Input!\n"
#define MSG_ERR_GENERIC       "An Error Has Occurred\n"

#define STATUS_SUCCESS      0
#define STATUS_ERROR        1
#define STATUS_ERROR_MALLOC STATUS_ERROR
#define STATUS_ERROR_FOPEN  STATUS_ERROR
#define STATUS_ERROR_FORMAT 2

#define RESULT_FOPEN_SUCCESS 0
#define RESULT_FOPEN_ERROR   1

#define EPSILON ((double)0.001)

#define JACOBI_MAX_ROTATIONS ((uint)100)
#define JACOBI_EPSILON       ((real)1e-5)

/* FIXME - what is the actual value of DBL_MIN? */
#ifndef INFINITY
#define INFINITY __DBL_MAX__
#endif

#ifndef INF
#define INF INFINITY
#endif

#ifndef NEGATIVE_INFINITY
#define NEGATIVE_INFINITY __DBL_MIN__
#endif

#ifndef NEG_INF
#define NEG_INF NEGATIVE_INFINITY
#endif


#endif