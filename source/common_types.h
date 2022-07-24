#ifndef H_SPKMEANS_TYPES
#define H_SPKMEANS_TYPES


#include <stdint.h>
#include <stdbool.h>

typedef float           real;
typedef uint32_t        uint;

typedef struct Point {
    double* coord;
    int cluster;
} point_t;

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


#endif