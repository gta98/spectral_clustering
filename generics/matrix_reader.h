#ifndef H_MATRIX_READER
#define H_MATRIX_READER

#include "generics/common.h"
#include "generics/matrix.h"

/* allocates dst and reads from path_to_input, returns SUCCESS or a failure code */
status_t read_data(mat_t** dst, char* path_to_input);

/* writes src into path_to_output, returns SUCCESS or a failure code */
status_t write_data(mat_t* src, char* path_to_output);

#endif