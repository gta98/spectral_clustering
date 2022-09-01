/* FIXME - fill with Python wrapper */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "generics/common_types.h"
#include "generics/matrix.h"
#include "generics/matrix_reader.h"
#include "spkmeans.h"

static mat_t* PyListListFloat_to_Mat(PyObject* py_mat);
static PyObject* Mat_to_PyListListFloat(mat_t* mat);
static PyObject* MatDiag_to_PyListFloat(mat_t* mat);

static mat_t* parse_mat_from_args(PyObject* args) {
    PyObject* py_data;
    char* path;
    mat_t* data;
    status_t result;

    if (!PyArg_ParseTuple(args, "O", &py_data)) return NULL;
    if (PyErr_Occurred()) return NULL;

    if (PyUnicode_Check(py_data)) {
        path = (char*) PyUnicode_AsUTF8(py_data);
        if (!path || PyErr_Occurred()) return NULL;
        result = read_data(&data, path);
        /*Py_DECREF(py_data);*/
        if (result != SUCCESS) return NULL;
        return data;
    } else if (PyList_Check(py_data)) {
        data = PyListListFloat_to_Mat(py_data);
        /*Py_DECREF(py_data);*/
        return data;
    } else {
        return NULL;
    }
}

static mat_t* parse_mat_and_mat_from_args(PyObject* args, mat_t** data_1, mat_t** data_2) {
    PyObject* py_data_1;
    PyObject* py_data_2;

    if (!PyArg_ParseTuple(args, "OO", &py_data_1, &py_data_2)) return NULL;
    if (PyErr_Occurred()) return NULL;

    *data_1 = PyListListFloat_to_Mat(py_data_1);
    *data_2 = PyListListFloat_to_Mat(py_data_2);

    return NULL;
}

static status_t parse_mat_and_k_from_args(PyObject* args, mat_t** data, uint* k) {
    PyObject* py_data;
    char* path;
    status_t result;

    if (!PyArg_ParseTuple(args, "Oi", &py_data, k)) return ERROR;
    if (PyErr_Occurred()) return ERROR;

    if (PyUnicode_Check(py_data)) {
        path = (char*) PyUnicode_AsUTF8(py_data);
        if (!path || PyErr_Occurred()) return ERROR;
        result = read_data(data, path);
        /*Py_DECREF(py_data);*/
        return result;
    } else if (PyList_Check(py_data)) {
        *data = PyListListFloat_to_Mat(py_data);
        /*Py_DECREF(py_data);*/
        if (!(*data)) return ERROR;
        return SUCCESS;
    } else {
        return ERROR;
    }
}

static PyObject* full_wam(PyObject* self, PyObject* args) {
    mat_t* data;
    mat_t* result;
    PyObject* py_result;

    data = parse_mat_from_args(args);
    if (data == NULL) return NULL;

    result = calc_full_wam(data);
    mat_free(&data);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* full_ddg(PyObject* self, PyObject* args) {
    mat_t* data;
    mat_t* result;
    PyObject* py_result;

    data = parse_mat_from_args(args);
    if (data == NULL) return NULL;

    result = calc_full_ddg(data);
    mat_free(&data);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* full_ddg_inv_sqrt(PyObject* self, PyObject* args) {
    mat_t* data;
    mat_t* result;
    PyObject* py_result;

    data = parse_mat_from_args(args);
    if (data == NULL) return NULL;

    result = calc_full_ddg_inv_sqrt(data);
    mat_free(&data);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* full_lnorm(PyObject* self, PyObject* args) {
    mat_t* data;
    mat_t* result;
    PyObject* py_result;

    data = parse_mat_from_args(args);
    if (data == NULL) return NULL;

    result = calc_full_lnorm(data);
    mat_free(&data);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* calc_lnorm_capi(PyObject* self, PyObject* args) {
    mat_t* W;
    mat_t* D_inv_sqrt;
    mat_t* result;
    PyObject* py_result;

    parse_mat_and_mat_from_args(args, &W, &D_inv_sqrt);

    result = calc_lnorm(W,D_inv_sqrt);
    mat_free(&W);
    mat_free(&D_inv_sqrt);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* _full_jacobi(PyObject* self, PyObject* args, bool sort) {
    PyObject* py_result_vectors;
    PyObject* py_result_values;
    PyObject* py_result_tuple;
    mat_t* data;
    mat_t* result_vectors;
    mat_t* result_values;
    status_t status;

    py_result_vectors = NULL;
    py_result_values = NULL;
    py_result_tuple = NULL;
    data = NULL;
    result_vectors = NULL;
    result_values = NULL;

    /*printd("======EEEEEE===========\n");*/

    data = parse_mat_from_args(args);
    if (data == NULL) goto set_tuple_failed_malloc;

    result_vectors = NULL, result_values = NULL;
    calc_full_jacobi(data, &result_vectors, &result_values);
    if (!result_vectors || !result_values) goto jacobi_failed_main;
    mat_free(&data);
    data = NULL;

    if (sort) {
        /*printd("about to sort: ");
        mat_print_diagonal(result_values);*/
        status = sort_cols_by_vector_desc(result_vectors, result_values);
        if (status != SUCCESS) goto jacobi_failed_sort;
    }

    py_result_vectors = Mat_to_PyListListFloat(result_vectors);
    if (!py_result_vectors) goto set_tuple_failed_malloc;
    /*printd("======Hola 1===========\n");*/
    mat_free(&result_vectors);
    result_vectors = NULL;

    py_result_values = MatDiag_to_PyListFloat(result_values); /* FIXME */
    if (!py_result_values) goto set_tuple_failed_malloc;
    /*printd("======Hola 2===========\n");*/
    mat_free(&result_values);
    result_values = NULL;

    py_result_tuple = PyList_New(2);
    if (!py_result_tuple) goto set_tuple_failed_malloc;
    /*printd("======Hola 3===========\n");*/

    PyList_SetItem(py_result_tuple, 0, py_result_values);
    /*printd("======Hola 4===========\n");*/
    PyList_SetItem(py_result_tuple, 1, py_result_vectors);
    /*printd("======Hola 5===========\n");*/

    return py_result_tuple;

    jacobi_failed_main:
    goto free_vectors_values_failure;
    jacobi_failed_sort:
    goto free_vectors_values_failure;
    set_tuple_failed_malloc:
    printd("Tuple failed malloc\n");
    py_result_tuple = PyErr_NoMemory();
    goto free_vectors_values_failure;
    free_vectors_values_failure:
    if (result_vectors) mat_free(&result_vectors);
    if (result_values) mat_free(&result_values);
    if (py_result_vectors) Py_DECREF(py_result_vectors);
    if (py_result_values) Py_DECREF(py_result_values);
    return py_result_tuple;
}

static PyObject* full_jacobi(PyObject* self, PyObject* args) {
    return _full_jacobi(self, args, false);
}

static PyObject* full_jacobi_sorted(PyObject* self, PyObject* args) {
    return _full_jacobi(self, args, true);
}

static PyObject* normalize_matrix_by_rows(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result;
    mat_t* data;

    if (!PyArg_ParseTuple(args, "O", &py_data)) {
        return NULL;
    }

    data = PyListListFloat_to_Mat(py_data);
    if (data == NULL) return NULL;

    mat_normalize_rows(data, data);
    /*mat_print(data);*/

    py_result = Mat_to_PyListListFloat(data);
    mat_free(&data);

    return py_result;
}

static mat_t* PyListListFloat_to_Mat(PyObject* py_mat) {
    mat_t* mat;
    PyObject* py_row;
    PyObject* py_cell;
    uint i, j;
    int h, w;
    real mat_cell;

    if (!py_mat) {
        printd("=========RETURNING ERROR 21==========\n");
        return NULL;
    }
    
    mat = NULL;

    h = (int) PyList_Size(py_mat); /* originally Py_ssize_t */
    if (h < 0) goto invalid_data_1; /* Not a list */
    if (h == 0) return mat_init(0,0);

    py_row = PyList_GetItem(py_mat, 0);
    w = (int) PyList_Size(py_row);
    if (w < 0) goto invalid_data_2;

    mat = mat_init((const uint)h,(const uint)w);
    if (!mat) return NULL;

    for (i=0; i<(uint)h; i++) {
        py_row = PyList_GetItem(py_mat, i);
        w = PyList_Size(py_row);
        if ((w < 0) || (((uint)w) != mat->w)) goto invalid_data_3;

        for (j=0; j<(uint)w; j++) {
            py_cell = PyList_GetItem(py_row, j);
            if (!py_cell) goto invalid_data_4;
            mat_cell = PyFloat_AsDouble(py_cell);
            if (PyErr_Occurred()) goto error_occurred;
            mat_set(mat, i, j, mat_cell);
        }

    }

    return mat;

    error_occurred:
    printd("errror occurred\n");
    invalid_data_1:
    printd("invalid_data_1\n");
    invalid_data_2:
    printd("invalid_data_2\n");
    invalid_data_3:
    printd("invalid_data_3\n");
    invalid_data_4:
    printd("invalid_data_4\n");
    if(mat) mat_free(&mat);
    return NULL;
}

static mat_t* PyListFloat_to_Mat(PyObject* py_mat, bool flat) {
    mat_t* mat;
    PyObject* py_cell;
    uint i, j;
    int w,h;
    real mat_cell;

    if (!py_mat) {
        printd("=========RETURNING ERROR 31==========\n");
        return NULL;
    }
    
    mat = NULL;

    w = (int) PyList_Size(py_mat);
    if (w < 0) goto invalid_data_22;
    h = flat? 1 :w;

    mat = mat_init((const uint)h,(const uint)w);
    if (!mat) return NULL;

    for (i=0; i<(uint)h; i++) {
        for (j=0; j<(uint)w; j++) {
            mat_set(mat, i, j, 0);
        }
    }

    for (j=0; j<(uint)w; j++) {
        py_cell = PyList_GetItem(py_mat, j);
        mat_cell = PyFloat_AsDouble(py_cell);
        if (PyErr_Occurred()) goto error_occurred_2;
        
        if (flat) mat_set(mat, 0, j, mat_cell);
        else mat_set(mat, j, j, mat_cell);
    }

    return mat;

    error_occurred_2:
    printd("errror occurred\n");
    invalid_data_22:
    printd("invalid_data_2\n");
    if(mat) mat_free(&mat);
    return NULL;
}

static PyObject* Mat_to_PyListListFloat(mat_t* mat) {
    PyObject* py_mat;
    PyObject* py_row;
    PyObject* py_cell;
    uint i, j, h, w;
    uint h_err, w_err;
    real cell;

    if (!mat) {
        printd("=========RETURNING ERROR 1==========\n");
        return PyErr_Occurred();
    }

    h = mat->h, w = mat->w;
    i=0, j=0;

    py_mat = PyList_New(h);
    if (!py_mat) {
        printd("=========RETURNING ERROR 2==========\n");
        return PyErr_NoMemory();
    }

    for (i=0; i<h; i++) {
        py_row = PyList_New(w);
        if (!py_row || PyErr_Occurred()) goto error_malloc_mat_to_listlist;

        for (j=0; j<w; j++) {
            cell = mat_get(mat, i, j);
            py_cell = PyFloat_FromDouble(cell);
            if (!py_cell || PyErr_Occurred()) {
                printd("Could not build value!!! Numero uno\n");
                goto error_malloc_mat_to_listlist;
            }
            PyList_SetItem(py_row, j, py_cell);
        }

        PyList_SetItem(py_mat, i, py_row);
    }

    return py_mat;

    error_malloc_mat_to_listlist:
    printd("=========RETURNING ERROR 3==========\n");
    h_err = i, w_err = j;
    if (py_row) {
        for (j=0; j<w_err; j++) {
            Py_DECREF(PyList_GetItem(py_row, j));
        }
        Py_DECREF(py_row);
    }
    for (i=0; i<h_err; i++) {
        py_row = PyList_GetItem(py_mat, i);
        for (j=0; j<w; j++) {
            Py_DECREF(PyList_GetItem(py_row, j));
        }
        Py_DECREF(py_row);
    }
    return PyErr_NoMemory();
}

static PyObject* MatDiag_to_PyListFloat(mat_t* mat) {
    PyObject* py_mat;
    PyObject* py_cell;
    uint i, h;
    real cell;

    if (!mat) {
        printd("=========RETURNING ERROR 11==========\n");
        return NULL;
    }

    h = mat->h;

    py_mat = PyList_New(h);
    if (!py_mat) {
        printd("=========RETURNING ERROR 12==========\n");
        return NULL;
    }

    for (i=0; i<h; i++) {
        cell = mat_get(mat, i, i);
        py_cell = PyFloat_FromDouble(cell);
        if (!py_cell || PyErr_Occurred()) {
            /* FIXME - low severity - value error, might wanna set later */
            return NULL;
        }
        PyList_SetItem(py_mat, i, py_cell);
    }

    return py_mat;
}

static PyObject* full_calc_k(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result;
    mat_t* data;
    uint result;

    if (!PyArg_ParseTuple(args, "O", &py_data)) {
        return NULL;
    }

    data = PyListFloat_to_Mat(py_data, true);
    if (data == NULL) return NULL;

    result = calc_k(data);
    /*printd("\nCALCULATED K: %d\n",result);*/
    mat_free(&data);

    py_result = PyLong_FromUnsignedLong((unsigned long) result);
    return py_result;
}

static PyObject* full_spk_1_to_5(PyObject* self, PyObject* args) {
    mat_t* data;
    PyObject* py_T;
    /*PyObject* py_result_tuple;
    PyObject* py_k;*/
    mat_t* L_norm;
    mat_t* eigenvalues;
    mat_t* eigenvectors;
    uint k;
    mat_t* U;
    status_t status;
    uint i,j;

    data = NULL;
    L_norm = NULL;
    eigenvalues = NULL;
    eigenvectors = NULL;
    k = 0;
    U = NULL;
    py_T = NULL;

    status = parse_mat_and_k_from_args(args, &data, &k); /* NEED TO FREE data */
    if (status != SUCCESS) goto spk_tuple_failed_malloc;

    L_norm = calc_full_lnorm(data); /* NEED TO FREE L_norm */
    
    calc_jacobi(L_norm, &eigenvectors, &eigenvalues); /* NEED TO FREE eigenvectors, eigenvalues */
    if (!eigenvectors || !eigenvalues) goto spk_could_not_calc_jacobi;
    status = sort_cols_by_vector_desc(eigenvectors, eigenvalues);
    if (status != SUCCESS) goto spk_had_a_problem;
    if (k==0) k = calc_k(eigenvalues);
    U = mat_init(eigenvectors->h,k);
    if (!U) goto spk_had_a_problem;
    for (i=0;i<U->h;i++) {
        for (j=0;j<U->w;j++) {
            mat_set(U,i,j,mat_get(eigenvectors,i,j));
        }
    }
    mat_normalize_rows(U, U);
    py_T = Mat_to_PyListListFloat(U);

    goto spk_free_and_return;

    spk_tuple_failed_malloc:
    /* set error here */
    goto spk_free_and_return;
    spk_could_not_calc_jacobi:
    goto spk_free_and_return;
    spk_had_a_problem:
    /* set error here */
    goto spk_free_and_return;
    spk_free_and_return:
    if (data) mat_free(&data);
    if (L_norm) mat_free(&L_norm);
    if (eigenvalues) mat_free(&eigenvalues);
    if (eigenvectors) mat_free(&eigenvectors);
    if (U) mat_free(&U);
    return py_T;
}

#ifdef FLAG_DEBUG

static PyObject* test_read_data(PyObject* self, PyObject* args) {
    PyObject* py_path;
    PyObject* py_mat;
    mat_t* mat;
    char* path;
    status_t result;
    if (!PyArg_ParseTuple(args, "O", &py_path)) {
        return NULL;
    }

    /* explicit casting to discard const safely */
    path = (char*) PyUnicode_AsUTF8(py_path);
    result = read_data(&mat, path);
    /*Py_DECREF(py_path);*/
    if (result != SUCCESS) {
        return PyErr_Occurred();
    }

    py_mat = Mat_to_PyListListFloat(mat);
    mat_free(&mat);

    return py_mat;
}

static PyObject* test_write_data(PyObject* self, PyObject* args) {
    PyObject* py_path;
    PyObject* py_mat;
    mat_t* mat;
    char* path;
    status_t result;
    if (!PyArg_ParseTuple(args, "OO", &py_mat, &py_path)) {
        return NULL;
    }
/*    printd("bloooooo\n");*/

    mat = PyListListFloat_to_Mat(py_mat);
    if (!mat) return PyErr_NoMemory();

    /* explicit casting to discard const safely */
    path = (char*) PyUnicode_AsUTF8(py_path);

    result = write_data(mat, path);
    /*Py_DECREF(py_path);*/

    if (result != SUCCESS) {
        return PyErr_Occurred();
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* wrap_mat_cellwise(void (*operation)(mat_t*, mat_t*, mat_t*), PyObject* mat_tuple) {
    PyObject* py_mat_1;
    PyObject* py_mat_2;
    mat_t* mat_1;
    mat_t* mat_2;
    mat_t* dst;
    PyObject* py_dst;

    py_mat_1 = NULL;
    py_mat_2 = NULL;
    mat_1 = NULL;
    mat_2 = NULL;
    dst = NULL;
    py_dst = NULL;

    if (!PyArg_ParseTuple(mat_tuple, "OO", &py_mat_1, &py_mat_2)) {
        PyErr_SetString(PyExc_ValueError, "Could not parse arguments - expected 2 matrices");
        return NULL;
    }

    mat_1 = PyListListFloat_to_Mat(py_mat_1);
    if (!mat_1) goto wrap_mat_operation_no_memory;
    py_mat_1 = NULL;

    mat_2 = PyListListFloat_to_Mat(py_mat_2);
    if (!mat_2) goto wrap_mat_operation_no_memory;
    py_mat_2 = NULL;

    if ((mat_1->h != mat_2->h) || (mat_1->w != mat_2->w)) {
        printd("(%d != %d) || (%d != %d)\n", mat_1->h, mat_2->h, mat_1->w, mat_2->w);
        goto wrap_mat_operation_diff_dims;
    }

    dst = mat_init(mat_1->h, mat_1->w);
    if (!dst) goto wrap_mat_operation_no_memory;
    
    operation(dst, mat_1, mat_2);

    py_dst = Mat_to_PyListListFloat(dst);
    if (!py_dst) goto wrap_mat_operation_no_memory;

    goto wrap_mat_operation_finish;

    wrap_mat_operation_no_memory:
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory");
    goto wrap_mat_operation_finish;

    wrap_mat_operation_diff_dims:
    PyErr_SetString(PyExc_ValueError, "Matrices should have identical dimensions! (mat_1->h != mat_2->h) || (mat_1->w != mat_2->w)");
    goto wrap_mat_operation_finish;

    wrap_mat_operation_finish:
    if (mat_1) mat_free(&mat_1);
    if (mat_2) mat_free(&mat_2);
    if (dst) mat_free(&dst);
    return py_dst;
}

static PyObject* wrap_mat_cellwise_add(PyObject* self, PyObject* args) {
    return wrap_mat_cellwise(&mat_add_cellwise, args);
}

static PyObject* wrap_mat_cellwise_sub(PyObject* self, PyObject* args) {
    return wrap_mat_cellwise(&mat_sub_cellwise, args);
}

static PyObject* wrap_mat_cellwise_mul(PyObject* self, PyObject* args) {
    return wrap_mat_cellwise(&mat_mul_cellwise, args);
}

static PyObject* wrap_mat_cellwise_div(PyObject* self, PyObject* args) {
    return wrap_mat_cellwise(&mat_mul_cellwise, args);
}

static PyObject* wrap_matmul(PyObject* self, PyObject* mat_tuple) {
    PyObject* py_mat_1;
    PyObject* py_mat_2;
    mat_t* mat_1;
    mat_t* mat_2;
    mat_t* dst;
    PyObject* py_dst;

    py_mat_1 = NULL;
    py_mat_2 = NULL;
    mat_1 = NULL;
    mat_2 = NULL;
    dst = NULL;
    py_dst = NULL;

    if (!PyArg_ParseTuple(mat_tuple, "OO", &py_mat_1, &py_mat_2)) {
        PyErr_SetString(PyExc_ValueError, "Could not parse arguments - expected 2 matrices");
        return NULL;
    }

    mat_1 = PyListListFloat_to_Mat(py_mat_1);
    if (!mat_1) goto wrap_matmul_no_memory;
    py_mat_1 = NULL;

    mat_2 = PyListListFloat_to_Mat(py_mat_2);
    if (!mat_2) goto wrap_matmul_no_memory;
    py_mat_2 = NULL;

    if (mat_1->w != mat_2->h) goto wrap_matmul_diff_dims;

    mat_transpose(mat_1); mat_transpose(mat_2);
    dst = matmul(mat_2, mat_1);
    if (!dst) goto wrap_matmul_no_memory;
    mat_transpose(dst);

    py_dst = Mat_to_PyListListFloat(dst);
    if (!py_dst) goto wrap_matmul_no_memory;

    goto wrap_matmul_finish;

    wrap_matmul_no_memory:
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory");
    goto wrap_matmul_finish;

    wrap_matmul_diff_dims:
    PyErr_SetString(PyExc_ValueError, "Matrices dimensions should match!");
    goto wrap_matmul_finish;

    wrap_matmul_finish:
    if (mat_1) mat_free(&mat_1);
    if (mat_2) mat_free(&mat_2);
    if (dst) mat_free(&dst);
    return py_dst;
}

static PyObject* test_PTAP(PyObject* self, PyObject* mat_tuple) {
    PyObject* py_A;
    PyObject* py_P;
    mat_t* A;
    mat_t* P;
    mat_t* dst;
    mat_t* tmp;
    PyObject* py_dst;

    py_A = NULL;
    py_P = NULL;
    A = NULL;
    P = NULL;
    dst = NULL;
    py_dst = NULL;
    tmp = NULL;

    if (!PyArg_ParseTuple(mat_tuple, "OO", &py_A, &py_P)) {
        PyErr_SetString(PyExc_ValueError, "Could not parse arguments - expected 2 matrices - A, P");
        return NULL;
    }

    A = PyListListFloat_to_Mat(py_A);
    if (!A) goto wrap_PTAP_test_no_memory;
    py_A = NULL;

    P = PyListListFloat_to_Mat(py_P);
    if (!P) goto wrap_PTAP_test_no_memory;
    py_P = NULL;

    if (A->w != P->h) goto wrap_PTAP_test_diff_dims;

    dst = mat_init(A->h, A->w);
    if (!dst) goto wrap_PTAP_test_no_memory;

    tmp = mat_init(A->h, A->w);
    if (!tmp) goto wrap_PTAP_test_no_memory;

    /*transform_A_tag(dst, A, P, tmp);*/

    py_dst = Mat_to_PyListListFloat(dst);
    if (!py_dst) goto wrap_PTAP_test_no_memory;

    goto wrap_PTAP_test_finish;

    wrap_PTAP_test_no_memory:
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory");
    goto wrap_PTAP_test_finish;

    wrap_PTAP_test_diff_dims:
    PyErr_SetString(PyExc_ValueError, "Matrices dimensions should match!");
    goto wrap_PTAP_test_finish;

    wrap_PTAP_test_finish:
    if (A) mat_free(&A);
    if (P) mat_free(&P);
    if (tmp) mat_free(&tmp);
    if (dst) mat_free(&dst);
    return py_dst;
}

static PyObject* wrap_reorder_mat_cols_by_indices(PyObject* self, PyObject* args) {
    PyObject* py_mat;
    PyObject* py_indices;
    mat_t* mat;
    mat_t* mat_indices;
    uint* indices;
    PyObject* py_dst;
    uint j;
    status_t result;

    py_mat = NULL;
    py_indices = NULL;
    mat = NULL;
    mat_indices = NULL;
    indices = NULL;
    py_dst = NULL;
    result = INVALID;

    if (!PyArg_ParseTuple(args, "OO", &py_mat, &py_indices)) {
        PyErr_SetString(PyExc_ValueError, "Could not parse arguments - expected 2 matrices");
        return NULL;
    }

    mat = PyListListFloat_to_Mat(py_mat);
    if (!mat) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert py_mat into mat");
        goto wrap_mat_reorder_finish;
    }
    py_mat = NULL;

    mat_indices = PyListFloat_to_Mat(py_indices, true);
    if (!mat_indices) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert py_indices into mat_indices");
        goto wrap_mat_reorder_finish;
    }
    py_indices = NULL;

    if (mat->w != mat_indices->w) goto wrap_mat_reorder_diff_dims;
    if (mat_indices->h != 1) goto wrap_mat_reorder_diff_dims;

    indices = malloc(sizeof(uint)*mat_indices->w);
    if (!indices) goto wrap_mat_reorder_no_memory;
    for (j=0; j<mat_indices->w; j++) indices[j] = mat_get(mat_indices,0,j);

    result = reorder_mat_cols_by_indices(mat, indices);
    if (result != SUCCESS) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert reorder mat cols by indices");
        goto wrap_mat_reorder_finish;
    };

    py_dst = Mat_to_PyListListFloat(mat);
    if (!py_dst) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert mat into py_dst");
        goto wrap_mat_reorder_finish;
    }

    goto wrap_mat_reorder_finish;

    wrap_mat_reorder_no_memory:
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory");
    goto wrap_mat_reorder_finish;

    wrap_mat_reorder_diff_dims:
    PyErr_SetString(PyExc_ValueError, "Matrices dimensions should match!");
    goto wrap_mat_reorder_finish;

    wrap_mat_reorder_finish:
    if (mat) mat_free(&mat);
    if (mat_indices) mat_free(&mat_indices);
    if (indices) free(indices);
    return py_dst;
}

static PyObject* wrap_sort_cols_by_vector_desc(PyObject* self, PyObject* args) {
    PyObject* py_mat;
    PyObject* py_eigenvalues;
    mat_t* mat;
    mat_t* mat_eigenvalues;
    mat_t* mat_eigenvalues_flat;
    PyObject* py_dst;
    uint j;
    status_t result;

    py_mat = NULL;
    py_eigenvalues = NULL;
    mat = NULL;
    mat_eigenvalues = NULL;
    mat_eigenvalues_flat = NULL;
    py_dst = NULL;
    result = INVALID;

    if (!PyArg_ParseTuple(args, "OO", &py_mat, &py_eigenvalues)) {
        PyErr_SetString(PyExc_ValueError, "Could not parse arguments - expected 2 matrices");
        return NULL;
    }

    mat = PyListListFloat_to_Mat(py_mat);
    if (!mat) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert py_mat into mat");
        goto wrap_mat_sort_cols_desc_finish;
    }
    py_mat = NULL;

    mat_eigenvalues_flat = PyListFloat_to_Mat(py_eigenvalues, true);
    if (!mat_eigenvalues_flat) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert py_eigenvalues into mat_eigenvalues_flat");
        goto wrap_mat_sort_cols_desc_finish;
    }
    py_eigenvalues = NULL;

    mat_eigenvalues = mat_init_full(mat_eigenvalues_flat->w, mat_eigenvalues_flat->w, 0);
    if (!mat_eigenvalues) {
        PyErr_SetString(PyExc_MemoryError, "Could not initialize mat_eigenvalues");
        goto wrap_mat_sort_cols_desc_finish;
    }
    for (j=0; j<mat_eigenvalues_flat->w; j++) mat_set(mat_eigenvalues, j, j, mat_get(mat_eigenvalues_flat, 0, j));

    if (mat->w != mat_eigenvalues->w) {
        PyErr_SetString(PyExc_ValueError, "Matrix width does not match eigenvalues length!");
        goto wrap_mat_sort_cols_desc_finish;
    }

    result = sort_cols_by_vector_desc(mat, mat_eigenvalues);
    if (result != SUCCESS) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert reorder mat cols by eigenvalues");
        goto wrap_mat_sort_cols_desc_finish;
    };

    py_dst = Mat_to_PyListListFloat(mat);
    if (!py_dst) {
        PyErr_SetString(PyExc_MemoryError, "Could not convert mat into py_dst");
        goto wrap_mat_sort_cols_desc_finish;
    }

    wrap_mat_sort_cols_desc_finish:
    if (mat) mat_free(&mat);
    if (mat_eigenvalues) mat_free(&mat_eigenvalues);

    return py_dst;
}

static PyObject* test_calc_L_norm(PyObject* self, PyObject* args) { /* W, D */
    PyObject* py_W; PyObject* py_D;
    mat_t* W; mat_t* D; mat_t* result;
    PyObject* py_result;

    W = NULL;
    D = NULL;
    result = NULL;
    py_result = NULL;

    if (!PyArg_ParseTuple(args, "OO", &py_W, &py_D)) {
        PyErr_SetString(PyExc_ValueError, "Expected W, D as inputs");
        goto test_calc_L_norm_finish;
    }

    W = PyListListFloat_to_Mat(py_W);
    if (W == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not parse W");
        goto test_calc_L_norm_finish;
    }

    D = PyListListFloat_to_Mat(py_D);
    if (D == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not parse D");
        goto test_calc_L_norm_finish;
    }

    result = calc_lnorm(W, D);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Could not calculate Lnorm from W, D");
        goto test_calc_L_norm_finish;
    }

    py_result = Mat_to_PyListListFloat(result);

    test_calc_L_norm_finish:
    if (W) mat_free(&W);
    if (D) mat_free(&D);
    if (result) mat_free(&result);
    return py_result;
}

#endif


static PyMethodDef spkmeansmoduleMethods[] = {
#ifdef FLAG_DEBUG
    {"test_read_data", 
      (PyCFunction) test_read_data,
      METH_VARARGS, 
      PyDoc_STR("This method performs the k-means algorithm with the specified arguments")},
    {"test_write_data", 
      (PyCFunction) test_write_data,
      METH_VARARGS, 
      PyDoc_STR("This method performs the k-means algorithm with the specified arguments")},
#endif
    {"full_wam", 
      (PyCFunction) full_wam,
      METH_VARARGS, 
      PyDoc_STR("Calculates weighted adjacency matrix on given datapoints")},
    {"full_ddg", 
      (PyCFunction) full_ddg,
      METH_VARARGS, 
      PyDoc_STR("Calculates diagonal degree matrix on given datapoints")},
    {"full_ddg_inv_sqrt", 
      (PyCFunction) full_ddg_inv_sqrt,
      METH_VARARGS, 
      PyDoc_STR("Calculates diagonal degree matrix (inv sqrt) on given datapoints")},
    {"full_lnorm", 
      (PyCFunction) full_lnorm,
      METH_VARARGS, 
      PyDoc_STR("Calculates L-Norm on given datapoints")},
    {"calc_lnorm", 
      (PyCFunction) calc_lnorm_capi,
      METH_VARARGS, 
      PyDoc_STR("Calculates L-Norm on W (arg 1), D (arg 2)")},
    {"full_jacobi", 
      (PyCFunction) full_jacobi,
      METH_VARARGS, 
      PyDoc_STR("Performs Jacobi algorithm on datapoints, returns tuple containing eigenvalues, eigenvectors")},
    {"full_jacobi_sorted",
      (PyCFunction) full_jacobi_sorted,
      METH_VARARGS, 
      PyDoc_STR("Performs Jacobi algorithm on datapoints, returns tuple containing eigenvalues, eigenvectors (sorted)")},
    {"normalize_matrix_by_rows",
      (PyCFunction) normalize_matrix_by_rows,
      METH_VARARGS, 
      PyDoc_STR("As the name states, normalizes the given matrix by rows, returns new matrix")},
    {"full_calc_k",
      (PyCFunction) full_calc_k,
      METH_VARARGS, 
      PyDoc_STR("Given eigenvalues, determines k according to eigengap heuristic")},
    {"full_spk_1_to_5",
      (PyCFunction) full_spk_1_to_5,
      METH_VARARGS, 
      PyDoc_STR("Performs spk stages 1-5 on mat in path, returns T")},
#ifdef FLAG_DEBUG
    {"mat_cellwise_add",
      (PyCFunction) wrap_mat_cellwise_add,
      METH_VARARGS, 
      PyDoc_STR("Adds (cellwise) two matrices together")},
    {"mat_cellwise_sub",
      (PyCFunction) wrap_mat_cellwise_sub,
      METH_VARARGS, 
      PyDoc_STR("Subtracts (cellwise) second matrix from first")},
    {"mat_cellwise_mul",
      (PyCFunction) wrap_mat_cellwise_mul,
      METH_VARARGS, 
      PyDoc_STR("Calculates (cellwise) multiple of two matrices")},
    {"mat_cellwise_div",
      (PyCFunction) wrap_mat_cellwise_div,
      METH_VARARGS, 
      PyDoc_STR("Calculates (cellwise) division of two matrices")},
    {"matmul",
      (PyCFunction) wrap_matmul,
      METH_VARARGS, 
      PyDoc_STR("Calculates multiple of two matrices")},
    {"reorder_mat_cols_by_indices",
      (PyCFunction) wrap_reorder_mat_cols_by_indices,
      METH_VARARGS, 
      PyDoc_STR("Reorders matrix (arg 1) by cols, according to indices (arg 2)")},
    {"sort_cols_by_vector_desc",
      (PyCFunction) wrap_sort_cols_by_vector_desc,
      METH_VARARGS, 
      PyDoc_STR("Sorts matrix (arg 1) by cols, according to vector (arg 2) sorting")},
    {"test_PTAP",
      (PyCFunction) test_PTAP,
      METH_VARARGS, 
      PyDoc_STR("Returns P.transpose @ A @ P, where A == arg1, P == arg2")},
    {"test_calc_L_norm",
      (PyCFunction) test_calc_L_norm,
      METH_VARARGS, 
      PyDoc_STR("Calculates Lnorm from W (arg1), D (arg2)")},
#endif
    {NULL, NULL, 0, NULL}
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef spkmeans_moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    spkmeansmoduleMethods, /* the PyMethodDef array from before containing the methods of the extension */
    NULL, /* m_slots */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL /* m_free */
};

PyMODINIT_FUNC PyInit_spkmeansmodule(void)
{
    PyObject *m;
    m = PyModule_Create(&spkmeans_moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
