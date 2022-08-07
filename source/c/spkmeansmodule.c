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


static PyObject* full_wam(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result;
    mat_t* data;
    mat_t* result;

    if (!PyArg_ParseTuple(args, "O", &py_data)) {
        return NULL;
    }

    data = PyListListFloat_to_Mat(py_data);
    if (data == NULL) return NULL;

    result = calc_full_wam(data);
    mat_free(&data);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* full_ddg(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result;
    mat_t* data;
    mat_t* result;

    if (!PyArg_ParseTuple(args, "O", &py_data)) {
        return NULL;
    }

    data = PyListListFloat_to_Mat(py_data);
    if (data == NULL) return NULL;

    result = calc_full_ddg(data);
    mat_free(&data);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* full_lnorm(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result;
    mat_t* data;
    mat_t* result;

    if (!PyArg_ParseTuple(args, "O", &py_data)) {
        return NULL;
    }

    data = PyListListFloat_to_Mat(py_data);
    if (data == NULL) return NULL;

    result = calc_full_lnorm(data);
    mat_free(&data);

    py_result = Mat_to_PyListListFloat(result);
    mat_free(&result);
    return py_result;
}

static PyObject* full_jacobi(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result_vectors;
    PyObject* py_result_values;
    PyObject* py_result_tuple;
    mat_t* data;
    mat_t* result_vectors;
    mat_t* result_values;

    py_data = NULL;
    py_result_vectors = NULL;
    py_result_values = NULL;
    py_result_tuple = NULL;
    data = NULL;
    result_vectors = NULL;
    result_values = NULL;

    printd("======EEEEEE===========\n");

    if (!PyArg_ParseTuple(args, "O", &py_data)) {
        goto set_tuple_failed_parse;
    }

    data = PyListListFloat_to_Mat(py_data);
    if (data == NULL) goto set_tuple_failed_malloc;

    result_vectors = NULL, result_values = NULL;
    calc_full_jacobi(data, &result_vectors, &result_values);
    printd("======Hola===========\n");
    mat_print(result_values);
    mat_free(&data);
    data = NULL;

    py_result_vectors = Mat_to_PyListListFloat(result_vectors);
    if (!py_result_vectors) goto set_tuple_failed_malloc;
    printd("======Hola 1===========\n");
    mat_free(&result_vectors);
    result_vectors = NULL;

    py_result_values = MatDiag_to_PyListFloat(result_values); /* FIXME */
    if (!py_result_values) goto set_tuple_failed_malloc;
    printd("======Hola 2===========\n");
    mat_free(&result_values);
    result_values = NULL;

    py_result_tuple = PyList_New(2);
    if (!py_result_tuple) goto set_tuple_failed_malloc;
    printd("======Hola 3===========\n");

    PyList_SetItem(py_result_tuple, 0, py_result_values);
    printd("======Hola 4===========\n");
    PyList_SetItem(py_result_tuple, 1, py_result_vectors);
    printd("======Hola 5===========\n");

    return py_result_tuple;

    set_tuple_failed_parse:
    py_result_tuple = PyErr_Occurred();
    goto free_vectors_values_failure;
    set_tuple_failed_malloc:
    py_result_tuple = PyErr_NoMemory();
    goto free_vectors_values_failure;
    free_vectors_values_failure:
    if (result_vectors) mat_free(&result_vectors);
    if (result_values) mat_free(&result_values);
    if (py_result_vectors) Py_DECREF(py_result_vectors);
    if (py_result_values) Py_DECREF(py_result_values);
    return py_result_tuple;
}

static PyObject* full_jacobi_sorted(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result_vectors;
    PyObject* py_result_values;
    PyObject* py_result_tuple;
    mat_t* data;
    mat_t* result_vectors;
    mat_t* result_values;
    status_t status;

    py_data = NULL;
    py_result_vectors = NULL;
    py_result_values = NULL;
    py_result_tuple = NULL;
    data = NULL;
    result_vectors = NULL;
    result_values = NULL;

    if (!PyArg_ParseTuple(args, "O", &data)) {
        return NULL;
    }

    data = PyListListFloat_to_Mat(py_data);
    if (data == NULL) return NULL;

    result_vectors = NULL, result_values = NULL;
    calc_full_jacobi(data, &result_vectors, &result_values);
    mat_free(&data);
    data = NULL;

    status = sort_cols_by_vector_desc(result_vectors, result_values);
    if (status != SUCCESS) goto free_vectors_values_failure_2;

    py_result_vectors = Mat_to_PyListListFloat(result_vectors);
    if (!py_result_vectors) goto free_vectors_values_failure_2;
    mat_free(&result_vectors);
    result_vectors = NULL;

    py_result_values = MatDiag_to_PyListFloat(result_values); /* FIXME */
    if (!py_result_values) goto free_vectors_values_failure_2;
    mat_free(&result_values);
    result_values = NULL;

    py_result_tuple = PyList_New(2);
    if (!py_result_tuple) goto free_vectors_values_failure_2;

    PyList_SetItem(py_result_tuple, 0, py_result_values);
    PyList_SetItem(py_result_tuple, 1, py_result_vectors);

    return py_result_tuple;

    free_vectors_values_failure_2:
    if (result_vectors) mat_free(&result_vectors);
    if (result_values) mat_free(&result_values);
    if (py_result_vectors) Py_DECREF(py_result_vectors);
    if (py_result_values) Py_DECREF(py_result_values);
    return NULL;
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

    py_result = Mat_to_PyListListFloat(data);
    mat_free(&data);

    return py_result;
}

static PyObject* full_calc_k(PyObject* self, PyObject* args) {
    PyObject* py_data;
    PyObject* py_result;
    mat_t* data;
    uint result;

    if (!PyArg_ParseTuple(args, "O", &py_data)) {
        return NULL;
    }

    data = PyListListFloat_to_Mat(py_data);
    if (data == NULL) return NULL;

    result = calc_k(data);
    mat_free(&data);

    py_result = PyLong_FromUnsignedLong((unsigned long) result);
    return py_result;
}

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
    Py_DECREF(py_path);
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
    printd("converted mat\n");
    /* explicit casting to discard const safely */
    path = (char*) PyUnicode_AsUTF8(py_path);

    printd("wrooooot\n");
    printd("pooth is %s\n", path);
    result = write_data(mat, path);
    Py_DECREF(py_path);

    if (result != SUCCESS) {
        return PyErr_Occurred();
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static mat_t* PyListListFloat_to_Mat(PyObject* py_mat) {
    mat_t* mat;
    PyObject* py_row;
    PyObject* py_cell;
    uint i, j;
    int h, w;
    double mat_cell;

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

    py_mat = PyList_New(h);
    if (!py_mat) {
        printd("=========RETURNING ERROR 2==========\n");
        return PyErr_NoMemory();
    }

    for (i=0; i<h; i++) {
        py_row = PyList_New(w);
        if (!py_row) goto error_malloc_mat_to_listlist;

        for (j=0; j<w; j++) {
            cell = mat_get(mat, i, j);
            py_cell = Py_BuildValue("d", cell);
            if (!py_cell || PyErr_Occurred()) {
                printd("Could not build value!!! Numero uno\n");
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
        return PyErr_Occurred();
    }

    h = mat->h;

    py_mat = PyList_New(h);
    if (!py_mat) {
        printd("=========RETURNING ERROR 12==========\n");
        return PyErr_NoMemory();
    }

    for (i=0; i<h; i++) {
        cell = mat_get(mat, i, i);
        py_cell = Py_BuildValue("d", cell);
        if (!py_cell || PyErr_Occurred()) {
            printd("Could not build value for %f!!! Numero dos\n", cell);
        } else {
            printd("Numero dos passed for %f\n", cell);
        }
        PyList_SetItem(py_mat, i, py_cell);
    }

    return py_mat;
}

#ifdef FLAG_DEBUG
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

    if ((mat_1->h != mat_2->h) || (mat_1->w != mat_2->w)) goto wrap_mat_operation_diff_dims;

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
    PyErr_SetString(PyExc_ValueError, "Matrices should have identical dimensions!");
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
    Py_DECREF(py_mat_1);
    py_mat_1 = NULL;

    mat_2 = PyListListFloat_to_Mat(py_mat_2);
    if (!mat_2) goto wrap_matmul_no_memory;
    Py_DECREF(py_mat_2);
    py_mat_2 = NULL;

    if (mat_1->w != mat_2->h) goto wrap_matmul_diff_dims;

    dst = mat_init(mat_1->h, mat_1->w);
    if (!dst) goto wrap_matmul_no_memory;
    
    dst = matmul(mat_1, mat_2);
    if (!dst) goto wrap_matmul_no_memory;
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
    if (py_mat_1) Py_DECREF(py_mat_1);
    if (py_mat_2) Py_DECREF(py_mat_2);
    return py_dst;
}

#endif


static PyMethodDef spkmeansmoduleMethods[] = {
    {"test_read_data", 
      (PyCFunction) test_read_data,
      METH_VARARGS, 
      PyDoc_STR("This method performs the k-means algorithm with the specified arguments")},
    {"test_write_data", 
      (PyCFunction) test_write_data,
      METH_VARARGS, 
      PyDoc_STR("This method performs the k-means algorithm with the specified arguments")},
    {"full_wam", 
      (PyCFunction) full_wam,
      METH_VARARGS, 
      PyDoc_STR("Calculates weighted adjacency matrix on given datapoints")},
    {"full_ddg", 
      (PyCFunction) full_ddg,
      METH_VARARGS, 
      PyDoc_STR("Calculates diagonal degree matrix on given datapoints")},
    {"full_lnorm", 
      (PyCFunction) full_lnorm,
      METH_VARARGS, 
      PyDoc_STR("Calculates L-Norm on given datapoints")},
    {"full_jacobi", 
      (PyCFunction) full_jacobi,
      METH_VARARGS, 
      PyDoc_STR("Performs Jacobi algorithm on datapoints, returns tuple containing eigenvalues, eigenvectors")},
    {"full_jacobi_sorted", /* FIXME - PyDocs */
      (PyCFunction) full_jacobi_sorted,
      METH_VARARGS, 
      PyDoc_STR("Performs Jacobi algorithm on datapoints, returns tuple containing eigenvalues, eigenvectors")},
    {"normalize_matrix_by_rows", /* FIXME - PyDocs */
      (PyCFunction) normalize_matrix_by_rows,
      METH_VARARGS, 
      PyDoc_STR("Performs Jacobi algorithm on datapoints, returns tuple containing eigenvalues, eigenvectors")},
    {"full_calc_k", /* FIXME - PyDocs */
      (PyCFunction) full_calc_k,
      METH_VARARGS, 
      PyDoc_STR("Performs Jacobi algorithm on datapoints, returns tuple containing eigenvalues, eigenvectors")},
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
    #endif
    {NULL, NULL, 0, NULL}
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef spkmeans_moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    spkmeansmoduleMethods /* the PyMethodDef array from before containing the methods of the extension */
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
