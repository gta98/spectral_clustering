/* FIXME - fill with Python wrapper */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "generics/common_types.h"
#include "generics/matrix.h"
#include "generics/matrix_reader.h"

static mat_t* PyListListFloat_to_Mat(PyObject* py_mat);
static PyObject* Mat_to_PyListListFloat(mat_t* mat);


static PyObject* test_read_data(PyObject* self, PyObject* args) {
    PyObject* py_path;
    mat_t* mat;
    char* path;
    status_t result;
    if (!PyArg_ParseTuple(args, "s", &py_path)) {
        return NULL;
    }

    path = PyUnicode_AsUTF8(py_path);
    result = read_data(&mat, path);
    Py_DECREF(py_path);
    if (result != SUCCESS) {
        return PyErr_Occurred();
    }

    return mat;
}
static PyObject* fit(PyObject *self, PyObject *args) {
    int argc, 
    point_t *centroids_list;
    PyObject *obj_initial_centroids, *obj_datapoints;
    PyObject *centroids_list_as_pyobject;
    int point_count, dims_count;
    int k;
    int max_iter;
    double epsilon;

    if (!self) return NULL;

    if (!PyArg_ParseTuple(args, "OOiiiid", &obj_initial_centroids, &obj_datapoints, &dims_count, &k, &point_count, &max_iter, &epsilon)) {
        return NULL;
    }

    centroids_list = calculate_centroids(obj_initial_centroids, obj_datapoints, dims_count, k, point_count, max_iter, epsilon);
    if (!centroids_list) return PyErr_NoMemory();
    centroids_list_as_pyobject = centroids_to_PyObject(centroids_list, k, dims_count);
    pointlist_free(&centroids_list, k);

    return centroids_list_as_pyobject;

}


static mat_t* PyListListFloat_to_Mat(PyObject* py_mat) {
    mat_t* mat;
    PyObject* py_row;
    PyObject* py_cell;
    uint i, j, h, w;
    double mat_cell;

    if (!py_mat) return NULL;
    
    mat = NULL;

    h = (int) PyList_Size(list); /* originally Py_ssize_t */
    if (h < 0) goto invalid_data; /* Not a list */
    if (h == 0) return mat_init(0,0);

    py_row = PyList_GetItem(py_mat, 0);
    w = (int) PyList_Size(py_row);
    if (w < 0) goto invalid_data;

    mat = mat_init(h,w);
    if (!mat) return PyErr_NoMemory();

    for (i=0; i<n; i++) {
        py_row = PyList_GetItem(py_mat, i);
        w = (uint) PyList_Size(py_row);
        if (w != mat->w) goto invalid_data;

        for (j=0; j<w; j++) {
            py_cell = PyList_GetItem(py_mat, j);
            if (!PyFloat_Check(py_cell)) goto invalid_data;
            mat_cell = PyFloat_AsDouble(py_cell);
            if (PyErr_Occurred()) goto error_occurred;
            mat_set(mat, i, j, mat_cell);
        }

    }

    return mat;

    error_occurred:
    invalid_data:
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

    if (!mat) return NULL;

    h = mat->h, w = mat->w;

    py_mat = PyList_New(h);
    if (!py_mat) return NULL;

    for (i=0; i<h; i++) {
        py_row = PyList_New(w);
        if (!py_row) goto error_malloc;

        for (j=0; j<w; j++) {
            cell = mat_get(mat, i, j);
            py_cell = Py_BuildValue("d", cell);
            PyList_SetItem(py_row, j, py_cell);
        }

        PyList_SetItem(py_mat, i, py_row);
    }

    return py_mat;

    error_malloc:
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

static PyObject* fit(PyObject *self, PyObject *args) {
    int argc, 
    point_t *centroids_list;
    PyObject *obj_initial_centroids, *obj_datapoints;
    PyObject *centroids_list_as_pyobject;
    int point_count, dims_count;
    int k;
    int max_iter;
    double epsilon;

    if (!self) return NULL;

    if (!PyArg_ParseTuple(args, "OOiiiid", &obj_initial_centroids, &obj_datapoints, &dims_count, &k, &point_count, &max_iter, &epsilon)) {
        return NULL;
    }

    centroids_list = calculate_centroids(obj_initial_centroids, obj_datapoints, dims_count, k, point_count, max_iter, epsilon);
    if (!centroids_list) return PyErr_NoMemory();
    centroids_list_as_pyobject = centroids_to_PyObject(centroids_list, k, dims_count);
    pointlist_free(&centroids_list, k);

    return centroids_list_as_pyobject;

}

static PyMethodDef spkmeansmoduleMethods[] = {
    {"fit", 
      (PyCFunction) fit,
      METH_VARARGS, 
      PyDoc_STR("This method performs the k-means algorithm with the specified arguments")},
    {NULL, NULL, 0, NULL} 
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    spkmeansmoduleMethods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_spkmeansmodule(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
