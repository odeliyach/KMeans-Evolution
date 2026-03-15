#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "clustering.h"

/* ============================================================================
   Python C API Extension Module - clustering_engine
   
   Provides Python binding for high-performance C clustering implementation.
   ============================================================================ */

/**
 * clustering_engine.fit(data, centroids, k, max_iter, epsilon)
 * 
 * Executes Lloyd's clustering algorithm on the input data.
 * 
 * Args:
 *     data: List of lists (N samples × D features)
 *     centroids: List of lists (K centroids × D features)
 *     k: Number of clusters (int)
 *     max_iter: Maximum iterations (int)
 *     epsilon: Convergence threshold (float)
 * 
 * Returns:
 *     List of lists representing final centroids
 */
static PyObject* clustering_fit(PyObject* self, PyObject* args) {
    PyObject *data_obj, *centroids_obj;
    int k, max_iter;
    double epsilon;
    Py_ssize_t i, j, num_points, dimension;
    
    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "OOiid",
                         &data_obj,
                         &centroids_obj,
                         &k,
                         &max_iter,
                         &epsilon)) {
        PyErr_SetString(PyExc_TypeError,
                       "fit() expects (list, list, int, int, float)");
        return NULL;
    }
    
    /* Validate input types */
    if (!PyList_Check(data_obj) || !PyList_Check(centroids_obj)) {
        PyErr_SetString(PyExc_TypeError,
                       "data and centroids must be lists");
        return NULL;
    }
    
    num_points = PyList_Size(data_obj);
    if (num_points == 0) {
        PyErr_SetString(PyExc_ValueError, "data cannot be empty");
        return NULL;
    }
    
    /* Get dimension from first data point */
    PyObject *first_row = PyList_GetItem(data_obj, 0);
    if (!PyList_Check(first_row)) {
        PyErr_SetString(PyExc_TypeError, "data must be list of lists");
        return NULL;
    }
    dimension = PyList_Size(first_row);
    
    /* Allocate C arrays for data */
    double **data_points = malloc(num_points * sizeof(double*));
    if (!data_points) {
        PyErr_NoMemory();
        return NULL;
    }
    
    /* Convert Python data to C arrays */
    for (i = 0; i < num_points; i++) {
        PyObject *row = PyList_GetItem(data_obj, i);
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "data must be list of lists");
            free(data_points);
            return NULL;
        }
        
        if (PyList_Size(row) != dimension) {
            PyErr_SetString(PyExc_ValueError,
                           "all data points must have same dimension");
            free(data_points);
            return NULL;
        }
        
        data_points[i] = malloc(dimension * sizeof(double));
        if (!data_points[i]) {
            PyErr_NoMemory();
            free(data_points);
            return NULL;
        }
        
        for (j = 0; j < dimension; j++) {
            PyObject *item = PyList_GetItem(row, j);
            data_points[i][j] = PyFloat_AsDouble(item);
            
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError,
                               "data elements must be numeric");
                free(data_points[i]);
                free(data_points);
                return NULL;
            }
        }
    }
    
    /* Allocate C arrays for centroids */
    double **centroids = malloc(k * sizeof(double*));
    if (!centroids) {
        for (i = 0; i < num_points; i++) free(data_points[i]);
        free(data_points);
        PyErr_NoMemory();
        return NULL;
    }
    
    /* Convert Python centroids to C arrays */
    for (i = 0; i < k; i++) {
        PyObject *row = PyList_GetItem(centroids_obj, i);
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError,
                           "centroids must be list of lists");
            free(centroids);
            for (j = 0; j < num_points; j++) free(data_points[j]);
            free(data_points);
            return NULL;
        }
        
        centroids[i] = malloc(dimension * sizeof(double));
        if (!centroids[i]) {
            PyErr_NoMemory();
            free(centroids);
            for (j = 0; j < num_points; j++) free(data_points[j]);
            free(data_points);
            return NULL;
        }
        
        for (j = 0; j < dimension; j++) {
            PyObject *item = PyList_GetItem(row, j);
            centroids[i][j] = PyFloat_AsDouble(item);
            
            if (PyErr_Occurred()) {
                PyErr_SetString(PyExc_TypeError,
                               "centroid elements must be numeric");
                free(centroids[i]);
                free(centroids);
                for (j = 0; j < num_points; j++) free(data_points[j]);
                free(data_points);
                return NULL;
            }
        }
    }
    
    /* Call C clustering function */
    double **result = lloyd_clustering(
        (int)num_points,
        (int)dimension,
        k,
        max_iter,
        epsilon,
        data_points,
        centroids
    );
    
    /* Convert C result back to Python list */
    PyObject *result_list = PyList_New(k);
    if (!result_list) {
        PyErr_NoMemory();
        for (i = 0; i < k; i++) free(result[i]);
        free(result);
        free(centroids);
        for (i = 0; i < num_points; i++) free(data_points[i]);
        free(data_points);
        return NULL;
    }
    
    for (i = 0; i < k; i++) {
        PyObject *row = PyList_New(dimension);
        if (!row) {
            PyErr_NoMemory();
            Py_DECREF(result_list);
            for (j = 0; j < k; j++) free(result[j]);
            free(result);
            free(centroids);
            for (j = 0; j < num_points; j++) free(data_points[j]);
            free(data_points);
            return NULL;
        }
        
        for (j = 0; j < dimension; j++) {
            PyObject *val = PyFloat_FromDouble(result[i][j]);
            if (!val) {
                PyErr_NoMemory();
                Py_DECREF(row);
                Py_DECREF(result_list);
                for (int k_idx = 0; k_idx < k; k_idx++) free(result[k_idx]);
                free(result);
                free(centroids);
                for (int p_idx = 0; p_idx < num_points; p_idx++)
                    free(data_points[p_idx]);
                free(data_points);
                return NULL;
            }
            PyList_SetItem(row, j, val);
        }
        PyList_SetItem(result_list, i, row);
    }
    
    /* Free all C memory */
    for (i = 0; i < k; i++) {
        free(result[i]);
        free(centroids[i]);
    }
    free(result);
    free(centroids);
    
    for (i = 0; i < num_points; i++) {
        free(data_points[i]);
    }
    free(data_points);
    
    return result_list;
}

/* Method definition */
static PyMethodDef clustering_methods[] = {
    {
        "fit",
        (PyCFunction)clustering_fit,
        METH_VARARGS,
        "fit(data, centroids, k, max_iter, epsilon)\n"
        "\n"
        "Execute Lloyd's k-means clustering algorithm.\n"
        "\n"
        "Args:\n"
        "    data: List of data points (N × D)\n"
        "    centroids: List of initial centroids (K × D)\n"
        "    k: Number of clusters\n"
        "    max_iter: Maximum iterations\n"
        "    epsilon: Convergence threshold\n"
        "\n"
        "Returns:\n"
        "    List of final centroids (K × D)\n"
    },
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef clustering_module = {
    PyModuleDef_HEAD_INIT,
    "clustering_engine",
    "High-performance Lloyd's clustering with Python C API",
    -1,
    clustering_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_clustering_engine(void) {
    PyObject *m;
    m = PyModule_Create(&clustering_module);
    if (!m) {
        return NULL;
    }
    return m;
}
