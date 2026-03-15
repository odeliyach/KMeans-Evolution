#ifndef CLUSTERING_H_
#define CLUSTERING_H_

/* ============================================================================
   Lloyd's K-Means Clustering Algorithm Header
   
   High-performance C implementation of Lloyd's clustering algorithm
   for use as a Python extension module.
   ============================================================================ */

/**
 * lloyd_clustering - Execute Lloyd's clustering algorithm
 * 
 * @num_points: Number of data points
 * @dimension: Feature dimension
 * @num_clusters: Number of clusters
 * @max_iterations: Maximum iterations
 * @epsilon: Convergence threshold
 * @data_points: Array of data points (num_points × dimension)
 * @initial_centroids: Array of initial centroids (num_clusters × dimension)
 * 
 * Returns: Array of final centroids (num_clusters × dimension)
 *          Caller is responsible for freeing memory
 */
double** lloyd_clustering(
    int num_points,
    int dimension,
    int num_clusters,
    int max_iterations,
    double epsilon,
    double** data_points,
    double** initial_centroids
);

#endif
