#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
   K-Means Clustering Implementation (Lloyd's Algorithm)
   
   This implementation partitions N datapoints in d dimensions into K clusters
   using the Lloyd's algorithm with Euclidean distance metric.
   ============================================================================ */

#define DEFAULT_MAX_ITERATIONS 400
#define CONVERGENCE_EPSILON 0.001

/* Calculates the Euclidean distance between two points in d dimensions */
double euclidean_distance(double *point1, double *point2, int dimension);

/* Lloyd's clustering algorithm: iteratively assigns points to nearest centroid
   and updates centroids until convergence or max iterations reached */
void lloyd_clustering(int num_points, int dimension, int num_clusters,
                      int max_iterations, double convergence_threshold,
                      double **dataset);

/* Character validation helper */
int is_digit(char c);

int is_digit(char c) {
    return (c >= '0' && c <= '9');
}

/* Validates command-line argument as a valid integer */
int validate_integer_argument(const char *arg, const char *error_msg) {
    int i, j;
    
    for (i = 0; arg[i] != '\0'; i++) {
        if (!is_digit(arg[i])) {
            if (arg[i] == '.') {
                for (j = i + 1; arg[j] != '\0'; j++) {
                    if (arg[j] != '0') {
                        printf("%s\n", error_msg);
                        return 0;
                    }
                }
            } else {
                printf("%s\n", error_msg);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int num_clusters, max_iterations, num_points, dimension;
    char *line = NULL;
    size_t len = 0;
    long read;
    int i, j;
    char *parse_ptr;
    int scan_result;
    double **dataset = NULL;
    double convergence_threshold = CONVERGENCE_EPSILON;
    double **temp;

    /* Validate command-line arguments */
    if (argc < 2) {
        printf("Incorrect number of clusters!\n");
        return 1;
    }

    /* Parse first argument (number of clusters) */
    if (!validate_integer_argument(argv[1], "Incorrect number of clusters!")) {
        return 1;
    }
    num_clusters = atoi(argv[1]);

    /* Parse second argument (max iterations) if provided */
    max_iterations = DEFAULT_MAX_ITERATIONS;
    if (argc == 3) {
        if (!validate_integer_argument(argv[2], "Incorrect maximum iteration!")) {
            return 1;
        }
        max_iterations = atoi(argv[2]);
    } else if (argc > 3) {
        printf("Incorrect number of clusters!\n");
        return 1;
    }

    /* Validate parameter ranges */
    if (num_clusters <= 1) {
        printf("Incorrect number of clusters!\n");
        return 1;
    }

    if (max_iterations <= 1 || max_iterations >= 1000) {
        printf("Incorrect maximum iteration!\n");
        return 1;
    }

    /* Read input data from stdin */
    num_points = 0;
    dimension = -1;

    while ((read = getline(&line, &len, stdin)) != -1) {
        if (line[0] == '\n' || line[0] == '\0') continue;

        /* Determine dimensionality from first line */
        if (dimension == -1) {
            dimension = 1;
            for (i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
                if (line[i] == ',') dimension++;
            }
        }

        /* Allocate space for new datapoint */
        temp = realloc(dataset, (num_points + 1) * sizeof(double *));
        dataset = temp;
        dataset[num_points] = malloc(dimension * sizeof(double));

        /* Parse comma-separated values */
        parse_ptr = line;
        for (j = 0; j < dimension; j++) {
            sscanf(parse_ptr, "%lf%n", &dataset[num_points][j], &scan_result);
            parse_ptr += scan_result;
            if (j < dimension - 1 && *parse_ptr == ',') {
                parse_ptr++;
            }
        }

        num_points++;
    }

    free(line);

    /* Validate that K < N */
    if (num_clusters >= num_points) {
        printf("Incorrect number of clusters!\n");
        return 1;
    }

    /* Run Lloyd's clustering algorithm */
    lloyd_clustering(num_points, dimension, num_clusters, max_iterations,
                     convergence_threshold, dataset);

    /* Free allocated memory */
    for (i = 0; i < num_points; i++) {
        free(dataset[i]);
    }
    free(dataset);

    return 0;
}

/* Euclidean distance: d(p,q) = sqrt(sum((p_i - q_i)^2)) */
double euclidean_distance(double *point1, double *point2, int dimension) {
    double sum_of_squares = 0.0;
    int i;
    
    for (i = 0; i < dimension; i++) {
        double delta = point1[i] - point2[i];
        sum_of_squares += delta * delta;
    }
    
    return sqrt(sum_of_squares);
}

/* Lloyd's K-Means Algorithm
   
   Algorithm:
   1. Initialize centroids as first K datapoints
   2. For each iteration:
      a. Assign each point to nearest centroid
      b. Update centroids as mean of assigned points
      c. Check if max centroid movement < epsilon (convergence)
   3. Output final centroids
*/
void lloyd_clustering(int num_points, int dimension, int num_clusters,
                      int max_iterations, double convergence_threshold,
                      double **dataset) {
    int i, j, iteration_count;
    double max_centroid_shift;
    
    /* Allocate centroid arrays */
    double **cluster_representatives = malloc(num_clusters * sizeof(double *));
    double **new_representatives = malloc(num_clusters * sizeof(double *));
    int *cluster_sizes = malloc(num_clusters * sizeof(int));

    for (i = 0; i < num_clusters; i++) {
        cluster_representatives[i] = malloc(dimension * sizeof(double));
        new_representatives[i] = malloc(dimension * sizeof(double));
    }

    /* Initialize centroids as first K datapoints */
    for (i = 0; i < num_clusters; i++) {
        for (j = 0; j < dimension; j++) {
            cluster_representatives[i][j] = dataset[i][j];
        }
    }

    /* Main clustering loop */
    for (iteration_count = 0; iteration_count < max_iterations; iteration_count++) {
        
        /* Reset cluster accumulators */
        for (i = 0; i < num_clusters; i++) {
            cluster_sizes[i] = 0;
            for (j = 0; j < dimension; j++) {
                new_representatives[i][j] = 0.0;
            }
        }

        /* Assignment step: assign each point to nearest centroid */
        for (i = 0; i < num_points; i++) {
            double minimum_distance = euclidean_distance(dataset[i],
                                                        cluster_representatives[0],
                                                        dimension);
            int closest_cluster_index = 0;
            
            for (j = 1; j < num_clusters; j++) {
                double distance = euclidean_distance(dataset[i],
                                                    cluster_representatives[j],
                                                    dimension);
                if (distance < minimum_distance) {
                    minimum_distance = distance;
                    closest_cluster_index = j;
                }
            }
            
            /* Accumulate point coordinates and count */
            cluster_sizes[closest_cluster_index]++;
            for (j = 0; j < dimension; j++) {
                new_representatives[closest_cluster_index][j] += dataset[i][j];
            }
        }

        /* Update step: recompute centroids as mean of assigned points */
        for (i = 0; i < num_clusters; i++) {
            if (cluster_sizes[i] > 0) {
                for (j = 0; j < dimension; j++) {
                    new_representatives[i][j] /= cluster_sizes[i];
                }
            } else {
                /* Empty cluster: keep previous centroid */
                for (j = 0; j < dimension; j++) {
                    new_representatives[i][j] = cluster_representatives[i][j];
                }
            }
        }

        /* Convergence check: compute max centroid movement */
        max_centroid_shift = 0.0;
        for (i = 0; i < num_clusters; i++) {
            double centroid_shift = euclidean_distance(cluster_representatives[i],
                                                      new_representatives[i],
                                                      dimension);
            if (centroid_shift > max_centroid_shift) {
                max_centroid_shift = centroid_shift;
            }
        }

        /* Update centroids for next iteration */
        for (i = 0; i < num_clusters; i++) {
            for (j = 0; j < dimension; j++) {
                cluster_representatives[i][j] = new_representatives[i][j];
            }
        }

        /* Early termination if converged */
        if (max_centroid_shift < convergence_threshold) {
            break;
        }
    }

    /* Output final centroids */
    for (i = 0; i < num_clusters; i++) {
        for (j = 0; j < dimension; j++) {
            printf("%.4f", cluster_representatives[i][j]);
            if (j < dimension - 1)
                printf(",");
        }
        printf("\n");
    }

    /* Free allocated memory */
    for (i = 0; i < num_clusters; i++) {
        free(cluster_representatives[i]);
        free(new_representatives[i]);
    }
    free(cluster_representatives);
    free(new_representatives);
    free(cluster_sizes);
}
