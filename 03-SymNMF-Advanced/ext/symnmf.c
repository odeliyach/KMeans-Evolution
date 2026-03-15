#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

void print_mat(double** M, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (j > 0) {
                printf(",");
            }
            printf("%.4f", M[i][j]);
        }
        printf("\n");
    }
}

void handle_error_and_exit() {
    printf("An Error Has Occurred\n");
    exit(1);
}

void free_matrix(double** M, int rows) {
    int i;
    if (M == NULL) return;
    for (i = 0; i < rows; i++) {
        if (M[i] != NULL) {
            free(M[i]); 
        }
    }
    free(M);
}

double** compute_similarity_matrix_fromX(double** X, int N, int d) {
    double** A = malloc(N * sizeof(double*));
    double dist;
    int i,j,k;
    if (A == NULL) {
        handle_error_and_exit();
    }
    dist = 0;
    for (i = 0; i < N; i++) {
        A[i] = calloc(N, sizeof(double));
        if (A[i] == NULL) {
            free_matrix(A, i);
            handle_error_and_exit();
        }
        for (j = 0; j < N; j++) {
            if (i == j) 
             continue;
            for (k = 0; k < d; k++) {
                double diff = X[i][k] - X[j][k];
                dist += diff * diff;
            }
            A[i][j] = exp(-dist / 2.0);
            dist = 0;
        }
    }
    return A;
}

double** compute_diagonal_degre_matrix(double** A, int N) {
    int i, j;
    double sum;
    double** D = malloc(N * sizeof(double*));
    if (D == NULL) {
        handle_error_and_exit();
    }
    for (i = 0; i < N; i++) {
        D[i] = calloc(N, sizeof(double));
        if (D[i] == NULL) {
            free_matrix(D, i);
            handle_error_and_exit();
        }
        sum = 0;
        for (j = 0; j < N; j++) 
         sum += A[i][j];
        D[i][i] = sum;
    }
    return D;
}

double** compute_normalized_similarity_matrix(double** A, double** D, int N) {
    int i, j;
    double di, dj;
    const double epsilon = 1e-16;
    double** W = malloc(N * sizeof(double*));
    if (W == NULL) handle_error_and_exit();
    for (i = 0; i < N; i++) {
        W[i] = calloc(N, sizeof(double));
        if (W[i] == NULL) {
            free_matrix(W, i);
            handle_error_and_exit();
        }

        di = (D[i][i] == 0.0) ? 1.0 / sqrt(epsilon) : 1.0 / sqrt(D[i][i]);

        for (j = 0; j < N; j++) {
            dj = (D[j][j] == 0.0) ? 1.0 / sqrt(epsilon) : 1.0 / sqrt(D[j][j]);
            W[i][j] = di * A[i][j] * dj;
        }
    }

    return W;
}

double** save_file_to_mat(const char* file_name, int* N, int* d) {
    double** mat = NULL,** temp_mat;
    int rows=0, cols=-1, count;
    char* line= NULL, *cordinate_value;
    double* row,* temp;
    size_t len=0;
    ssize_t read;
    FILE* file = fopen(file_name, "r");
    if (!file) { handle_error_and_exit();}
    while ((read = getline(&line, &len, file)) != -1) {
        count = 0;
        row = NULL;
        cordinate_value = strtok(line, ", \t\n\r");
        while (cordinate_value) {
            double value = atof(cordinate_value);
            temp = realloc(row, (count + 1) * sizeof(double));
            if (temp == NULL) {
                free(row);
                free_matrix(mat, rows);
                handle_error_and_exit(); 
            }
            row = temp;
            row[count] = value;
            count++;
            cordinate_value = strtok(NULL, ", \t\n\r"); 
        }
        if (cols == -1) { cols = count;}
        if (count != cols) {
            free(row);
            free_matrix(mat, rows);
            handle_error_and_exit(); 
        }
        temp_mat = realloc(mat, (rows + 1) * sizeof(double*));
        if (temp_mat == NULL) {
            free(row);
            free_matrix(mat, rows);
            handle_error_and_exit();
        }
        mat = temp_mat;
        mat[rows++] = row;
    }
    free(line);
    fclose(file);
    *N = rows;
    *d = cols;
    return mat;
}

int main(int argc, char* argv[]) {
    int N, d;
    double** A,**D,**W,** X;
    char* goal,* fileName;
    (void)argc;
    goal = argv[1];
    fileName = argv[2];
    X = save_file_to_mat(fileName, &N, &d);
    if (N == 0 || d <= 0 || X == NULL) {
       handle_error_and_exit();
    }
    if (strcmp(goal, "sym") == 0) {
        A = compute_similarity_matrix_fromX(X, N, d);
        print_mat(A, N, N);
        free_matrix(A, N);
    } 
    else if (strcmp(goal, "ddg") == 0){
        A = compute_similarity_matrix_fromX(X, N, d);
        D = compute_diagonal_degre_matrix(A, N);
        print_mat(D, N, N);
        free_matrix(A, N);
        free_matrix(D, N);
    } 
    else if (strcmp(goal, "norm") == 0) {
        A = compute_similarity_matrix_fromX(X, N, d);
        D = compute_diagonal_degre_matrix(A, N);
        W = compute_normalized_similarity_matrix(A, D, N);
        print_mat(W, N, N);
        free_matrix(A, N);
        free_matrix(D, N);
        free_matrix(W, N);
    } 
    else {
       printf("An Error Has Occurred\n");
       free_matrix(X, N);   
       return 1;
    }
    free_matrix(X, N);
    return 0;
}
