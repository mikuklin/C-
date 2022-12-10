#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char **argv)
{
    int n, t;
    n = atoi(argv[1]);
    t = atoi(argv[2]);
    omp_set_num_threads(t);
    double **matrix = malloc(n * sizeof(double*));
    double **inverse = malloc(n * sizeof(double*));

    for (int i = 0; i < n; ++i) {
        matrix[i] = malloc(n * sizeof(double));
        inverse[i] = malloc(n * sizeof(double));
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = 1 ? i + j == n - 1 : 0;
            inverse[i][j] = 1 ? i == j : 0;
        }
    }

    double start; 
    double end; 
    start = omp_get_wtime();
    for (int i = 0; i < n; ++i) {

        double maxmod = 0;
        int pos = i;
        for (int k = i; k < n; ++k) {
            if (abs(matrix[k][i]) > maxmod) {
                pos = k;
                maxmod = fabs(matrix[k][i]);
            }
        }
        if (pos != i) {
            double *tmp = matrix[i];
            matrix[i] = matrix[pos];
            matrix[pos] = tmp;

            tmp = inverse[i];
            inverse[i] = inverse[pos];
            inverse[pos] = tmp;
        }

        double tmp = matrix[i][i];
        #pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            matrix[i][j] /= tmp;
            inverse[i][j] /= tmp;
        }
        #pragma omp parallel for
        for (int k = i + 1; k < n; ++k) {
            tmp = matrix[k][i];
            #pragma omp parallel for
            for (int j = 0; j < n; ++j) {
                inverse[k][j] -= tmp * inverse[i][j];
                matrix[k][j] -= matrix[i][j] * tmp;
            }
        }
    }
    
    for (int i = n - 1; i >= 0; --i)
    {
    #pragma omp parallel for
        for (int k = 0; k < i; ++k) {
            double tmp = matrix[k][i];
            #pragma omp parallel for
            for (int j = 0; j < n; ++j) {
                inverse[k][j] -= tmp * inverse[i][j];
                matrix[k][j] -= matrix[i][j] * tmp;
            }
        }
    }
    end = omp_get_wtime(); 
    printf("%f\n", end - start);
    return 0;
}
