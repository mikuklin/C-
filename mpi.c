#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv) {

    int rank, size;
    double start, end;
    double **matrix;
    int n = atoi(argv[1]);
    matrix = malloc(n * sizeof(double*));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 0; i < n; ++i) {
        matrix[i] = malloc(2 * n * sizeof(double));
    }
    if (rank == 0) {
        start = MPI_Wtime();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 2 * n; ++j) {
                matrix[i][j] = (j < n) ? rand() % 10 : ((j - n == i) ? 1 : 0);
            }
        }
    }
  
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&(matrix[0][0]), n * n * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double lead[n * 2];
    int block = n / (size - 1),
        start_row = block * rank, 
        end_row = block * (rank + 1);
    if (rank == size - 1)
        end_row = n;
    
    for (int i = 0; i < n; i++) {
        if (rank == 0) {
            double maxmod = 0;
            int pos = i;
            for (int k = i; k < n; ++k) {
                if (fabs(matrix[k][i]) > maxmod) {
                    pos = k;
                    maxmod = fabs(matrix[k][i]);
                }
            }
            if (pos != i) {
                double *tmp = matrix[i];
                matrix[i] = matrix[pos];
                matrix[pos] = tmp;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&(matrix[0][0]), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        

        if (rank == i / block) {
            for (int j = 0; j < n * 2; j++) {
                lead[j] = matrix[i][j];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(lead, n * 2, MPI_DOUBLE, i / block, MPI_COMM_WORLD);

        for (int j = start_row; j < end_row; j++) {
            if (j == i) {
                double d = matrix[i][i];
                for (int k = 0; k < n * 2; k++) 
                    matrix[j][k] /= d;
                continue;
            }
            double d = matrix[j][i] / lead[i];
            for (int k = 0; k < n * 2; k++) {
                matrix[j][k] -= d * lead[k];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        printf("%f\n", end - start);
    } 
    MPI_Finalize();
    return 0;
}
