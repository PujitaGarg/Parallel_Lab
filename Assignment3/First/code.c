#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N (1 << 16)  

void daxpy(double a, double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = a * x[i] + y[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *X, *Y;
    double a = 2.0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = N / size;
    double *local_X = (double *)malloc(local_n * sizeof(double));
    double *local_Y = (double *)malloc(local_n * sizeof(double));

    if (rank == 0) {
        X = (double *)malloc(N * sizeof(double));
        Y = (double *)malloc(N * sizeof(double));

        
        for (int i = 0; i < N; i++) {
            X[i] = (double)i;
            Y[i] = (double)(N - i);
        }
    }

    start_time = MPI_Wtime();

   
    MPI_Scatter(X, local_n, MPI_DOUBLE, local_X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y, local_n, MPI_DOUBLE, local_Y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    daxpy(a, local_X, local_Y, local_n);

    
    MPI_Gather(local_X, local_n, MPI_DOUBLE, X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Time taken: %f seconds\n", end_time - start_time);

        
        free(X);
        free(Y);
    }

    free(local_X);
    free(local_Y);

    MPI_Finalize();
    return 0;
}
