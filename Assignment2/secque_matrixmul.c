#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 70 

void matrix_multiply_serial(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double A[N][N], B[N][N], C[N][N];
    double start_time, run_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = (double)(i + j);
                B[i][j] = (double)(i - j);
            }
        }
    }

   
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rows_per_process = N / size;
    double local_A[rows_per_process][N];
    double local_C[rows_per_process][N];

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, local_A, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    start_time = MPI_Wtime();
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }
    run_time = MPI_Wtime() - start_time;

    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE, C, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Parallel matrix multiplication time: %f seconds\n", run_time);
        double serial_C[N][N];
        start_time = MPI_Wtime();
        matrix_multiply_serial(A, B, serial_C);
        run_time = MPI_Wtime() - start_time;
        printf("Serial matrix multiplication time: %f seconds\n", run_time);
    }

    MPI_Finalize();
    return 0;
}
