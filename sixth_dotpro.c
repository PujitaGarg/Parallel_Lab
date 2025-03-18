#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double dot_product(double *a, double *b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 100; 
    double *a, *b;
    double local_dot, global_dot;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int local_n = n / size;
    a = (double *)malloc(local_n * sizeof(double));
    b = (double *)malloc(local_n * sizeof(double));

    for (int i = 0; i < local_n; i++) {
        a[i] = 0.1 * (rank * local_n + i);
        b[i] = 0.2 * (rank * local_n + i);
    }

   
    local_dot = dot_product(a, b, local_n);

    
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   
    if (rank == 0) {
        printf("Dot product: %f\n", global_dot);
    }

    free(a);
    free(b);
    MPI_Finalize();

    return 0;
}