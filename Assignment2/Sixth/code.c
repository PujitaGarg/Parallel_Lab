#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

double dot_product(double *a, double *b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n;
    double *a = NULL, *b = NULL;
    double *local_a, *local_b;
    double local_dot, global_dot;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter vector size: ");
        scanf("%d", &n);

        if (n % size != 0) {
            printf("Vector size must be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        a = (double *)malloc(n * sizeof(double));
        b = (double *)malloc(n * sizeof(double));

        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            a[i] = (rand() % 100) + 1;
            b[i] = (rand() % 100) + 1;
        }

        // Display initial vectors
        printf("\nVector A: ");
        for (int i = 0; i < n; i++) {
            printf("%.2f ", a[i]);
        }
        printf("\n");

        printf("Vector B: ");
        for (int i = 0; i < n; i++) {
            printf("%.2f ", b[i]);
        }
        printf("\n");
    }

    // Broadcast vector size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;
    local_a = (double *)malloc(local_n * sizeof(double));
    local_b = (double *)malloc(local_n * sizeof(double));

    // Scatter vectors to all processes
    MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_n, MPI_DOUBLE, local_b, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local dot product
    local_dot = dot_product(local_a, local_b, local_n);

    // Display local dot product from each process
    printf("Process %d: Local dot product = %.2f\n", rank, local_dot);

    // Reduce to compute global dot product at root process
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal dot product = %.2f\n", global_dot);
        free(a);
        free(b);
    }

    free(local_a);
    free(local_b);
    MPI_Finalize();

    return 0;
}
