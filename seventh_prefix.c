#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to compute the prefix sum of an array
void prefix_sum(int *array, int n) {
    for (int i = 1; i < n; i++) {
        array[i] += array[i - 1];
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 16; // Size of the global array
    int local_n = n / size; // Size of the local array for each process

    // Allocate memory for the local array
    int *local_array = (int *)malloc(local_n * sizeof(int));

    // Initialize the local array with some values
    for (int i = 0; i < local_n; i++) {
        local_array[i] = rank * local_n + i + 1;
    }

    // Compute the local prefix sum
    prefix_sum(local_array, local_n);

    // Gather the last element of each local prefix sum
    int *last_elements = NULL;
    if (rank == 0) {
        last_elements = (int *)malloc(size * sizeof(int));
    }
    MPI_Gather(&local_array[local_n - 1], 1, MPI_INT, last_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the prefix sum of the last elements on the root process
    if (rank == 0) {
        prefix_sum(last_elements, size);
    }

    // Scatter the prefix sums of the last elements to all processes
    int offset = 0;
    MPI_Scatter(last_elements, 1, MPI_INT, &offset, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Add the offset to the local prefix sum
    for (int i = 0; i < local_n; i++) {
        local_array[i] += offset;
    }

    // Gather the final prefix sums from all processes
    int *global_prefix_sum = NULL;
    if (rank == 0) {
        global_prefix_sum = (int *)malloc(n * sizeof(int));
    }
    MPI_Gather(local_array, local_n, MPI_INT, global_prefix_sum, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the final prefix sum on the root process
    if (rank == 0) {
        printf("Global Prefix Sum: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", global_prefix_sum[i]);
        }
        printf("\n");
        free(global_prefix_sum);
        free(last_elements);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}