#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 200 

int main(int argc, char *argv[]) {
    int rank, size;
    int *local_array = NULL; 
    int local_sum = 0; 
    int global_sum = 0; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (ARRAY_SIZE % size != 0) {
        if (rank == 0) {
            printf("Error: Array size (%d) must be divisible by the number of processes (%d).\n", ARRAY_SIZE, size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_size = ARRAY_SIZE / size; 
    local_array = (int *)malloc(local_size * sizeof(int));

    
    for (int i = 0; i < local_size; i++) {
        local_array[i] = rank * local_size + i + 1; 
    }


    for (int i = 0; i < local_size; i++) {
        local_sum += local_array[i];
    }

    
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    printf("Process %d: Local sum = %d\n", rank, local_sum);
    if (rank == 0) {
        printf("Global sum = %d\n", global_sum);
    }


    free(local_array);

    MPI_Finalize();
    return 0;
}
