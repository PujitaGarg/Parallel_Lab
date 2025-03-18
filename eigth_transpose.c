#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 40 

// Function to print a matrix
void print_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *matrix = NULL; 
    int *local_matrix = NULL;
    int *local_transpose = NULL; 
    int *global_transpose = NULL; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (N % size != 0) {
        if (rank == 0) {
            printf("Error: Matrix size (%d) must be divisible by the number of processes (%d).\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_rows = N / size; 
    int local_cols = N; 

    
    local_matrix = (int *)malloc(local_rows * local_cols * sizeof(int));
    local_transpose = (int *)malloc(local_cols * local_rows * sizeof(int));

    
    if (rank == 0) {
        matrix = (int *)malloc(N * N * sizeof(int));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = i * N + j; 
            }
        }
        printf("Original Matrix:\n");
        print_matrix(matrix, N, N);
    }

   
    MPI_Scatter(matrix, local_rows * N, MPI_INT, local_matrix, local_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_transpose[j * local_rows + i] = local_matrix[i * N + j];
        }
    }

    
    if (rank == 0) {
        global_transpose = (int *)malloc(N * N * sizeof(int));
    }
    MPI_Gather(local_transpose, local_rows * N, MPI_INT, global_transpose, local_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

  
    if (rank == 0) {
        printf("Transposed Matrix:\n");
        print_matrix(global_transpose, N, N);
    }


    free(local_matrix);
    free(local_transpose);
    if (rank == 0) {
        free(matrix);
        free(global_transpose);
    }

    MPI_Finalize();
    return 0;
}