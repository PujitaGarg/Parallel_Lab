/*Odd-Even Sort Algorithm
1.Odd Phase: Compare and swap elements at odd indices with their right neighbors.
2.Even Phase: Compare and swap elements at even indices with their right neighbors.
3.Repeat the above phases until the array is sorted.*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void local_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *global_data = NULL;
    int *local_data;         
    int local_size;          
    int n = 16;             

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    local_size = n / size;

    
    local_data = (int *)malloc(local_size * sizeof(int));


    if (rank == 0) {
        global_data = (int *)malloc(n * sizeof(int));
        printf("Unsorted array: ");
        for (int i = 0; i < n; i++) {
            global_data[i] = rand() % 100; // Fill with random numbers
            printf("%d ", global_data[i]);
        }
        printf("\n");
    }

    
    MPI_Scatter(global_data, local_size, MPI_INT, local_data, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    local_sort(local_data, local_size);

    int temp;
    int sorted = 0;
    while (!sorted) {
        sorted = 1;

        // Odd Phase
        for (int i = 1; i < size; i += 2) {
            if (rank == i && rank + 1 < size) {
                
                MPI_Send(&local_data[local_size - 1], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                
                MPI_Recv(&temp, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (local_data[local_size - 1] > temp) {
                    swap(&local_data[local_size - 1], &temp);
                    sorted = 0;
                }
               
                MPI_Send(&temp, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            } else if (rank == i + 1 && rank < size) {
                
                MPI_Recv(&temp, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               
                MPI_Send(&local_data[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                if (temp > local_data[0]) {
                    swap(&temp, &local_data[0]);
                    sorted = 0;
                }
              
                MPI_Recv(&local_data[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Even Phase
        for (int i = 0; i < size; i += 2) {
            if (rank == i && rank + 1 < size) {
               
                MPI_Send(&local_data[local_size - 1], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                
                MPI_Recv(&temp, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (local_data[local_size - 1] > temp) {
                    swap(&local_data[local_size - 1], &temp);
                    sorted = 0;
                }
                
                MPI_Send(&temp, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            } else if (rank == i + 1 && rank < size) {
                
                MPI_Recv(&temp, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                MPI_Send(&local_data[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                if (temp > local_data[0]) {
                    swap(&temp, &local_data[0]);
                    sorted = 0;
                }
                
                MPI_Recv(&local_data[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    
    MPI_Gather(local_data, local_size, MPI_INT, global_data, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the sorted array in the root process
    if (rank == 0) {
        printf("Sorted array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", global_data[i]);
        }
        printf("\n");
        free(global_data);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}