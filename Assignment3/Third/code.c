#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>


int is_prime(int n) {
    if (n < 2) return 0; // Not prime
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return 0; // Not prime
    }
    return 1; // Prime
}

int main(int argc, char *argv[]) {
    int rank, size;
    int max_value = 100; 
    int num_to_test, result;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        
        int primes_found = 0;
        int next_num = 2; // Start testing from 2

        
        while (next_num <= max_value) {
            
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (result == 0) {
                
                MPI_Send(&next_num, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                next_num++;
            } else {
                
                if (result > 0) {
                    printf("%d is prime\n", result);
                    primes_found++;
                }
            }
        }

        
        int terminate = -1;
        for (int i = 1; i < size; i++) {
            MPI_Send(&terminate, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        printf("Total primes found: %d\n", primes_found);
    } else {
        
        while (1) {
            
            int request = 0;
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            
            MPI_Recv(&num_to_test, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            if (num_to_test == -1) {
                
                break;
            }

            
            if (is_prime(num_to_test)) {
                result = num_to_test; 
            } else {
                result = -num_to_test; 
            }

            
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
