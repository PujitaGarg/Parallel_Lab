#include <stdio.h>
#include <mpi.h>

static long num_steps = 100000; 

int main(int argc, char *argv[]) {
    int rank, size;
    double step, x, pi, sum = 0.0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    step = 1.0 / (double)num_steps;

    start_time = MPI_Wtime();

   
    for (int i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    
    double total_sum;
    MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    
    if (rank == 0) {
        pi = step * total_sum;
        printf("Approximate value of pi: %.16f\n", pi);
        printf("Time taken: %.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}