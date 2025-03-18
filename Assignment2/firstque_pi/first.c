// The Monte Carlo method generates random points inside a square and counts the number of points that fall inside a circle inscribed in the square to estimate pi

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define TOTAL_POINTS 1000000

int main(int argc, char *argv[]) {
    int rank, size;
    int points_per_process, total_points_in_circle = 0, points_in_circle = 0;
    double x, y, pi_estimate;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    points_per_process = TOTAL_POINTS / size;

    srand(time(NULL) + rank); // Seed the random number generator

    start_time = MPI_Wtime();

    for (int i = 0; i < points_per_process; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            points_in_circle++;
        }
    }

    MPI_Reduce(&points_in_circle, &total_points_in_circle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi_estimate = 4.0 * (double)total_points_in_circle / TOTAL_POINTS;
        end_time = MPI_Wtime();

        printf("Estimated value of Pi: %f\n", pi_estimate);
        printf("Time taken: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
