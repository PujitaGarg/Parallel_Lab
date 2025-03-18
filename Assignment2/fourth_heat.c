#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NX 100  // Number of grid points in x-direction
#define NY 100  // Number of grid points in y-direction
#define NT 100  // Number of time steps
#define DX 0.1  // Grid spacing in x-direction
#define DY 0.1  // Grid spacing in y-direction
#define DT 0.01 // Time step
#define ALPHA 0.01 // Thermal diffusivity

void initialize(double *u, int nx, int ny) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            u[i * ny + j] = 0.0;
        }
    }
    
    u[(nx/2) * ny + (ny/2)] = 100.0;
}

void update(double *u, double *u_new, int nx, int ny) {
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            u_new[i * ny + j] = u[i * ny + j] + ALPHA * DT * (
                (u[(i+1) * ny + j] - 2 * u[i * ny + j] + u[(i-1) * ny + j]) / (DX * DX) +
                (u[i * ny + (j+1)] - 2 * u[i * ny + j] + u[i * ny + (j-1)]) / (DY * DY));
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx_local = NX / size;
    double *u = (double *)malloc(nx_local * NY * sizeof(double));
    double *u_new = (double *)malloc(nx_local * NY * sizeof(double));

    initialize(u, nx_local, NY);

    for (int t = 0; t < NT; t++) {
        update(u, u_new, nx_local, NY);

        
        if (rank > 0) {
            MPI_Send(u_new + NY, NY, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(u_new, NY, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(u_new + (nx_local - 2) * NY, NY, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(u_new + (nx_local - 1) * NY, NY, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

      
        double *temp = u;
        u = u_new;
        u_new = temp;
    }


    if (rank == 0) {
        double *u_global = (double *)malloc(NX * NY * sizeof(double));
        MPI_Gather(u, nx_local * NY, MPI_DOUBLE, u_global, nx_local * NY, MPI_DOUBLE, 0, MPI_COMM_WORLD);

       
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                printf("%f ", u_global[i * NY + j]);
            }
            printf("\n");
        }
        free(u_global);
    } 
    else {
        MPI_Gather(u, nx_local * NY, MPI_DOUBLE, NULL, nx_local * NY, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return 0;
}
