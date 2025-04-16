#include <stdio.h>
#include <cuda_runtime.h>

#define N 9

// Statically defined device memory
__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

__global__ void add() {
    int i = threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main() {
    int a[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int b[N] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    int c[N] = {0};
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Copy data to GPU using cudaMemcpyToSymbol
    cudaMemcpyToSymbol(d_a, a, N * sizeof(int));
    cudaMemcpyToSymbol(d_b, b, N * sizeof(int));
    
    // Record start event
    cudaEventRecord(start);
    
    // Run kernel with 1 block of N threads
    add<<<1, N>>>();
    
    // Record stop event
    cudaEventRecord(stop);
    
    // Check for kernel errors and synchronize
    cudaGetLastError();
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back using cudaMemcpyFromSymbol
    cudaMemcpyFromSymbol(c, d_c, N * sizeof(int));
    
    // Print results
    printf("Results:\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    
    // Print timing information
    printf("\nKernel execution time: %.3f milliseconds\n", milliseconds);
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
