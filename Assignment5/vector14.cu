#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20) // 1 million elements

__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

__global__ void add() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main() {
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Calculate theoretical bandwidth
    double theoreticalBW = (prop.memoryClockRate * 1e-6 * prop.memoryBusWidth * 2) / 8;
    
    printf("Device: %s\n", prop.name);
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n\n", theoreticalBW);

    // Initialize host arrays
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy data to GPU
    cudaMemcpyToSymbol(d_a, a, N * sizeof(int));
    cudaMemcpyToSymbol(d_b, b, N * sizeof(int));

    // Set up timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEventRecord(start);
    add<<<blocksPerGrid, threadsPerBlock>>>();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back
    cudaMemcpyFromSymbol(c, d_c, N * sizeof(int));
    
    // Calculate bandwidth
    size_t RBytes = N * sizeof(int) * 2;
    size_t WBytes = N * sizeof(int);
    double measuredBW = (RBytes + WBytes) / (milliseconds * 1e6); // Convert to GB/s
    
    printf("Array size: %d elements (%zu MB)\n", N, (RBytes + WBytes) >> 20);
    printf("Kernel time: %.3f ms\n", milliseconds);
    printf("Measured Bandwidth: %.2f GB/s\n", measuredBW);
    printf("Percentage of Theoretical: %.1f%%\n", (measuredBW/theoreticalBW)*100);
    
    // Cleanup
    delete[] a; delete[] b; delete[] c;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
