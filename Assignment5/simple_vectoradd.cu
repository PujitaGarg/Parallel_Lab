#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int N = 9;
    int a[N] = {1, 2, 3, 4, 5,6,7,8,9};
    int b[N] = {10, 20, 30, 40, 50,60,70,80,90};
    int c[N] = {0};
    
    int *d_a, *d_b, *d_c;
    
    // Allocate GPU memory
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Run kernel with 1 block of N threads
    add<<<1, N>>>(d_a, d_b, d_c);
    
    // Check for kernel errors and synchronize
    cudaGetLastError();
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Results:\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
