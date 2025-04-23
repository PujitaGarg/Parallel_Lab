#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM_ITERATIONS 100
#define WARMUP_ITERATIONS 5
#define NUM_SIZES 4  // Now a compile-time constant



__global__ void squareRootKernel(float* A, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = sqrtf(A[i]);
    }
}

void saveToCSV(const char* filename, int* sizes, float* times, int count) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Could not open file %s\n", filename);
        return;
    }
    fprintf(fp, "ArraySize,Time(ms)\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%d,%f\n", sizes[i], times[i]);
    }
    fclose(fp);
}

int main() {
   
    const int sizes[NUM_SIZES] = {50000, 500000, 5000000, 50000000};
    float times[4];  
    int threadsPerBlock = 256;
    
    printf("Benchmarking Square Root Computation\n");
    printf("Array Size\tAvg Time (ms)\n");
    printf("------------------------\n");
    
    for (int i = 0; i < NUM_SIZES; i++) {
        int n = sizes[i];
        size_t bytes = n * sizeof(float);
        
        float *h_A = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);
        
        // Initialize with random values
        for (int j = 0; j < n; j++) {
            h_A[j] = (float)(rand() % 1000 + 1);
        }

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Using GPU: %s\n", prop.name);
        
        float *d_A, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        // Warm-up runs
        for (int iter = 0; iter < WARMUP_ITERATIONS; iter++) {
            squareRootKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, n);
        }
        cudaDeviceSynchronize();
        
        // Timed runs
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            squareRootKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float total_ms;
        cudaEventElapsedTime(&total_ms, start, stop);
        times[i] = total_ms / NUM_ITERATIONS;
        
        printf("%d\t\t%.3f\n", n, times[i]);
        
        // Cleanup
        free(h_A);
        free(h_C);
        cudaFree(d_A);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    saveToCSV("cuda_timings.csv", (int*)sizes, times, NUM_SIZES);
    printf("\nResults saved to cuda_timings.csv\n");
    
    return 0;
}
