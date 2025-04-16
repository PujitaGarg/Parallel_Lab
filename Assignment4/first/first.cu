#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define NUM_THREADS_PER_BLOCK 256
#define SHOW_ELEMENTS 10 // Number of elements to show from start and end

// Kernel function for iterative sum
__global__ void iterativeSumKernel(int *input, int *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Only first thread does the iterative sum
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += input[i];
        }
        output[0] = sum;
    }
}

// Kernel function for formula-based sum
__global__ void formulaSumKernel(int *input, int *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Only second thread does the formula sum
    if (tid == 1) {
        output[1] = (N * (N + 1)) / 2;
    }
}

void printArrayElements(int *array) {
    printf("Array elements (first and last %d elements):\n", SHOW_ELEMENTS);
    
    // Print first few elements
    printf("First %d elements: ", SHOW_ELEMENTS);
    for (int i = 0; i < SHOW_ELEMENTS; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
    
    // Print last few elements
    printf("Last %d elements: ", SHOW_ELEMENTS);
    for (int i = N - SHOW_ELEMENTS; i < N; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    int *h_input, *h_output;
    int *d_input, *d_output;
    
    // Allocate host memory
    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc(2 * sizeof(int)); // Two outputs: [0] for iterative, [1] for formula
    
    // Initialize input array with first N integers
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
    }
    
    // Print array elements
    printArrayElements(h_input);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, 2 * sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Define block and grid sizes
    dim3 blockSize(NUM_THREADS_PER_BLOCK);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    // Launch kernels
    iterativeSumKernel<<<gridSize, blockSize>>>(d_input, d_output);
    formulaSumKernel<<<gridSize, blockSize>>>(d_input, d_output);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("\nSum using iterative approach: %d\n", h_output[0]);
    printf("Sum using formula: %d\n", h_output[1]);
    
    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
