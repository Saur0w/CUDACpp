#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__global__ void kernel(const float* A, const float* B, float* sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        sum[i] = A[i] + B[i];
    }
}

int main() {
    size_t bytes = sizeof(float) * N;

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_sum = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.3f;
        h_B[i] = 2.6f;
        h_sum[i] = 0.0f;
    }

    float *d_A, *d_B, *d_sum;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_sum, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1)/blockSize;

    kernel<<<gridSize, blockSize>>>(d_A, d_B,d_sum);

    cudaDeviceSynchronize();

    cudaMemcpy(h_sum, d_sum, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Array A: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Array B: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_B[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Array sum: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_sum[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_sum);

    free(h_A);
    free(h_B);
    free(h_sum);

    return 0;
}