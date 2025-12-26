#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__global__ void kernel(const float* A, float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        B[i] = A[i] + B[i];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    kernel<<<gridSize, blockSize>>>(d_A, d_B);

    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Array A: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " ";
    }

    std::cout << std::endl;

    std::cout << "Adding....." << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_B[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}