#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__global__ void kernel(const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < N) {
        int i = row * N + col;
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t size = N * N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Launching kernel with Grid(%d, %d) and Block(%d, %d)\n",
        numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            printf("Error at index %d: %f\n", i, h_C[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Sucess! All values are 3.0\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}