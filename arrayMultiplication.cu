#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorMultiply(const int *A, const int *B, int *C) {
    int i = threadIdx.x;

    C[i] = A[i] * B[i];
}

int main() {
    const int N = 5;
    int h_A[N] = {1, 2, 3, 4, 5};
    int h_B[N] = {10, 20, 30, 40, 50};
    int h_C[N] = {0, 0, 0, 0};

    int *d_A = nullptr;
    int *d_B = nullptr;
    int *d_C = nullptr;

    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    vectorMultiply<<<1, N>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "C = { ";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << ((i == N - 1) ? " " : ", ");
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}