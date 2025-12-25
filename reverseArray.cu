#include <iostream>
#include <cuda_runtime.h>

#define N 101

__global__ void reverseArray(const int* d_in, int* d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        d_out[N - 1 - i] = d_in[i];
    }
}

int main() {
    size_t bytes = N * sizeof(int);

    int* h_in = new int[N];
    int* h_out = new int[N];

    std::cout << "Input: ";
    for (int i = 0; i < N; i++) {
        h_in[i] = i;
        std::cout << h_in[i] << " ";
    }
    std::cout << "\n";

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverseArray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    std::cout << " \n";
    std::cout << "Output: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    return 0;
}