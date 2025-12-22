#include <cstdio>
#include <cuda_runtime.h>

#define N 10

__global__ void kernel(int *A) {
    int i = threadIdx.x;
    if (i < N) {
        if (i % 2 == 0) {
            A[i] *= 2;
        } else {
            A[i] /= 2;
        }
    }
}

int main() {
    int h_A[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int* d_A = nullptr;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, N>>>(d_A);
    cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_A[i]);
    }
    printf("\n");
    cudaFree(d_A);
    return 0;
}