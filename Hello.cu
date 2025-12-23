#include <cstdio>
#include <cuda_runtime.h>

#define N 1000

__global__ void dkernel() {
    int sum = 0;
    for (int i = 1; i <= N; i++) {
        sum += i;
    }
    printf("Sum = %d\n", sum);
}

int main() {
    dkernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}