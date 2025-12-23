#include <cstdio>
#include <cuda.h>

__global__ void kernel() {
    printf("Hello world");
}

int main() {
    kernel<<<1, 1>>>();
    return 0;
}