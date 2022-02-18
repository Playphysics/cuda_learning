#include <cuda_runtime.h>
#include <stdio.h>

static void print8Data(const float *ptr) {
    for (int i = 0; i < 8; ++i) {
        printf("%10.5f", ptr[i]);
    }
    printf("\n");
}

#define CHECK(funcExpr)                                            \
    do {                                                           \
        const cudaError_t errCode = funcExpr;                      \
        if (errCode != cudaSuccess) {                              \
            printf("\n=== ERROR ====== ERROR ====== ERROR ===\n"); \
            printf("line: %d %s\n", __LINE__, #funcExpr);          \
            printf("%s\n\n", cudaGetErrorString(errCode));         \
        }                                                          \
    } while (0)

#define MAX_NUM 32U

__global__ void ArrayAdd(float *dest, const float *arrA, const float *arrB) {
    const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < MAX_NUM) {
        dest[idx] = arrA[idx] + arrB[idx];
    }
}

int main(int argc, char **argv) {
    float *arrA, *arrB, *arrC;
    const size_t nBytes = sizeof(float) * MAX_NUM;

    CHECK(cudaSetDevice(0));

    CHECK(cudaMallocManaged(&arrA, nBytes));
    CHECK(cudaMallocManaged(&arrB, nBytes));
    CHECK(cudaMallocManaged(&arrC, nBytes));

    for (int i = 0; i < MAX_NUM; ++i) {
        arrA[i] = i * 1.2f;
        arrB[i] = i / 2.0f;
    }

    dim3 block(MAX_NUM, 1), grid(1, 1);

    ArrayAdd<<<grid, block>>>(arrC, arrA, arrB);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    print8Data(&arrA[0]), print8Data(&arrA[8]), putchar('\n');
    print8Data(&arrB[0]), print8Data(&arrB[8]), putchar('\n');
    print8Data(&arrC[0]), print8Data(&arrC[8]), putchar('\n');

    CHECK(cudaFree(arrA));
    CHECK(cudaFree(arrB));
    CHECK(cudaFree(arrC));

    CHECK(cudaDeviceReset());
    return 0;
}
