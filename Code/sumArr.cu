#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <windows.h>
#include <time.h>

#define CHECK(funcExpr)                                            \
    do {                                                           \
        const cudaError_t errCode = funcExpr;                      \
        if (errCode != cudaSuccess) {                              \
            printf("\n=== ERROR ====== ERROR ====== ERROR ===\n"); \
            printf("line: %d %s\n", __LINE__, #funcExpr);          \
            printf("%s\n\n", cudaGetErrorString(errCode));         \
        }                                                          \
    } while (0)

#define BLOCK_SIZE (64 * 256)
#define NUM_PER_THREAD 4096

extern "C" {
__declspec(dllexport) void OpenDevice(void);
__declspec(dllexport) void CloseDevice(void);
__declspec(dllexport) double GetGpuResult(void);
__declspec(dllexport) double GetCpuResult(void);
}

void OpenDevice(void) {
    CHECK(cudaSetDevice(0));
}
void CloseDevice(void) {
    CHECK(cudaDeviceReset());
}

__global__ void GpuFunc(double retArr[]) {
    double ret = 0.0;
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int i = idx * NUM_PER_THREAD + 1;
    const int maxi = i + NUM_PER_THREAD;

    for (; i < maxi; ++i) {
        ret += sin(cos(tan(1.0 / i)));
    }
    retArr[idx] = ret;
}

double GetGpuResult(void) {
    double ret = 0.0, retArr[BLOCK_SIZE];
    dim3 grid(BLOCK_SIZE / 64, 1), block(64, 1);

    double *gpuResult;
    CHECK(cudaMalloc(&gpuResult, sizeof(retArr)));

    GpuFunc<<<grid, block>>>(gpuResult);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(retArr, gpuResult, sizeof(retArr), cudaMemcpyDeviceToHost));

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        ret += retArr[i];
    }
    return ret;
}

double GetCpuResult(void) {
    double ret = 0.0;
    for (int i = 1; i <= BLOCK_SIZE * NUM_PER_THREAD; ++i) {
        ret += sin(cos(tan(1.0 / i)));
    }
    return ret;
}

int main() {
    LARGE_INTEGER freq, t0, t1;
    volatile double ret, dt1, dt2;
    OpenDevice();

    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&t0);
    ret = GetGpuResult();
    QueryPerformanceCounter(&t1);

    dt1 = 1000.0 * (t1.QuadPart - t0.QuadPart) / freq.QuadPart;
    printf("ret: %f, time: %f ms\n", ret, dt1);

    QueryPerformanceCounter(&t0);
    ret = GetCpuResult();
    QueryPerformanceCounter(&t1);

    dt2 = 1000.0 * (t1.QuadPart - t0.QuadPart) / freq.QuadPart;
    printf("ret: %f, time: %f ms\n", ret, dt2);

    CloseDevice();
    return 0;
}
