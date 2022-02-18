#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <windows.h>

// === Test and monitor ====== Test and monitor ====== Test and monitor ===
// === Test and monitor ====== Test and monitor ====== Test and monitor ===
// === Test and monitor ====== Test and monitor ====== Test and monitor ===

#define CHECK(funcExpr)                                            \
    do {                                                           \
        const cudaError_t errCode = funcExpr;                      \
        if (errCode != cudaSuccess) {                              \
            printf("\n=== ERROR ====== ERROR ====== ERROR ===\n"); \
            printf("line: %d %s\n", __LINE__, #funcExpr);          \
            printf("%s\n\n", cudaGetErrorString(errCode));         \
            exit(1);                                               \
        }                                                          \
    } while (0)

static volatile long long lastTime;
static void TimeStart(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    lastTime = t.QuadPart;
}
static void TimeStop(const char preStr[] = "<->") {
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);

    double diffTick = static_cast<double>(t.QuadPart - lastTime);
    printf("The time of %s is %f ms\n", preStr, diffTick * 1000.0 / freq.QuadPart);
}

// === GPU ====== GPU ====== GPU ====== GPU ====== GPU ====== GPU ===
// === GPU ====== GPU ====== GPU ====== GPU ====== GPU ====== GPU ===
// === GPU ====== GPU ====== GPU ====== GPU ====== GPU ====== GPU ===

static constexpr size_t gpuBlockUnit = 8U;
static constexpr size_t GpuMatSizeRoundUp(size_t size) {
    return (size + gpuBlockUnit - 1U) & ~(gpuBlockUnit - 1U);
}

static constexpr unsigned int GpuGetMaxUnit(size_t size) {
    const size_t maxNum = size / gpuBlockUnit;
    // if ((maxNum & 0x0FU) == 0U) return (unsigned int)(gpuBlockUnit << 4U);
    // if ((maxNum & 0x07U) == 0U) return (unsigned int)(gpuBlockUnit << 3U);
    if ((maxNum & 0x03U) == 0U) return (unsigned int)(gpuBlockUnit << 2U);
    if ((maxNum & 0x01U) == 0U) return (unsigned int)(gpuBlockUnit << 1U);
    return (unsigned int)(gpuBlockUnit);
}

template <typename T, size_t row, size_t col, size_t gpuMatRow, size_t gpuMatCol>
static auto GpuCreateMat(const T init[row][col]) -> T (*)[gpuMatCol] {
    T(*retPtr)[gpuMatCol] = nullptr;

    CHECK(cudaMalloc(&retPtr, sizeof(T) * gpuMatRow * gpuMatCol));
    if (init == nullptr) return retPtr;

    for (size_t r = 0U; r < row; ++r) {
        CHECK(cudaMemcpy(retPtr[r], init[r], col * sizeof(T), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(retPtr[r] + col, 0, (gpuMatCol - col) * sizeof(T)));
    }
    for (size_t r = row; r < gpuMatRow; ++r) {
        CHECK(cudaMemset(retPtr[r], 0, gpuMatCol * sizeof(T)));
    }
    return retPtr;
}

template <typename T, size_t row, size_t col, size_t gpuMatCol>
static void GetGpuResult(T dest[row][col], const T gpuData[row][gpuMatCol]) {
    for (size_t r = 0U; r < row; ++r) {
        CHECK(cudaMemcpy(dest[r], gpuData[r], col * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template <typename T, size_t row, size_t col>
__global__ void GpuMatAdd(T dest[row][col], const T matA[row][col], const T matB[row][col]) {
    const size_t x = threadIdx.x + (size_t)blockDim.x * blockIdx.x;
    const size_t y = threadIdx.y + (size_t)blockDim.y * blockIdx.y;
    // dest[x][y] = matA[x][y] + matB[x][y];
    dest[y][x] = matA[y][x] + matB[y][x];
}

template <typename T, size_t row, size_t col, size_t sameDim>
__global__ void GpuMatMul(T dest[row][col], const T matA[row][sameDim], const T (*matB)[col]) {
    const size_t x = threadIdx.x + (size_t)blockDim.x * blockIdx.x;
    const size_t y = threadIdx.y + (size_t)blockDim.y * blockIdx.y;
    T tempResult{};

    for (size_t z = 0U; z < sameDim; ++z) {
        tempResult += matA[x][z] * matB[z][y];
    }
    dest[x][y] = tempResult;
}

template <typename T, size_t row, size_t col>
void MatrixAdd(T dest[row][col], const T matA[row][col], const T matB[row][col]) {
    constexpr size_t gpuMatRow = GpuMatSizeRoundUp(row);
    constexpr size_t gpuMatCol = GpuMatSizeRoundUp(col);

    TimeStart();
    T(*gpuMatA)[gpuMatCol] = GpuCreateMat<T, row, col, gpuMatRow, gpuMatCol>(matA);
    T(*gpuMatB)[gpuMatCol] = GpuCreateMat<T, row, col, gpuMatRow, gpuMatCol>(matB);
    T(*gpuDest)[gpuMatCol] = GpuCreateMat<T, row, col, gpuMatRow, gpuMatCol>(nullptr);
    TimeStop("MatrixAdd GpuCreateMat");

    // const dim3 block(GpuGetMaxUnit(gpuMatRow), GpuGetMaxUnit(gpuMatCol));
    // const dim3 grid(gpuMatRow / block.x, gpuMatCol / block.y);

    const dim3 block(GpuGetMaxUnit(gpuMatCol), GpuGetMaxUnit(gpuMatRow));
    const dim3 grid(gpuMatCol / block.x, gpuMatRow / block.y);

    TimeStart();
    GpuMatAdd<T, gpuMatRow, gpuMatCol><<<grid, block>>>(gpuDest, gpuMatA, gpuMatB);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    TimeStop("MatrixAdd GpuMatAdd");

    TimeStart();
    GetGpuResult<T, row, col, gpuMatCol>(dest, gpuDest);
    TimeStop("MatrixAdd GetGpuResult");

    TimeStart();
    CHECK(cudaFree(static_cast<void *>(gpuMatA)));
    CHECK(cudaFree(static_cast<void *>(gpuMatB)));
    CHECK(cudaFree(static_cast<void *>(gpuDest)));
    TimeStop("MatrixAdd cudaFree");
}

template <typename T, size_t row, size_t col, size_t sameDim>
void MatrixMul(T dest[row][col], const T matA[row][sameDim], const T (*matB)[col]) {
    constexpr size_t gpuMatRow = GpuMatSizeRoundUp(row);
    constexpr size_t gpuMatCol = GpuMatSizeRoundUp(col);

    TimeStart();
    T(*gpuMatA)[sameDim] = GpuCreateMat<T, row, sameDim, gpuMatRow, sameDim>(matA);
    T(*gpuMatB)[gpuMatCol] = GpuCreateMat<T, sameDim, col, sameDim, gpuMatCol>(matB);
    T(*gpuDest)[gpuMatCol] = GpuCreateMat<T, row, col, gpuMatRow, gpuMatCol>(nullptr);
    TimeStop("MatrixMul GpuCreateMat");

    const dim3 block(GpuGetMaxUnit(gpuMatRow), GpuGetMaxUnit(gpuMatCol));
    const dim3 grid(gpuMatRow / block.x, gpuMatCol / block.y);

    TimeStart();
    GpuMatMul<T, gpuMatRow, gpuMatCol><<<grid, block>>>(gpuDest, gpuMatA, gpuMatB);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    TimeStop("MatrixMul GpuMatMul");

    TimeStart();
    GetGpuResult<T, row, col, gpuMatCol>(dest, gpuDest);
    TimeStop("MatrixMul GetGpuResult");

    TimeStart();
    CHECK(cudaFree(static_cast<void *>(gpuMatA)));
    CHECK(cudaFree(static_cast<void *>(gpuMatB)));
    CHECK(cudaFree(static_cast<void *>(gpuDest)));
    TimeStop("MatrixMul cudaFree");
}

// === CPU ====== CPU ====== CPU ====== CPU ====== CPU ====== CPU ===
// === CPU ====== CPU ====== CPU ====== CPU ====== CPU ====== CPU ===
// === CPU ====== CPU ====== CPU ====== CPU ====== CPU ====== CPU ===

template <typename T, size_t row, size_t col>
void Cpu_MatrixAdd(T dest[row][col], const T matA[row][col], const T matB[row][col]) {
    for (size_t i = 0U; i < row; ++i) {
        for (size_t j = 0U; j < col; ++j) {
            dest[i][j] = matA[i][j] + matB[i][j];
        }
    }
}
template <typename T, size_t row, size_t col, size_t sameDim>
void Cpu_MatrixMul(T dest[row][col], const T matA[row][sameDim], const T (*matB)[col]) {
    for (size_t i = 0U; i < row; ++i) {
        for (size_t j = 0U; j < col; ++j) {
            T tempResult{};
            for (size_t k = 0U; k < sameDim; ++k) {
                tempResult += matA[i][k] * matB[k][j];
            }
            dest[i][j] = tempResult;
        }
    }
}

// === TEST ====== TEST ====== TEST ====== TEST ====== TEST ====== TEST ===
// === TEST ====== TEST ====== TEST ====== TEST ====== TEST ====== TEST ===
// === TEST ====== TEST ====== TEST ====== TEST ====== TEST ====== TEST ===

// Test class, only for init and delete
template <typename T, size_t row, size_t col>
struct Matrix {
    T (*ptr)[col];

    Matrix() {
        // printf("Create matrix\n");
        ptr = new T[row][col];
        for (size_t i = 0U; i < row; ++i) {
            for (size_t j = 0U; j < col; ++j) {
                ptr[i][j] = static_cast<T>((i + 1U) * 100U + j + 1U);
            }
        }
    }
    ~Matrix() {
        // printf("Destroy matrix\n");
        delete[] ptr;
    }
};

template <typename T, size_t row, size_t col>
static void PrtMat(const T mat[row][col]) {
    for (size_t i = 0U; i < row; ++i) {
        for (size_t j = 0U; j < col; ++j) {
            std::cout << std::setw(6) << mat[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    constexpr size_t row = 4000U;
    constexpr size_t col = row * 3 / 2;
    constexpr size_t sameDim = col * 3 / 2;

    CHECK(cudaSetDevice(0));

    printf("\nPre test\n");
    {
        constexpr size_t N = 10U;
        int A[N][N], B[N][N], C[N][N];
        MatrixAdd<int, N>(C, A, B);
    }
    printf("\nAdd test\n");
    {
        typedef unsigned int Elem_T;
        Matrix<Elem_T, row, col> matA, matB, matC, matC_cpu;

        MatrixAdd<Elem_T, row>(matC.ptr, matA.ptr, matB.ptr);
        // PrtMat<Elem_T, row>(matC.ptr);

        TimeStart();
        Cpu_MatrixAdd<Elem_T, row>(matC_cpu.ptr, matA.ptr, matB.ptr);
        TimeStop();
        // PrtMat<Elem_T, row>(matC_cpu.ptr);
    }
    printf("\nMul test\n");
    if (row * col < 2000000U) {
        typedef double Elem_T;
        Matrix<Elem_T, row, sameDim> matD;
        Matrix<Elem_T, sameDim, col> matE;
        Matrix<Elem_T, row, col> matF, matF_cpu;

        MatrixMul<Elem_T, row>(matF.ptr, matD.ptr, matE.ptr);
        // PrtMat<Elem_T, row>(matF.ptr);

        TimeStart();
        Cpu_MatrixMul<Elem_T, row>(matF_cpu.ptr, matD.ptr, matE.ptr);
        TimeStop();
        // PrtMat<Elem_T, row>(matF_cpu.ptr);
    }
    CHECK(cudaDeviceReset());
    return 0;
}
