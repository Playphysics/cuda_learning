
#include <ctime>
#include "matrixGpu.hpp"
#include "matrixCpu.hpp"

namespace {

volatile long lastTime;
void TimeStart(void) {
    lastTime = clock();
}
void TimeStop(const char preStr[] = "<->") {
    clock_t t = clock();

    const double diffTick = static_cast<double>(t - lastTime);
    const double diffMs = diffTick * 1000.0 / CLOCKS_PER_SEC;
    printf("The time of %s is %f ms\n", preStr, diffMs);
}

// Test class, only for init and delete
template <typename T, size_t row, size_t col>
struct TestMatrix {
    T (*ptr)[col];

    TestMatrix() {
        // printf("Create matrix\n");
        ptr = new T[row][col];
        for (size_t i = 0U; i < row; ++i) {
            for (size_t j = 0U; j < col; ++j) {
                ptr[i][j] = static_cast<T>((i + 1U) * 100U + j);
            }
        }
    }
    T *GetPtr() {
        return reinterpret_cast<T *>(ptr);
    }
    ~TestMatrix() {
        // printf("Destroy matrix\n");
        delete[] ptr;
    }
};

template <typename Elem_T, size_t row, size_t col>
void TestAdd() {
    TestMatrix<Elem_T, row, col> matA, matB;

    MatCpuOp::Matrix<Elem_T> cpuMatA(row, col, matA.GetPtr());
    MatCpuOp::Matrix<Elem_T> cpuMatB(row, col, matB.GetPtr());

    MatGpuOp::Matrix<Elem_T> gpuMatA(row, col, matA.GetPtr());
    MatGpuOp::Matrix<Elem_T> gpuMatB(row, col, matB.GetPtr());

    TimeStart();
    cpuMatA += cpuMatB;
    TimeStop("Cpu_MatrixAdd(a += b)");
    cpuMatA.Print();

    TimeStart();
    gpuMatA += gpuMatB;
    TimeStop("Gpu_MatrixAdd(a += b)");
    gpuMatA.Print();

    // TimeStart();
    // MatCpuOp::Matrix<Elem_T> cpuMatC = cpuMatA + cpuMatB;
    // TimeStop("Cpu_MatrixAdd(c = a + b)");
    // cpuMatC.Print();

    // TimeStart();
    // MatGpuOp::Matrix<Elem_T> gpuMatC = gpuMatA + gpuMatB;
    // TimeStop("Gpu_MatrixAdd(c = a + b)");
    // gpuMatC.Print();
}

template <typename Elem_T, size_t row, size_t col, size_t sameDim>
void TestMul() {
    TestMatrix<Elem_T, row, sameDim> matA;
    TestMatrix<Elem_T, sameDim, col> matB;

    MatCpuOp::Matrix<Elem_T> cpuMatA(row, sameDim, matA.GetPtr());
    MatCpuOp::Matrix<Elem_T> cpuMatB(sameDim, col, matB.GetPtr());

    MatGpuOp::Matrix<Elem_T> gpuMatA(row, sameDim, matA.GetPtr());
    MatGpuOp::Matrix<Elem_T> gpuMatB(sameDim, col, matB.GetPtr());

    TimeStart();
    MatCpuOp::Matrix<Elem_T> cpuMatC = cpuMatA * cpuMatB;
    TimeStop("Cpu_MatrixMul");
    cpuMatC.Print(12);

    TimeStart();
    MatGpuOp::Matrix<Elem_T> gpuMatC = gpuMatA * gpuMatB;
    TimeStop("Gpu_MatrixMul");
    gpuMatC.Print(12);
}

}  // namespace

// nvcc -O2 -arch=native .\matrixOperation.cu -o .\matrixOperation.exe

int main() {
    constexpr size_t row = 4567U;
    constexpr size_t col = 4321U;
    constexpr size_t sameDim = 123U;

    CudaCheckError(cudaSetDevice(0));

    std::cout << "\nTest add" << std::endl;
    TestAdd<int, row, col>();

    std::cout << "\nTest mul" << std::endl;
    TestMul<double, row, col, sameDim>();

    CudaCheckError(cudaDeviceReset());
    return 0;
}
