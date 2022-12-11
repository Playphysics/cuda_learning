#ifndef MATRIX_GPU_HPP_
#define MATRIX_GPU_HPP_

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
// #include <stdexcept>

#define CudaCheckError(funcExpr) PrintError(funcExpr, __LINE__, #funcExpr)
void PrintError(cudaError_t errCode, int line, const char *expression) {
    if (errCode != cudaSuccess) {
        printf("\n=== ERROR ====== ERROR ===\n");
        printf("line: %d %s => ", line, expression);
        printf("%s\n\n", cudaGetErrorString(errCode));
        exit(1);
        // throw std::runtime_error(cudaGetErrorString(errCode));
    }
}

namespace MatDeviceKernal {

template <typename T>
__global__ void MatIAdd(T *dest, const T *src, size_t col) {
    const size_t y = threadIdx.x + (size_t)blockDim.x * blockIdx.x;
    const size_t x = threadIdx.y + (size_t)blockDim.y * blockIdx.y;
    dest[x * col + y] += src[x * col + y];
}

template <typename T>
__global__ void MatISub(T *dest, const T *src, size_t col) {
    const size_t y = threadIdx.x + (size_t)blockDim.x * blockIdx.x;
    const size_t x = threadIdx.y + (size_t)blockDim.y * blockIdx.y;
    dest[x * col + y] -= src[x * col + y];
}

struct Coor {
    size_t x, y;
};

template <typename T>
__global__ void MatMul(T *dest, Coor offset, const T *srcA, const T *srcB, Coor sizeB) {
    size_t destX = threadIdx.x + (size_t)blockDim.x * blockIdx.x;
    size_t destY = threadIdx.y + (size_t)blockDim.y * blockIdx.y;

    destX += offset.x, destY += offset.y;

    T tempValue = srcA[destX * sizeB.x + 0U] * srcB[0U * sizeB.y + destY];

    for (size_t z = 1U; z < sizeB.x; ++z) {
        tempValue += srcA[destX * sizeB.x + z] * srcB[z * sizeB.y + destY];
    }
    dest[destX * sizeB.y + destY] = tempValue;
}

}  // namespace MatDeviceKernal

namespace MatGpuOp {

constexpr size_t GpuMinSize = 8U;
constexpr size_t GpuSizeMask = GpuMinSize - 1U;

// === declaration ====== declaration ====== declaration ===
// === declaration ====== declaration ====== declaration ===

template <typename T>
class Matrix final {
   private:
    size_t row, col;
    T *ptr;

    size_t GetGpuCol() const {
        return (col + GpuSizeMask) & ~GpuSizeMask;
    }
    void AllocGpuMemory() {
        // std::cout << "GpuMat alloc memory" << std::endl;
        const size_t gpuRow = (row + GpuSizeMask) & ~GpuSizeMask;
        const size_t gpuCol = (col + GpuSizeMask) & ~GpuSizeMask;
        CudaCheckError(cudaMalloc(&ptr, gpuRow * gpuCol * sizeof(T)));
    }
    void InitGpuData(const T *src, size_t srcCol, cudaMemcpyKind kind) const {
        T *dest = ptr;
        for (size_t r = 0U; r < row; ++r) {
            CudaCheckError(cudaMemcpy(dest, src, col * sizeof(T), kind));
            dest += GetGpuCol(), src += srcCol;
        }
    }

   public:
    Matrix() : row(0U), col(0U), ptr(nullptr) {}
    Matrix(size_t x, size_t y, const T *initData = nullptr) : row(x), col(y) {
        // std::cout << "GpuMat construct from value" << std::endl;
        AllocGpuMemory();
        if (initData == nullptr || ptr == nullptr) return;
        InitGpuData(initData, col, cudaMemcpyHostToDevice);
    }
    Matrix(const Matrix &mat) : row(mat.row), col(mat.col) {
        // std::cout << "GpuMat construct from copy" << std::endl;
        AllocGpuMemory();
        if (mat.ptr == nullptr || ptr == nullptr) return;
        InitGpuData(mat.ptr, GetGpuCol(), cudaMemcpyDeviceToDevice);
    }
    Matrix(Matrix &&mat) noexcept {
        // std::cout << "GpuMat construct from move" << std::endl;
        row = mat.row, col = mat.col, ptr = mat.ptr;
        mat.ptr = nullptr;
    }

    inline void swap(Matrix &mat1, Matrix &mat2) {
        const size_t tempX = mat1.row, tempY = mat1.col;
        T *const tempPtr = mat1.ptr;

        mat1.row = mat2.row, mat2.row = tempX;
        mat1.col = mat2.col, mat2.col = tempY;
        mat1.ptr = mat2.ptr, mat2.ptr = tempPtr;
    }
    Matrix &operator=(Matrix mat) {
        swap(*this, mat);
        return *this;
    }

    Matrix &operator+=(const Matrix &mat);
    Matrix &operator-=(const Matrix &mat);

    Matrix operator+(const Matrix &mat) const {
        Matrix ret(*this);
        // return ret += mat; this cause two copy_construction
        ret += mat;
        return ret;
    }
    Matrix operator-(const Matrix &mat) const {
        Matrix ret(*this);
        ret -= mat;
        return ret;
    }
    Matrix operator*(const Matrix &mat) const;

    void GetLine(T *destPtr, size_t matRow, size_t startCol = 0U, size_t elemNum = 1U) const {
        if (matRow >= row || startCol + elemNum > col) return;

        const T *const src = ptr + matRow * GetGpuCol() + startCol;
        CudaCheckError(cudaMemcpy(destPtr, src, elemNum * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void SetLine(const T *srcPtr, size_t matRow, size_t startCol = 0U, size_t elemNum = 1U) const {
        if (matRow >= row || startCol + elemNum > col) return;

        T *const dest = ptr + matRow * GetGpuCol() + startCol;
        CudaCheckError(cudaMemcpy(dest, srcPtr, elemNum * sizeof(T), cudaMemcpyHostToDevice));
    }

    void PrintLine(size_t row, int width = 8) const;
    void Print(int width = 8) const;

    ~Matrix() {
        // std::cout << "GpuMat destruct" << std::endl;
        if (ptr != nullptr) CudaCheckError(cudaFree(ptr));
    }
};

// === definition ====== definition ====== definition ===
// === definition ====== definition ====== definition ===

template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &mat) {
    if (row != mat.row || col != mat.col) return *this;

    const dim3 block(GpuMinSize, GpuMinSize);
    // const dim3 grid((row + GpuSizeMask) / block.x, (col + GpuSizeMask) / block.y);
    const dim3 grid((col + GpuSizeMask) / block.y, (row + GpuSizeMask) / block.x);

    MatDeviceKernal::MatIAdd<T><<<grid, block>>>(this->ptr, mat.ptr, GetGpuCol());
    CudaCheckError(cudaGetLastError());
    CudaCheckError(cudaDeviceSynchronize());
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &mat) {
    if (row != mat.row || col != mat.col) return *this;

    const dim3 block(GpuMinSize, GpuMinSize);
    const dim3 grid((col + GpuSizeMask) / block.y, (row + GpuSizeMask) / block.x);

    MatDeviceKernal::MatISub<T><<<grid, block>>>(this->ptr, mat.ptr, GetGpuCol());
    CudaCheckError(cudaGetLastError());
    CudaCheckError(cudaDeviceSynchronize());

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &mat) const {
    if (this->col != mat.row) return Matrix();
    Matrix retMat(this->row, mat.col);

    const dim3 block(GpuMinSize, GpuMinSize);
    const MatDeviceKernal::Coor sizeofMatB{GetGpuCol(), mat.GetGpuCol()};

    const auto GpuFunc = MatDeviceKernal::MatMul<T>;
    constexpr size_t baseBlockSize = 1024U;

    MatDeviceKernal::Coor offset{0U, 0U};

    for (offset.x = 0U; offset.x < this->row; offset.x += baseBlockSize) {
        const size_t height = std::min(this->row - offset.x, baseBlockSize) + GpuSizeMask;

        for (offset.y = 0U; offset.y < mat.col; offset.y += baseBlockSize) {
            const size_t width = std::min(mat.col - offset.y, baseBlockSize) + GpuSizeMask;
            const dim3 grid(height / block.x, width / block.y);

            GpuFunc<<<grid, block>>>(retMat.ptr, offset, this->ptr, mat.ptr, sizeofMatB);
            CudaCheckError(cudaGetLastError());
            CudaCheckError(cudaDeviceSynchronize());
        }
    }
    return retMat;
}

template <typename T>
void Matrix<T>::PrintLine(size_t row, int width) const {
    if (row >= this->row) return;
    T tempBuffer[16];

    if (this->col <= 8U) {
        GetLine(tempBuffer, row, 0U, this->col);
        for (size_t i = 0U; i < this->col; ++i) {
            std::cout << std::setw(width) << tempBuffer[i] << ' ';
        }
        std::cout << std::endl;
    } else {
        GetLine(tempBuffer, row, 0U, 4U);
        for (size_t i = 0U; i < 4; ++i) {
            std::cout << std::setw(width) << tempBuffer[i] << ' ';
        }
        std::cout << "...";

        GetLine(tempBuffer, row, this->col - 4, 4U);
        for (size_t i = 0U; i < 4U; ++i) {
            std::cout << std::setw(width) << tempBuffer[i] << ' ';
        }
        std::cout << std::endl;
    }
}

template <typename T>
void Matrix<T>::Print(int width) const {
    constexpr size_t maxRow = 4U;
    if (this->row <= maxRow) {
        for (size_t i = 0U; i < this->row; ++i) {
            PrintLine(i, width);
        }
    } else {
        for (size_t i = 0U; i < maxRow / 2U; ++i) {
            PrintLine(i, width);
        }
        std::cout << "    ... ... ... ..." << std::endl;

        for (size_t i = this->row - maxRow / 2U; i < this->row; ++i) {
            PrintLine(i, width);
        }
    }
    std::cout << std::endl;
}

}  // namespace MatGpuOp

#endif
