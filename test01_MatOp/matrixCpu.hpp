#ifndef MATRIX_CPU_HPP_
#define MATRIX_CPU_HPP_

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>

namespace MatCpuOp {

// === declaration ====== declaration ====== declaration ===
// === declaration ====== declaration ====== declaration ===

template <typename T>
class Matrix final {
   private:
    size_t row, col;
    T *ptr;

   public:
    Matrix(size_t x = 0U, size_t y = 0U, const T *initData = nullptr)
        : row(x), col(y), ptr(nullptr) {
        const size_t size = row * col * sizeof(T);

        if (size != 0U) ptr = static_cast<T *>(malloc(size));
        if (initData != nullptr) memcpy(ptr, initData, size);
    }
    Matrix(const Matrix &mat) : Matrix(mat.row, mat.col, mat.ptr) {}
    Matrix(Matrix &&mat) noexcept {
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
        ret += mat;
        return ret;
    }
    Matrix operator-(const Matrix &mat) const {
        Matrix ret(*this);
        ret -= mat;
        return ret;
    }
    Matrix operator*(const Matrix &mat) const;

    void PrintLine(size_t row, int width = 8) const;
    void Print(int width = 8) const;

    ~Matrix() {
        if (ptr != nullptr) free(ptr);
    }
};

// === definition ====== definition ====== definition ===
// === definition ====== definition ====== definition ===

template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &mat) {
    if (row != mat.row || col != mat.col) return *this;

    for (size_t i = 0U; i < row; ++i) {
        for (size_t j = 0U; j < col; ++j) {
            const size_t idx = i * col + j;
            this->ptr[idx] += mat.ptr[idx];
        }
    }
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &mat) {
    if (row != mat.row || col != mat.col) return *this;

    for (size_t i = 0U; i < row; ++i) {
        for (size_t j = 0U; j < col; ++j) {
            const size_t idx = i * col + j;
            this->ptr[idx] -= mat.ptr[idx];
        }
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &mat) const {
    if (this->col != mat.row) return Matrix();
    Matrix retMat(this->row, mat.col);
    memset(retMat.ptr, 0, retMat.row * retMat.col * sizeof(T));

    for (size_t k = 0U; k < this->col; ++k) {
        for (size_t i = 0U; i < this->row; ++i) {
            const T srcA = this->ptr[i * this->col + k];

            for (size_t j = 0U; j < mat.col; ++j) {
                retMat.ptr[i * mat.col + j] += srcA * mat.ptr[k * mat.col + j];
            }
        }
    }
    return retMat;
}

template <typename T>
void Matrix<T>::PrintLine(size_t row, int width) const {
    if (row >= this->row) return;
    const T *lineArr = ptr + row * this->col;

    if (this->col <= 8U) {
        for (size_t i = 0U; i < this->col; ++i) {
            std::cout << std::setw(width) << lineArr[i] << ' ';
        }
        std::cout << std::endl;
    } else {
        for (size_t i = 0U; i < 4; ++i) {
            std::cout << std::setw(width) << lineArr[i] << ' ';
        }
        std::cout << "...";

        for (size_t i = this->col - 4U; i < this->col; ++i) {
            std::cout << std::setw(width) << lineArr[i] << ' ';
        }
        std::cout << std::endl;
    }
}

template <typename T>
void Matrix<T>::Print(int width) const {
    constexpr size_t maxRow = 4U;
    if (row <= maxRow) {
        for (size_t i = 0U; i < row; ++i) {
            PrintLine(i, width);
        }
    } else {
        for (size_t i = 0U; i < maxRow / 2U; ++i) {
            PrintLine(i, width);
        }
        std::cout << "    ... ... ... ..." << std::endl;
        for (size_t i = row - maxRow / 2U; i < row; ++i) {
            PrintLine(i, width);
        }
    }
    std::cout << std::endl;
}

}  // namespace MatCpuOp

#endif
