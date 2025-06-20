#include "simd_matrix.hpp"
#include <algorithm>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace asekioml {
namespace simd {

// Simple SIMD-optimized matrix multiplication (basic implementation)
Matrix SIMDMatrix::multiply_simd(const Matrix& other) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    Matrix result(rows(), other.cols());
    
    // Simple blocked multiplication for better cache performance
    constexpr size_t BLOCK_SIZE = 64;
    
    for (size_t i = 0; i < rows(); i += BLOCK_SIZE) {
        for (size_t j = 0; j < other.cols(); j += BLOCK_SIZE) {
            for (size_t k = 0; k < cols(); k += BLOCK_SIZE) {
                size_t i_end = std::min(i + BLOCK_SIZE, rows());
                size_t j_end = std::min(j + BLOCK_SIZE, other.cols());
                size_t k_end = std::min(k + BLOCK_SIZE, cols());
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; ++jj) {
                        double sum = result(ii, jj);
                        for (size_t kk = k; kk < k_end; ++kk) {
                            sum += (*this)(ii, kk) * other(kk, jj);
                        }
                        result(ii, jj) = sum;
                    }
                }
            }
        }
    }
    
    return result;
}

// Simple parallel matrix multiplication
Matrix SIMDMatrix::multiply_parallel(const Matrix& other) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    Matrix result(rows(), other.cols());
    
    // Simple row-wise parallelization
    #pragma omp parallel for if(rows() > 64)
    for (int i = 0; i < static_cast<int>(rows()); ++i) {
        for (size_t j = 0; j < other.cols(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols(); ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

// Simple element-wise operations
Matrix SIMDMatrix::add_simd(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    
    Matrix result(rows(), cols());
    
    #pragma omp parallel for if(rows() * cols() > 1024)
    for (int i = 0; i < static_cast<int>(rows()); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    
    return result;
}

Matrix SIMDMatrix::hadamard_simd(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
    }
    
    Matrix result(rows(), cols());
    
    #pragma omp parallel for if(rows() * cols() > 1024)
    for (int i = 0; i < static_cast<int>(rows()); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result(i, j) = (*this)(i, j) * other(i, j);
        }
    }
    
    return result;
}

Matrix SIMDMatrix::relu_simd() const {
    Matrix result(rows(), cols());
    
    #pragma omp parallel for if(rows() * cols() > 1024)
    for (int i = 0; i < static_cast<int>(rows()); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result(i, j) = std::max(0.0, (*this)(i, j));
        }
    }
    
    return result;
}

Matrix SIMDMatrix::sigmoid_simd() const {
    Matrix result(rows(), cols());
    
    #pragma omp parallel for if(rows() * cols() > 1024)
    for (int i = 0; i < static_cast<int>(rows()); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result(i, j) = 1.0 / (1.0 + std::exp(-(*this)(i, j)));
        }
    }
    
    return result;
}

Matrix SIMDMatrix::tanh_simd() const {
    Matrix result(rows(), cols());
    
    #pragma omp parallel for if(rows() * cols() > 1024)
    for (int i = 0; i < static_cast<int>(rows()); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result(i, j) = std::tanh((*this)(i, j));
        }
    }
    
    return result;
}

// Blocked matrix multiplication implementation
Matrix BlockedMatrix::multiply_blocked(const Matrix& other, size_t block_size) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    Matrix result(rows(), other.cols());
    
    for (size_t i = 0; i < rows(); i += block_size) {
        for (size_t j = 0; j < other.cols(); j += block_size) {
            for (size_t k = 0; k < cols(); k += block_size) {
                size_t i_end = std::min(i + block_size, rows());
                size_t j_end = std::min(j + block_size, other.cols());
                size_t k_end = std::min(k + block_size, cols());
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; ++jj) {
                        double sum = result(ii, jj);
                        for (size_t kk = k; kk < k_end; ++kk) {
                            sum += (*this)(ii, kk) * other(kk, jj);
                        }
                        result(ii, jj) = sum;
                    }
                }
            }
        }
    }
    
    return result;
}

// Memory pool implementation (simplified)
MatrixPool& MatrixPool::instance() {
    static MatrixPool pool;
    return pool;
}

double* MatrixPool::allocate(size_t size) {
    // Simplified implementation - just use regular allocation
    return new double[size];
}

void MatrixPool::deallocate(double* ptr, size_t) {
    delete[] ptr;
}

// Pooled matrix implementation
MatrixPool::PooledMatrix::PooledMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), size_(rows * cols) {
    data_ = MatrixPool::instance().allocate(size_);
}

MatrixPool::PooledMatrix::~PooledMatrix() {
    if (data_) {
        MatrixPool::instance().deallocate(data_, size_);
    }
}

double& MatrixPool::PooledMatrix::operator()(size_t row, size_t col) {
    return data_[row * cols_ + col];
}

const double& MatrixPool::PooledMatrix::operator()(size_t row, size_t col) const {
    return data_[row * cols_ + col];
}

Matrix MatrixPool::PooledMatrix::to_matrix() const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) = (*this)(i, j);
        }
    }
    return result;
}

} // namespace simd
} // namespace asekioml
