#pragma once

#include "matrix.hpp"
#include <immintrin.h>
#include <thread>
#include <future>
#include <execution>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace asekioml {
namespace simd {

// SIMD-optimized matrix operations
class SIMDMatrix : public Matrix {
public:
    using Matrix::Matrix;
      // Constructor to convert from Matrix
    SIMDMatrix(const Matrix& other) : Matrix(other) {}
    
    // SIMD-optimized matrix multiplication
    Matrix multiply_simd(const Matrix& other) const;
    
    // Parallel matrix operations
    Matrix multiply_parallel(const Matrix& other) const;
    
    // AVX2-optimized element-wise operations
    Matrix add_simd(const Matrix& other) const;
    Matrix hadamard_simd(const Matrix& other) const;
    
    // SIMD activation functions
    Matrix relu_simd() const;
    Matrix sigmoid_simd() const;
    Matrix tanh_simd() const;
    
private:
    static constexpr size_t SIMD_WIDTH = 8; // AVX2 processes 8 doubles
    
    // Helper functions for SIMD operations
    void multiply_block_simd(const Matrix& A, const Matrix& B, Matrix& C,
                            size_t i_start, size_t i_end,
                            size_t j_start, size_t j_end,
                            size_t k_start, size_t k_end) const;
};

// Cache-friendly matrix multiplication with blocking
class BlockedMatrix : public Matrix {
public:
    using Matrix::Matrix;
    
    Matrix multiply_blocked(const Matrix& other, size_t block_size = 64) const;
    
private:
    static constexpr size_t DEFAULT_BLOCK_SIZE = 64;
};

// Memory pool for efficient allocation
class MatrixPool {
private:
    static constexpr size_t POOL_SIZE = 1024 * 1024; // 1MB chunks
    std::vector<std::unique_ptr<double[]>> pools_;
    std::vector<size_t> available_sizes_;
    std::mutex pool_mutex_;
    
public:
    static MatrixPool& instance();
    
    double* allocate(size_t size);
    void deallocate(double* ptr, size_t size);
    
    // RAII wrapper for pool-allocated matrices
    class PooledMatrix {
        double* data_;
        size_t size_;
        size_t rows_;
        size_t cols_;
        
    public:
        PooledMatrix(size_t rows, size_t cols);
        ~PooledMatrix();
        
        double& operator()(size_t row, size_t col);
        const double& operator()(size_t row, size_t col) const;
        
        Matrix to_matrix() const;
    };
};

} // namespace simd
} // namespace asekioml
