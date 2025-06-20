#pragma once

#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <stdexcept>
#include <initializer_list>

namespace asekioml {

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows_;
    size_t cols_;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double value);
    Matrix(const std::vector<std::vector<double>>& data);
    Matrix(std::initializer_list<std::initializer_list<double>> list);

    // Destructor
    ~Matrix();

    // Copy and move constructors
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;

    // Assignment operators
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;    // Getters
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    // Direct data access for SIMD operations
    const double* data_ptr() const;
    double* data_ptr();
    const std::vector<std::vector<double>>& get_data() const { return data; }

    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    std::vector<double>& operator[](size_t row);
    const std::vector<double>& operator[](size_t row) const;

    // Matrix operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    // Element-wise operations
    Matrix hadamard(const Matrix& other) const; // Element-wise multiplication
    Matrix apply(std::function<double(double)> func) const;

    // Matrix functions
    Matrix transpose() const;
    Matrix inverse() const;
    double determinant() const;
    double trace() const;
    double norm() const;

    // Utility functions
    void fill(double value);
    void randomize(double min = -1.0, double max = 1.0);
    void xavier_init(size_t fan_in, size_t fan_out);
    void he_init(size_t fan_in);

    // Static factory methods
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t size);
    static Matrix random(size_t rows, size_t cols, double min = -1.0, double max = 1.0);

    // I/O
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

    // Reshape and slicing
    Matrix reshape(size_t new_rows, size_t new_cols) const;
    Matrix slice(size_t start_row, size_t end_row, size_t start_col, size_t end_col) const;

    // Statistical functions
    double mean() const;
    double variance() const;
    double std_dev() const;
    Matrix sum_rows() const; // Sum along rows (returns column vector)
    Matrix sum_cols() const; // Sum along columns (returns row vector)
};

} // namespace asekioml
