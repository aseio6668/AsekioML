#include "matrix.hpp"
#include "memory_optimization.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace asekioml {

// Constructors
Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(size_t rows, size_t cols) 
    : data(rows, std::vector<double>(cols, 0.0)), rows_(rows), cols_(cols) {
    if (rows > 0 && cols > 0) {
        memory::MemoryPool::get_instance().track_allocation(rows * cols * sizeof(double));
    }
}

Matrix::Matrix(size_t rows, size_t cols, double value) 
    : data(rows, std::vector<double>(cols, value)), rows_(rows), cols_(cols) {
    if (rows > 0 && cols > 0) {
        memory::MemoryPool::get_instance().track_allocation(rows * cols * sizeof(double));
    }
}

Matrix::Matrix(const std::vector<std::vector<double>>& data) 
    : data(data), rows_(data.size()), cols_(data.empty() ? 0 : data[0].size()) {
    // Validate rectangular matrix
    for (const auto& row : data) {
        if (row.size() != cols_) {
            throw std::invalid_argument("Matrix must be rectangular");
        }
    }
    if (rows_ > 0 && cols_ > 0) {
        memory::MemoryPool::get_instance().track_allocation(rows_ * cols_ * sizeof(double));
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
    rows_ = list.size();
    cols_ = list.size() > 0 ? list.begin()->size() : 0;
    
    data.reserve(rows_);
    for (const auto& row : list) {
        if (row.size() != cols_) {
            throw std::invalid_argument("Matrix must be rectangular");
        }
        data.emplace_back(row);
    }
    if (rows_ > 0 && cols_ > 0) {
        memory::MemoryPool::get_instance().track_allocation(rows_ * cols_ * sizeof(double));
    }
}

// Destructor
Matrix::~Matrix() {
    if (rows_ > 0 && cols_ > 0) {
        memory::MemoryPool::get_instance().track_deallocation(rows_ * cols_ * sizeof(double));
    }
}

// Copy and move constructors
Matrix::Matrix(const Matrix& other) 
    : data(other.data), rows_(other.rows_), cols_(other.cols_) {
    if (rows_ > 0 && cols_ > 0) {
        memory::MemoryPool::get_instance().track_allocation(rows_ * cols_ * sizeof(double));
    }
}

Matrix::Matrix(Matrix&& other) noexcept 
    : data(std::move(other.data)), rows_(other.rows_), cols_(other.cols_) {
    other.rows_ = 0;
    other.cols_ = 0;
}

// Assignment operators
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows_ = other.rows_;
        cols_ = other.cols_;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        rows_ = other.rows_;
        cols_ = other.cols_;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[row][col];
}

std::vector<double>& Matrix::operator[](size_t row) {
    if (row >= rows_) {
        throw std::out_of_range("Matrix row index out of bounds");
    }
    return data[row];
}

const std::vector<double>& Matrix::operator[](size_t row) const {
    if (row >= rows_) {
        throw std::out_of_range("Matrix row index out of bounds");
    }
    return data[row];
}

// Direct data access for SIMD operations
const double* Matrix::data_ptr() const {
    // Note: This assumes continuous memory layout
    // For true SIMD operations, we'd need a different internal storage
    if (rows_ == 0 || cols_ == 0) return nullptr;
    return &data[0][0];
}

double* Matrix::data_ptr() {
    // Note: This assumes continuous memory layout
    // For true SIMD operations, we'd need a different internal storage
    if (rows_ == 0 || cols_ == 0) return nullptr;
    return &data[0][0];
}

// Matrix operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            for (size_t k = 0; k < cols_; ++k) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero");
    }
    return *this * (1.0 / scalar);
}

Matrix& Matrix::operator+=(const Matrix& other) {
    *this = *this + other;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    *this = *this - other;
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (auto& row : data) {
        for (auto& element : row) {
            element *= scalar;
        }
    }
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero");
    }
    return *this *= (1.0 / scalar);
}

// Element-wise operations
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data[i][j] = func(data[i][j]);
        }
    }
    return result;
}

// Matrix functions
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

double Matrix::trace() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Trace only defined for square matrices");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        sum += data[i][i];
    }
    return sum;
}

double Matrix::norm() const {
    double sum = 0.0;
    for (const auto& row : data) {
        for (double element : row) {
            sum += element * element;
        }
    }
    return std::sqrt(sum);
}

// Utility functions
void Matrix::fill(double value) {
    for (auto& row : data) {
        std::fill(row.begin(), row.end(), value);
    }
}

void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    for (auto& row : data) {
        for (auto& element : row) {
            element = dist(gen);
        }
    }
}

void Matrix::xavier_init(size_t fan_in, size_t fan_out) {
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    randomize(-limit, limit);
}

void Matrix::he_init(size_t fan_in) {
    double std_dev = std::sqrt(2.0 / fan_in);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std_dev);
    
    for (auto& row : data) {
        for (auto& element : row) {
            element = dist(gen);
        }
    }
}

// Static factory methods
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result.data[i][i] = 1.0;
    }
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max) {
    Matrix result(rows, cols);
    result.randomize(min, max);
    return result;
}

// I/O
std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "Matrix(" << matrix.rows_ << "x" << matrix.cols_ << "):\n";
    for (size_t i = 0; i < matrix.rows_; ++i) {
        os << "[";
        for (size_t j = 0; j < matrix.cols_; ++j) {
            os << matrix.data[i][j];
            if (j < matrix.cols_ - 1) os << ", ";
        }
        os << "]\n";
    }
    return os;
}

// Statistical functions
double Matrix::mean() const {
    if (rows_ == 0 || cols_ == 0) return 0.0;
    
    double sum = 0.0;
    for (const auto& row : data) {
        sum += std::accumulate(row.begin(), row.end(), 0.0);
    }
    return sum / (rows_ * cols_);
}

double Matrix::variance() const {
    if (rows_ == 0 || cols_ == 0) return 0.0;
    
    double m = mean();
    double sum_sq_diff = 0.0;
    
    for (const auto& row : data) {
        for (double element : row) {
            double diff = element - m;
            sum_sq_diff += diff * diff;
        }
    }
    return sum_sq_diff / (rows_ * cols_);
}

double Matrix::std_dev() const {
    return std::sqrt(variance());
}

Matrix Matrix::sum_rows() const {
    Matrix result(rows_, 1);
    for (size_t i = 0; i < rows_; ++i) {
        result.data[i][0] = std::accumulate(data[i].begin(), data[i].end(), 0.0);
    }
    return result;
}

Matrix Matrix::sum_cols() const {
    Matrix result(1, cols_);
    for (size_t j = 0; j < cols_; ++j) {
        for (size_t i = 0; i < rows_; ++i) {
            result.data[0][j] += data[i][j];
        }
    }
    return result;
}

} // namespace asekioml
