#include "../include/tensor.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept>

namespace asekioml {
namespace ai {

// Private helper methods
void Tensor::calculate_strides() {
    if (shape_.empty()) {
        strides_.clear();
        return;
    }
    
    strides_.resize(shape_.size());
    strides_.back() = 1;
    
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::get_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * strides_[i];
    }
    return index;
}

// Constructors
Tensor::Tensor() : total_size_(0) {}

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    validate_shape(shape);
    total_size_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
    data_.resize(total_size_, 0.0);
    calculate_strides();
}

Tensor::Tensor(const std::vector<size_t>& shape, double fill_value) : shape_(shape) {
    validate_shape(shape);
    total_size_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
    data_.resize(total_size_, fill_value);
    calculate_strides();
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& data) 
    : shape_(shape), data_(data) {
    validate_shape(shape);
    total_size_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
    
    if (data_.size() != total_size_) {
        throw std::invalid_argument("Data size must match tensor size");
    }
    calculate_strides();
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> data_2d) {
    if (data_2d.size() == 0) {
        total_size_ = 0;
        return;
    }
    
    size_t rows = data_2d.size();
    size_t cols = data_2d.begin()->size();
    
    // Validate that all rows have the same number of columns
    for (const auto& row : data_2d) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    
    shape_ = {rows, cols};
    total_size_ = rows * cols;
    data_.reserve(total_size_);
    
    for (const auto& row : data_2d) {
        for (double val : row) {
            data_.push_back(val);
        }
    }
    
    calculate_strides();
}

// Copy and move constructors
Tensor::Tensor(const Tensor& other) 
    : data_(other.data_), shape_(other.shape_), strides_(other.strides_), total_size_(other.total_size_) {}

Tensor::Tensor(Tensor&& other) noexcept 
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)), 
      strides_(std::move(other.strides_)), total_size_(other.total_size_) {
    other.total_size_ = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        total_size_ = other.total_size_;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        total_size_ = other.total_size_;
        other.total_size_ = 0;
    }
    return *this;
}

// Element access
double& Tensor::operator()(const std::vector<size_t>& indices) {
    return data_[get_index(indices)];
}

const double& Tensor::operator()(const std::vector<size_t>& indices) const {
    return data_[get_index(indices)];
}

double& Tensor::operator()(size_t row, size_t col) {
    if (shape_.size() != 2) {
        throw std::invalid_argument("2D indexing only valid for 2D tensors");
    }
    return data_[row * strides_[0] + col * strides_[1]];
}

const double& Tensor::operator()(size_t row, size_t col) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("2D indexing only valid for 2D tensors");
    }
    return data_[row * strides_[0] + col * strides_[1]];
}

double& Tensor::operator[](size_t index) {
    if (index >= total_size_) {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

const double& Tensor::operator[](size_t index) const {
    if (index >= total_size_) {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

// Shape operations
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    validate_shape(new_shape);
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    
    if (new_size != total_size_) {
        throw std::invalid_argument("New shape must have the same total size");
    }
    
    Tensor result(new_shape, data_);
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("transpose() without arguments only valid for 2D tensors");
    }
    return transpose({1, 0});
}

Tensor Tensor::transpose(const std::vector<size_t>& axes) const {
    if (axes.size() != shape_.size()) {
        throw std::invalid_argument("Number of axes must match tensor dimensions");
    }
    
    // Create new shape
    std::vector<size_t> new_shape(shape_.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= shape_.size()) {
            throw std::out_of_range("Axis out of range");
        }
        new_shape[i] = shape_[axes[i]];
    }
    
    Tensor result(new_shape);
    
    // Copy data with permuted indices
    std::vector<size_t> indices(shape_.size(), 0);
    std::vector<size_t> new_indices(shape_.size());
    
    for (size_t i = 0; i < total_size_; ++i) {
        // Convert linear index to multi-dimensional indices
        size_t temp = i;
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            indices[d] = temp % shape_[d];
            temp /= shape_[d];
        }
        
        // Permute indices
        for (size_t j = 0; j < axes.size(); ++j) {
            new_indices[j] = indices[axes[j]];
        }
        
        result(new_indices) = data_[i];
    }
    
    return result;
}

// Element-wise operations
Tensor Tensor::operator+(const Tensor& other) const {
    // Handle same shape case directly
    if (shape_ == other.shape_) {
        Tensor result(shape_);
        for (size_t i = 0; i < total_size_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    
    // Simple broadcasting support
    if (is_broadcastable(shape_, other.shape_)) {
        auto result_shape = broadcast_shapes(shape_, other.shape_);
        Tensor result(result_shape);
        
        // Apply broadcasting logic
        size_t result_size = result.total_size_;
        for (size_t i = 0; i < result_size; ++i) {
            // Get the index for this tensor
            size_t idx1 = get_broadcasted_index(i, result_shape, shape_);
            size_t idx2 = get_broadcasted_index(i, result_shape, other.shape_);
            
            result.data_[i] = data_[idx1] + other.data_[idx2];
        }
        
        return result;
    } else {
        throw std::runtime_error("Incompatible shapes for addition");
    }
}

Tensor Tensor::operator-(const Tensor& other) const {
    // Handle same shape case directly
    if (shape_ == other.shape_) {
        Tensor result(shape_);
        for (size_t i = 0; i < total_size_; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    
    // Simple broadcasting support
    if (is_broadcastable(shape_, other.shape_)) {
        auto result_shape = broadcast_shapes(shape_, other.shape_);
        Tensor result(result_shape);
        
        size_t result_size = result.total_size_;
        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = get_broadcasted_index(i, result_shape, shape_);
            size_t idx2 = get_broadcasted_index(i, result_shape, other.shape_);
            
            result.data_[i] = data_[idx1] - other.data_[idx2];
        }
        
        return result;
    } else {
        throw std::runtime_error("Incompatible shapes for subtraction");
    }
}

Tensor Tensor::operator*(const Tensor& other) const {
    // Handle same shape case directly
    if (shape_ == other.shape_) {
        Tensor result(shape_);
        for (size_t i = 0; i < total_size_; ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }
    
    // Simple broadcasting support
    if (is_broadcastable(shape_, other.shape_)) {
        auto result_shape = broadcast_shapes(shape_, other.shape_);
        Tensor result(result_shape);
        
        size_t result_size = result.total_size_;
        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = get_broadcasted_index(i, result_shape, shape_);
            size_t idx2 = get_broadcasted_index(i, result_shape, other.shape_);
            
            result.data_[i] = data_[idx1] * other.data_[idx2];
        }
        
        return result;
    } else {
        throw std::runtime_error("Incompatible shapes for multiplication");
    }
}

Tensor Tensor::operator/(const Tensor& other) const {
    auto broadcast_shape = broadcast_shapes(shape_, other.shape_);
    Tensor result(broadcast_shape);
    
    if (shape_ == other.shape_) {
        for (size_t i = 0; i < total_size_; ++i) {
            if (std::abs(other.data_[i]) < 1e-12) {
                throw std::runtime_error("Division by zero");
            }
            result.data_[i] = data_[i] / other.data_[i];
        }
    } else {
        throw std::runtime_error("Broadcasting not yet implemented");
    }
    
    return result;
}

// Scalar operations
Tensor Tensor::operator+(double scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(double scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-12) {
        throw std::runtime_error("Division by zero");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

// In-place operations
Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have the same shape for in-place operations");
    }
    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have the same shape for in-place operations");
    }
    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have the same shape for in-place operations");
    }
    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] *= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have the same shape for in-place operations");
    }
    for (size_t i = 0; i < total_size_; ++i) {
        if (std::abs(other.data_[i]) < 1e-12) {
            throw std::runtime_error("Division by zero");
        }
        data_[i] /= other.data_[i];
    }
    return *this;
}

// Reduction operations
Tensor Tensor::sum(int axis, bool keepdim) const {
    if (axis == -1) {
        // Sum all elements
        double total = std::accumulate(data_.begin(), data_.end(), 0.0);
        if (keepdim) {
            std::vector<size_t> result_shape(shape_.size(), 1);
            return Tensor(result_shape, total);
        } else {
            return Tensor({1}, {total});
        }
    } else {
        // Sum along specified axis
        if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
            throw std::out_of_range("Axis out of range");
        }
        
        std::vector<size_t> result_shape = shape_;
        if (keepdim) {
            result_shape[axis] = 1;
        } else {
            result_shape.erase(result_shape.begin() + axis);
        }
        
        Tensor result(result_shape);
        
        // Calculate the result using index iteration
        std::vector<size_t> indices(shape_.size(), 0);
        do {
            std::vector<size_t> result_indices = indices;
            if (!keepdim) {
                result_indices.erase(result_indices.begin() + axis);
            } else {
                result_indices[axis] = 0;
            }
            
            if (result_indices.size() > 0 && result_indices.size() <= result.shape_.size()) {
                // Pad with zeros if needed
                while (result_indices.size() < result.shape_.size()) {
                    result_indices.push_back(0);
                }
                
                // Ensure indices are within bounds
                bool valid = true;
                for (size_t i = 0; i < result_indices.size(); ++i) {
                    if (result_indices[i] >= result.shape_[i]) {
                        valid = false;
                        break;
                    }
                }
                
                if (valid) {
                    result(result_indices) += (*this)(indices);
                }
            }
        } while (increment_indices(indices, shape_));
        
        return result;
    }
}

Tensor Tensor::mean(int axis, bool keepdim) const {
    Tensor sum_result = sum(axis, keepdim);
    
    if (axis == -1) {
        // Mean of all elements
        double count = static_cast<double>(total_size_);
        return sum_result / count;
    } else {
        // Mean along specified axis
        double count = static_cast<double>(shape_[axis]);
        return sum_result / count;
    }
}

Tensor Tensor::max(int axis, bool keepdim) const {
    if (axis == -1) {
        // Max of all elements
        double max_val = *std::max_element(data_.begin(), data_.end());
        if (keepdim) {
            std::vector<size_t> result_shape(shape_.size(), 1);
            return Tensor(result_shape, max_val);
        } else {
            return Tensor({1}, {max_val});
        }
    } else {
        // Max along specified axis
        if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
            throw std::out_of_range("Axis out of range");
        }
        
        std::vector<size_t> result_shape = shape_;
        if (keepdim) {
            result_shape[axis] = 1;
        } else {
            result_shape.erase(result_shape.begin() + axis);
        }
        
        Tensor result(result_shape, -std::numeric_limits<double>::infinity());
        
        std::vector<size_t> indices(shape_.size(), 0);
        do {
            std::vector<size_t> result_indices = indices;
            if (!keepdim) {
                result_indices.erase(result_indices.begin() + axis);
            } else {
                result_indices[axis] = 0;
            }
            
            if (result_indices.size() > 0 && result_indices.size() <= result.shape_.size()) {
                while (result_indices.size() < result.shape_.size()) {
                    result_indices.push_back(0);
                }
                
                bool valid = true;
                for (size_t i = 0; i < result_indices.size(); ++i) {
                    if (result_indices[i] >= result.shape_[i]) {
                        valid = false;
                        break;
                    }
                }
                
                if (valid) {
                    double current_val = (*this)(indices);
                    if (current_val > result(result_indices)) {
                        result(result_indices) = current_val;
                    }
                }
            }
        } while (increment_indices(indices, shape_));
        
        return result;
    }
}

// Slicing operation
Tensor Tensor::slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
    if (ranges.size() != shape_.size()) {
        throw std::invalid_argument("Number of ranges must match tensor dimensions");
    }
    
    std::vector<size_t> result_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].first >= ranges[i].second || ranges[i].second > shape_[i]) {
            throw std::out_of_range("Invalid range for dimension " + std::to_string(i));
        }
        result_shape.push_back(ranges[i].second - ranges[i].first);
    }
    
    Tensor result(result_shape);
    
    std::vector<size_t> result_indices(result_shape.size(), 0);
    do {
        std::vector<size_t> source_indices(shape_.size());
        for (size_t i = 0; i < ranges.size(); ++i) {
            source_indices[i] = ranges[i].first + result_indices[i];
        }
        
        result(result_indices) = (*this)(source_indices);
    } while (increment_indices(result_indices, result_shape));
    
    return result;
}

// Helper methods
void Tensor::validate_shape(const std::vector<size_t>& shape) const {
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }
    }
}

bool Tensor::increment_indices(std::vector<size_t>& indices, const std::vector<size_t>& shape) const {
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        indices[i]++;
        if (indices[i] < shape[i]) {
            return true;
        }
        indices[i] = 0;
    }
    return false;
}

// Global operators
Tensor operator+(double scalar, const Tensor& tensor) {
    return tensor + scalar;
}

Tensor operator-(double scalar, const Tensor& tensor) {
    Tensor result(tensor.shape());
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = scalar - tensor[i];
    }
    return result;
}

Tensor operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor operator/(double scalar, const Tensor& tensor) {
    Tensor result(tensor.shape());
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (std::abs(tensor[i]) < 1e-12) {
            throw std::runtime_error("Division by zero");
        }
        result[i] = scalar / tensor[i];
    }    return result;
}

// Matrix operations
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }
    
    size_t m = shape_[0];
    size_t n = shape_[1];
    size_t p = other.shape_[1];
    
    Tensor result({m, p});
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

Tensor Tensor::dot(const Tensor& other) const {
    return matmul(other);
}

// Factory methods
Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    return Tensor(shape, 0.0);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    return Tensor(shape, 1.0);
}

Tensor Tensor::eye(size_t n) {
    Tensor result({n, n});
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Tensor Tensor::random(const std::vector<size_t>& shape, double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    Tensor result(shape);
    for (size_t i = 0; i < result.total_size_; ++i) {
        result.data_[i] = dist(gen);
    }
    return result;
}

Tensor Tensor::randn(const std::vector<size_t>& shape, double mean, double std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, std);
    
    Tensor result(shape);
    for (size_t i = 0; i < result.total_size_; ++i) {
        result.data_[i] = dist(gen);
    }
    return result;
}

// Instance methods for random initialization
void Tensor::random(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = dist(gen);
    }
}

void Tensor::randn(double mean, double std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, std);
    
    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = dist(gen);
    }
}

// String representation
std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], data=[";
    
    size_t max_elements = 10;
    for (size_t i = 0; i < std::min(total_size_, max_elements); ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(4) << data_[i];
    }
    if (total_size_ > max_elements) {
        oss << ", ...";
    }
    oss << "])";
    return oss.str();
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}

// Helper methods
std::vector<size_t> Tensor::broadcast_shapes(const std::vector<size_t>& shape1, 
                                            const std::vector<size_t>& shape2) const {
    // Handle identical shapes (most common case)
    if (shape1 == shape2) {
        return shape1;
    }
    
    // Numpy-style broadcasting rules:
    // 1. If tensors have different number of dimensions, prepend 1s to the smaller one
    // 2. For each dimension, the size must be either equal or one of them must be 1
    
    std::vector<size_t> s1 = shape1, s2 = shape2;
    
    // Make both shapes the same length by prepending 1s
    while (s1.size() < s2.size()) s1.insert(s1.begin(), 1);
    while (s2.size() < s1.size()) s2.insert(s2.begin(), 1);
    
    std::vector<size_t> result_shape;
    for (size_t i = 0; i < s1.size(); ++i) {
        if (s1[i] == s2[i]) {
            result_shape.push_back(s1[i]);
        } else if (s1[i] == 1) {
            result_shape.push_back(s2[i]);
        } else if (s2[i] == 1) {
            result_shape.push_back(s1[i]);
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }
    }
    
    return result_shape;
}

bool Tensor::is_broadcastable(const std::vector<size_t>& shape1, 
                             const std::vector<size_t>& shape2) const {
    try {
        broadcast_shapes(shape1, shape2);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

size_t Tensor::get_broadcasted_index(size_t linear_index, 
                                    const std::vector<size_t>& result_shape,
                                    const std::vector<size_t>& tensor_shape) const {
    // Convert linear index to multi-dimensional indices in result shape
    std::vector<size_t> result_indices(result_shape.size());
    size_t temp = linear_index;
    for (int i = static_cast<int>(result_shape.size()) - 1; i >= 0; --i) {
        result_indices[i] = temp % result_shape[i];
        temp /= result_shape[i];
    }
    
    // Adjust the shape to match tensor_shape by prepending 1s if needed
    std::vector<size_t> adjusted_shape = tensor_shape;
    while (adjusted_shape.size() < result_shape.size()) {
        adjusted_shape.insert(adjusted_shape.begin(), 1);
    }
    
    // Map result indices to tensor indices according to broadcasting rules
    std::vector<size_t> tensor_indices(adjusted_shape.size());
    for (size_t i = 0; i < adjusted_shape.size(); ++i) {
        if (adjusted_shape[i] == 1) {
            tensor_indices[i] = 0; // Broadcast this dimension
        } else {
            tensor_indices[i] = result_indices[i];
        }
    }
    
    // Remove leading dimensions that were added for broadcasting
    size_t dims_to_remove = adjusted_shape.size() - tensor_shape.size();
    tensor_indices.erase(tensor_indices.begin(), tensor_indices.begin() + dims_to_remove);
    
    // Convert back to linear index using original tensor strides
    return get_index(tensor_indices);
}

// Conversion methods
Matrix Tensor::to_matrix() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Can only convert 2D tensors to matrices");
    }
    
    Matrix result(shape_[0], shape_[1]);
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            result(i, j) = (*this)(i, j);
        }
    }
    return result;
}

Tensor Tensor::from_matrix(const Matrix& matrix) {
    Tensor result({matrix.rows(), matrix.cols()});
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            result(i, j) = matrix(i, j);
        }
    }
    return result;
}

Tensor Tensor::squeeze(int axis) const {
    std::vector<size_t> new_shape;
    
    if (axis == -1) {
        // Remove all dimensions of size 1
        for (size_t s : shape_) {
            if (s != 1) {
                new_shape.push_back(s);
            }
        }
    } else {
        // Remove specific axis if it has size 1
        if (axis < 0) {
            axis = static_cast<int>(shape_.size()) + axis;
        }
        
        if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
            throw std::runtime_error("Axis out of range for squeeze operation");
        }
        
        if (shape_[axis] != 1) {
            throw std::runtime_error("Cannot squeeze axis with size != 1");
        }
        
        for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
            if (i != axis) {
                new_shape.push_back(shape_[i]);
            }
        }
    }
    
    // If all dimensions were squeezed, result should be scalar (empty shape)
    if (new_shape.empty()) {
        new_shape.push_back(1);
    }
    
    Tensor result(new_shape);
    result.data_ = data_; // Copy the data
    return result;
}

Tensor Tensor::unsqueeze(size_t axis) const {
    if (axis > shape_.size()) {
        throw std::runtime_error("Axis out of range for unsqueeze operation");
    }
    
    std::vector<size_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + axis, 1);
    
    Tensor result(new_shape);
    result.data_ = data_; // Copy the data
    return result;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}

} // namespace ai
} // namespace asekioml
