#pragma once

#include "matrix.hpp"
#include <vector>
#include <memory>
#include <initializer_list>
#include <functional>

namespace clmodel {
namespace ai {

/**
 * @brief Multi-dimensional tensor class for advanced AI operations
 * 
 * Extends the existing Matrix concept to support N-dimensional arrays
 * while maintaining backward compatibility with the current Matrix API.
 */
class Tensor {
private:
    std::vector<double> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t total_size_;
    
    void calculate_strides();
    size_t get_index(const std::vector<size_t>& indices) const;
    
public:
    // Constructors
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, double fill_value);
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& data);
    Tensor(std::initializer_list<std::initializer_list<double>> data_2d);
    
    // Copy and move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor() = default;
    
    // Element access
    double& operator()(const std::vector<size_t>& indices);
    const double& operator()(const std::vector<size_t>& indices) const;
    
    // For 2D tensors (backward compatibility with Matrix usage)
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    
    // For 1D tensors
    double& operator[](size_t index);
    const double& operator[](size_t index) const;
    
    // Shape operations
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor transpose() const; // For 2D tensors
    Tensor transpose(const std::vector<size_t>& axes) const; // For N-D tensors
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(size_t axis) const;
    Tensor permute(const std::vector<size_t>& dims) const;
    
    // Broadcasting and element-wise operations
    Tensor broadcast_to(const std::vector<size_t>& target_shape) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // Scalar operations
    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    
    // In-place operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    // Reduction operations
    Tensor sum(int axis = -1, bool keepdim = false) const;
    Tensor mean(int axis = -1, bool keepdim = false) const;
    Tensor max(int axis = -1, bool keepdim = false) const;
    Tensor min(int axis = -1, bool keepdim = false) const;
    Tensor std(int axis = -1, bool keepdim = false) const;
    Tensor var(int axis = -1, bool keepdim = false) const;
    
    // Matrix operations (for 2D tensors)
    Tensor matmul(const Tensor& other) const;
    Tensor dot(const Tensor& other) const;
    
    // Utility methods
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return total_size_; }
    size_t ndim() const { return shape_.size(); }
    size_t size(size_t dim) const { return (dim < shape_.size()) ? shape_[dim] : 1; }
    bool is_empty() const { return total_size_ == 0; }
    
    // Data access
    std::vector<double>& data() { return data_; }
    const std::vector<double>& data() const { return data_; }
    double* raw_data() { return data_.data(); }
    const double* raw_data() const { return data_.data(); }
    
    // Conversion methods
    Matrix to_matrix() const;
    static Tensor from_matrix(const Matrix& matrix);
    
    // Factory methods
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor eye(size_t n);
    static Tensor random(const std::vector<size_t>& shape, double min = 0.0, double max = 1.0);
    static Tensor randn(const std::vector<size_t>& shape, double mean = 0.0, double std = 1.0);
    static Tensor arange(double start, double stop, double step = 1.0);
    static Tensor linspace(double start, double stop, size_t num);
    
    // Instance initialization methods
    void random(double min = 0.0, double max = 1.0);
    void randn(double mean = 0.0, double std = 1.0);
    
    // Advanced operations (will be implemented in later phases)
    Tensor conv2d(const Tensor& kernel, size_t stride = 1, size_t padding = 0) const;
    Tensor max_pool2d(size_t kernel_size, size_t stride = 0) const;
    Tensor avg_pool2d(size_t kernel_size, size_t stride = 0) const;
    
    // Slicing and indexing
    Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const;
    Tensor select(size_t dim, size_t index) const;
    
    // String representation
    std::string to_string() const;
    void print() const;
    
    // Comparison operations
    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;
    bool allclose(const Tensor& other, double rtol = 1e-05, double atol = 1e-08) const;
    
private:
    // Helper methods
    std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1, 
                                        const std::vector<size_t>& shape2) const;
    bool is_broadcastable(const std::vector<size_t>& shape1, 
                         const std::vector<size_t>& shape2) const;
    void validate_shape(const std::vector<size_t>& shape) const;
    void validate_indices(const std::vector<size_t>& indices) const;
    bool increment_indices(std::vector<size_t>& indices, const std::vector<size_t>& shape) const;
    size_t get_broadcasted_index(size_t linear_index, 
                               const std::vector<size_t>& result_shape,
                               const std::vector<size_t>& tensor_shape) const;
};

// Global operators for scalar-tensor operations
Tensor operator+(double scalar, const Tensor& tensor);
Tensor operator-(double scalar, const Tensor& tensor);
Tensor operator*(double scalar, const Tensor& tensor);
Tensor operator/(double scalar, const Tensor& tensor);

// Stream operators
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

} // namespace ai
} // namespace clmodel
