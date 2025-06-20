#include "layer.hpp"
#include <random>
#include <cmath>
#include <sstream>
#include <fstream>

namespace clmodel {

// DenseLayer Implementation
DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : input_size_(input_size), output_size_(output_size),
      weights_(output_size, input_size), biases_(output_size, 1),
      weight_gradients_(output_size, input_size), bias_gradients_(output_size, 1) {
    initialize_xavier();
}

Matrix DenseLayer::forward(const Matrix& input) {
    if (input.cols() != input_size_) {
        throw std::invalid_argument("Input size mismatch in DenseLayer");
    }
    
    last_input_ = input;
    
    // Compute: output = weights * input + biases
    Matrix output = weights_ * input.transpose();
    
    // Add biases to each column
    for (size_t i = 0; i < output.cols(); ++i) {
        for (size_t j = 0; j < output.rows(); ++j) {
            output[j][i] += biases_[j][0];
        }
    }
    
    return output.transpose();
}

Matrix DenseLayer::backward(const Matrix& gradient) {
    if (gradient.cols() != output_size_) {
        throw std::invalid_argument("Gradient size mismatch in DenseLayer");
    }
      // Compute weight gradients: dW = gradient^T * last_input  
    weight_gradients_ = gradient.transpose() * last_input_;
    
    // Compute bias gradients: db = sum of gradients across batch (rows)
    // bias_gradients should have same shape as biases: [output_size, 1]
    bias_gradients_ = Matrix(output_size_, 1);
    for (size_t i = 0; i < output_size_; ++i) {
        double bias_grad = 0.0;
        for (size_t j = 0; j < gradient.rows(); ++j) {
            bias_grad += gradient[j][i];
        }
        bias_gradients_[i][0] = bias_grad;
    }    // Compute input gradients: dx = gradient * weights
    Matrix input_gradient = gradient * weights_;
    
    return input_gradient;
}

void DenseLayer::update_weights(double learning_rate) {
    weights_ -= weight_gradients_ * learning_rate;
    biases_ -= bias_gradients_ * learning_rate;
}

std::unique_ptr<Layer> DenseLayer::clone() const {
    auto cloned = std::make_unique<DenseLayer>(input_size_, output_size_);
    cloned->weights_ = weights_;
    cloned->biases_ = biases_;
    return cloned;
}

void DenseLayer::initialize_xavier() {
    weights_.xavier_init(input_size_, output_size_);
    biases_.fill(0.0);
}

void DenseLayer::initialize_he() {
    weights_.he_init(input_size_);
    biases_.fill(0.0);
}

void DenseLayer::initialize_random(double min, double max) {
    weights_.randomize(min, max);
    biases_.randomize(min, max);
}

std::string DenseLayer::serialize_to_json() const {
    std::ostringstream ss;
    ss << "{";
    ss << "\"type\":\"Dense\",";
    ss << "\"input_size\":" << input_size_ << ",";
    ss << "\"output_size\":" << output_size_;
    ss << "}";
    return ss.str();
}

void DenseLayer::serialize_weights(std::ofstream& file) const {
    // Write weights matrix
    uint32_t rows = static_cast<uint32_t>(weights_.rows());
    uint32_t cols = static_cast<uint32_t>(weights_.cols());
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    for (size_t i = 0; i < weights_.rows(); ++i) {
        for (size_t j = 0; j < weights_.cols(); ++j) {
            double value = weights_(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
    
    // Write biases matrix
    rows = static_cast<uint32_t>(biases_.rows());
    cols = static_cast<uint32_t>(biases_.cols());
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    for (size_t i = 0; i < biases_.rows(); ++i) {
        for (size_t j = 0; j < biases_.cols(); ++j) {
            double value = biases_(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
}

void DenseLayer::deserialize_weights(std::ifstream& file) {
    // Read weights matrix
    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    weights_ = Matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            weights_(i, j) = value;
        }
    }
    
    // Read biases matrix
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    biases_ = Matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            biases_(i, j) = value;
        }
    }
    
    // Reinitialize gradient matrices
    weight_gradients_ = Matrix(weights_.rows(), weights_.cols());
    bias_gradients_ = Matrix(biases_.rows(), biases_.cols());
}

size_t DenseLayer::get_weights_size() const {
    return weights_.rows() * weights_.cols() + biases_.rows() * biases_.cols();
}

// ActivationLayer Implementation
ActivationLayer::ActivationLayer(std::unique_ptr<ActivationFunction> activation_func, size_t size)
    : activation_func_(std::move(activation_func)), size_(size) {}

ActivationLayer::ActivationLayer(const std::string& activation_name, size_t size)
    : activation_func_(create_activation(activation_name)), size_(size) {}

ActivationLayer::ActivationLayer(const ActivationLayer& other)
    : activation_func_(other.activation_func_->clone()), 
      last_input_(other.last_input_), size_(other.size_) {}

ActivationLayer& ActivationLayer::operator=(const ActivationLayer& other) {
    if (this != &other) {
        activation_func_ = other.activation_func_->clone();
        last_input_ = other.last_input_;
        size_ = other.size_;
    }
    return *this;
}

Matrix ActivationLayer::forward(const Matrix& input) {
    if (input.cols() != size_) {
        throw std::invalid_argument("Input size mismatch in ActivationLayer");
    }
    
    last_input_ = input;
    return activation_func_->forward(input);
}

Matrix ActivationLayer::backward(const Matrix& gradient) {
    if (gradient.cols() != size_) {
        throw std::invalid_argument("Gradient size mismatch in ActivationLayer");
    }
    
    Matrix activation_derivative = activation_func_->backward(last_input_);
    return gradient.hadamard(activation_derivative);
}

std::unique_ptr<Layer> ActivationLayer::clone() const {
    return std::make_unique<ActivationLayer>(*this);
}

std::string ActivationLayer::serialize_to_json() const {
    std::ostringstream ss;
    ss << "{";
    ss << "\"type\":\"Activation\",";
    ss << "\"activation\":\"" << activation_func_->name() << "\",";
    ss << "\"size\":" << size_;
    ss << "}";
    return ss.str();
}

void ActivationLayer::serialize_weights(std::ofstream& file) const {
    // Activation layers have no weights to serialize
}

void ActivationLayer::deserialize_weights(std::ifstream& file) {
    // Activation layers have no weights to deserialize
}

size_t ActivationLayer::get_weights_size() const {
    return 0; // No weights in activation layers
}

// DropoutLayer is now implemented in regularization.cpp

} // namespace clmodel
