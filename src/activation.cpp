#include "activation.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace clmodel {

// ReLU Implementation
Matrix ReLU::forward(const Matrix& input) const {
    return input.apply([](double x) { return std::max(0.0, x); });
}

Matrix ReLU::backward(const Matrix& input) const {
    return input.apply([](double x) { return x > 0.0 ? 1.0 : 0.0; });
}

std::unique_ptr<ActivationFunction> ReLU::clone() const {
    return std::make_unique<ReLU>();
}

// Sigmoid Implementation
Matrix Sigmoid::forward(const Matrix& input) const {
    return input.apply([](double x) { 
        // Prevent overflow
        x = std::max(-500.0, std::min(500.0, x));
        return 1.0 / (1.0 + std::exp(-x)); 
    });
}

Matrix Sigmoid::backward(const Matrix& input) const {
    Matrix sigmoid_output = forward(input);
    return sigmoid_output.hadamard(sigmoid_output.apply([](double x) { return 1.0 - x; }));
}

std::unique_ptr<ActivationFunction> Sigmoid::clone() const {
    return std::make_unique<Sigmoid>();
}

// Tanh Implementation
Matrix Tanh::forward(const Matrix& input) const {
    return input.apply([](double x) { 
        // Prevent overflow
        x = std::max(-500.0, std::min(500.0, x));
        return std::tanh(x); 
    });
}

Matrix Tanh::backward(const Matrix& input) const {
    Matrix tanh_output = forward(input);
    return tanh_output.apply([](double x) { return 1.0 - x * x; });
}

std::unique_ptr<ActivationFunction> Tanh::clone() const {
    return std::make_unique<Tanh>();
}

// Softmax Implementation
Matrix Softmax::forward(const Matrix& input) const {
    Matrix result(input.rows(), input.cols());
    
    for (size_t i = 0; i < input.rows(); ++i) {
        // Find max for numerical stability
        double max_val = input[i][0];
        for (size_t j = 1; j < input.cols(); ++j) {
            max_val = std::max(max_val, input[i][j]);
        }
        
        // Compute exponentials
        double sum = 0.0;
        for (size_t j = 0; j < input.cols(); ++j) {
            double exp_val = std::exp(input[i][j] - max_val);
            result[i][j] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (size_t j = 0; j < input.cols(); ++j) {
            result[i][j] /= sum;
        }
    }
    
    return result;
}

Matrix Softmax::backward(const Matrix& input) const {
    Matrix softmax_output = forward(input);
    Matrix result(input.rows(), input.cols());
    
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            double s_j = softmax_output[i][j];
            
            // Jacobian of softmax: s_j * (1 - s_j) for diagonal elements
            // For simplicity, we return the diagonal of the Jacobian
            result[i][j] = s_j * (1.0 - s_j);
        }
    }
    
    return result;
}

std::unique_ptr<ActivationFunction> Softmax::clone() const {
    return std::make_unique<Softmax>();
}

// LeakyReLU Implementation
Matrix LeakyReLU::forward(const Matrix& input) const {
    return input.apply([this](double x) { return x > 0.0 ? x : alpha_ * x; });
}

Matrix LeakyReLU::backward(const Matrix& input) const {
    return input.apply([this](double x) { return x > 0.0 ? 1.0 : alpha_; });
}

std::unique_ptr<ActivationFunction> LeakyReLU::clone() const {
    return std::make_unique<LeakyReLU>(alpha_);
}

// Linear Implementation
Matrix Linear::forward(const Matrix& input) const {
    return input; // Identity function
}

Matrix Linear::backward(const Matrix& input) const {
    return Matrix::ones(input.rows(), input.cols()); // Derivative is 1
}

std::unique_ptr<ActivationFunction> Linear::clone() const {
    return std::make_unique<Linear>();
}

// Factory function
std::unique_ptr<ActivationFunction> create_activation(const std::string& name) {
    if (name == "relu" || name == "ReLU") {
        return std::make_unique<ReLU>();
    } else if (name == "sigmoid" || name == "Sigmoid") {
        return std::make_unique<Sigmoid>();
    } else if (name == "tanh" || name == "Tanh") {
        return std::make_unique<Tanh>();
    } else if (name == "softmax" || name == "Softmax") {
        return std::make_unique<Softmax>();
    } else if (name == "leaky_relu" || name == "LeakyReLU") {
        return std::make_unique<LeakyReLU>();
    } else if (name == "linear" || name == "Linear") {
        return std::make_unique<Linear>();
    } else {
        throw std::invalid_argument("Unknown activation function: " + name);
    }
}

} // namespace clmodel
