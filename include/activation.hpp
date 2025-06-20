#pragma once

#include "matrix.hpp"
#include <memory>

namespace asekioml {

// Abstract base class for activation functions
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual Matrix forward(const Matrix& input) const = 0;
    virtual Matrix backward(const Matrix& input) const = 0;
    virtual std::unique_ptr<ActivationFunction> clone() const = 0;
    virtual std::string name() const = 0;
};

// Activation function implementations
class ReLU : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
    std::string name() const override { return "ReLU"; }
};

class Sigmoid : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
    std::string name() const override { return "Sigmoid"; }
};

class Tanh : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
    std::string name() const override { return "Tanh"; }
};

class Softmax : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
    std::string name() const override { return "Softmax"; }
};

class LeakyReLU : public ActivationFunction {
private:
    double alpha_;

public:
    explicit LeakyReLU(double alpha = 0.01) : alpha_(alpha) {}
    
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
    std::string name() const override { return "LeakyReLU"; }
};

class Linear : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
    std::unique_ptr<ActivationFunction> clone() const override;
    std::string name() const override { return "Linear"; }
};

// Factory function for creating activation functions
std::unique_ptr<ActivationFunction> create_activation(const std::string& name);

} // namespace asekioml
