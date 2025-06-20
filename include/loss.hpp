#pragma once

#include "matrix.hpp"
#include <memory>

namespace asekioml {

// Abstract base class for loss functions
class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual double compute_loss(const Matrix& predictions, const Matrix& targets) const = 0;
    virtual Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const = 0;
    virtual std::unique_ptr<LossFunction> clone() const = 0;
    virtual std::string name() const = 0;
};

// Mean Squared Error Loss
class MeanSquaredError : public LossFunction {
public:
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override;
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::unique_ptr<LossFunction> clone() const override;
    std::string name() const override { return "MeanSquaredError"; }
};

// Cross Entropy Loss (for classification)
class CrossEntropyLoss : public LossFunction {
public:
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override;
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::unique_ptr<LossFunction> clone() const override;
    std::string name() const override { return "CrossEntropyLoss"; }
};

// Binary Cross Entropy Loss
class BinaryCrossEntropyLoss : public LossFunction {
public:
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override;
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::unique_ptr<LossFunction> clone() const override;
    std::string name() const override { return "BinaryCrossEntropyLoss"; }
};

// Mean Absolute Error Loss
class MeanAbsoluteError : public LossFunction {
public:
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override;
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::unique_ptr<LossFunction> clone() const override;
    std::string name() const override { return "MeanAbsoluteError"; }
};

// Huber Loss (combination of MSE and MAE)
class HuberLoss : public LossFunction {
private:
    double delta_;

public:
    explicit HuberLoss(double delta = 1.0) : delta_(delta) {}
    
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override;
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::unique_ptr<LossFunction> clone() const override;
    std::string name() const override { return "HuberLoss"; }
};

// Factory function for creating loss functions
std::unique_ptr<LossFunction> create_loss_function(const std::string& name);

} // namespace asekioml
