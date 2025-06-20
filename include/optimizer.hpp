#pragma once

#include "matrix.hpp"
#include <memory>
#include <vector>

namespace clmodel {

// Abstract base class for optimizers
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(Matrix& weights, const Matrix& gradients) = 0;
    virtual void update_bias(Matrix& biases, const Matrix& gradients) = 0;
    virtual std::unique_ptr<Optimizer> clone() const = 0;
    virtual std::string name() const = 0;
    virtual void reset() {} // Reset internal state if needed
    
    // Learning rate control (default implementations)
    virtual void set_learning_rate(double lr) = 0;
    virtual double get_learning_rate() const = 0;
};

// Stochastic Gradient Descent
class SGD : public Optimizer {
private:
    double learning_rate_;
    double momentum_;
    Matrix velocity_weights_;
    Matrix velocity_bias_;
    bool momentum_initialized_;

public:
    explicit SGD(double learning_rate = 0.01, double momentum = 0.0);
    
    void update(Matrix& weights, const Matrix& gradients) override;
    void update_bias(Matrix& biases, const Matrix& gradients) override;
    std::unique_ptr<Optimizer> clone() const override;
    std::string name() const override { return "SGD"; }
    void reset() override;
    
    double get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(double lr) override { learning_rate_ = lr; }
};

// Adam Optimizer
class Adam : public Optimizer {
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    size_t t_; // Time step
    
    Matrix m_weights_; // First moment estimate for weights
    Matrix v_weights_; // Second moment estimate for weights
    Matrix m_bias_;    // First moment estimate for bias
    Matrix v_bias_;    // Second moment estimate for bias
    bool initialized_;

public:
    explicit Adam(double learning_rate = 0.001, double beta1 = 0.9, 
                  double beta2 = 0.999, double epsilon = 1e-8);
    
    void update(Matrix& weights, const Matrix& gradients) override;
    void update_bias(Matrix& biases, const Matrix& gradients) override;
    std::unique_ptr<Optimizer> clone() const override;
    std::string name() const override { return "Adam"; }
    void reset() override;
    
    void set_learning_rate(double lr) override { learning_rate_ = lr; }
    double get_learning_rate() const override { return learning_rate_; }
};

// RMSprop Optimizer
class RMSprop : public Optimizer {
private:
    double learning_rate_;
    double decay_rate_;
    double epsilon_;
    
    Matrix cache_weights_;
    Matrix cache_bias_;
    bool initialized_;

public:
    explicit RMSprop(double learning_rate = 0.001, double decay_rate = 0.9, 
                     double epsilon = 1e-8);
    
    void update(Matrix& weights, const Matrix& gradients) override;
    void update_bias(Matrix& biases, const Matrix& gradients) override;
    std::unique_ptr<Optimizer> clone() const override;
    std::string name() const override { return "RMSprop"; }
    void reset() override;
    
    double get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(double lr) override { learning_rate_ = lr; }
};

// AdaGrad Optimizer
class AdaGrad : public Optimizer {
private:
    double learning_rate_;
    double epsilon_;
    
    Matrix cache_weights_;
    Matrix cache_bias_;
    bool initialized_;

public:
    explicit AdaGrad(double learning_rate = 0.01, double epsilon = 1e-8);
    
    void update(Matrix& weights, const Matrix& gradients) override;
    void update_bias(Matrix& biases, const Matrix& gradients) override;
    std::unique_ptr<Optimizer> clone() const override;
    std::string name() const override { return "AdaGrad"; }
    void reset() override;
    
    double get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(double lr) override { learning_rate_ = lr; }
};

// Factory function for creating optimizers
std::unique_ptr<Optimizer> create_optimizer(const std::string& name, double learning_rate = 0.01);

} // namespace clmodel
