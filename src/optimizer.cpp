#include "optimizer.hpp"
#include <cmath>
#include <stdexcept>

namespace asekioml {

// SGD Implementation
SGD::SGD(double learning_rate, double momentum)
    : learning_rate_(learning_rate), momentum_(momentum), momentum_initialized_(false) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (momentum < 0.0 || momentum >= 1.0) {
        throw std::invalid_argument("Momentum must be in [0, 1)");
    }
}

void SGD::update(Matrix& weights, const Matrix& gradients) {
    if (momentum_ > 0.0) {
        if (!momentum_initialized_ || velocity_weights_.rows() != weights.rows() || 
            velocity_weights_.cols() != weights.cols()) {
            velocity_weights_ = Matrix::zeros(weights.rows(), weights.cols());
            momentum_initialized_ = true;
        }
        
        velocity_weights_ = velocity_weights_ * momentum_ + gradients * learning_rate_;
        weights -= velocity_weights_;
    } else {
        weights -= gradients * learning_rate_;
    }
}

void SGD::update_bias(Matrix& biases, const Matrix& gradients) {
    if (momentum_ > 0.0) {
        if (!momentum_initialized_ || velocity_bias_.rows() != biases.rows() || 
            velocity_bias_.cols() != biases.cols()) {
            velocity_bias_ = Matrix::zeros(biases.rows(), biases.cols());
        }
        
        velocity_bias_ = velocity_bias_ * momentum_ + gradients * learning_rate_;
        biases -= velocity_bias_;
    } else {
        biases -= gradients * learning_rate_;
    }
}

std::unique_ptr<Optimizer> SGD::clone() const {
    return std::make_unique<SGD>(learning_rate_, momentum_);
}

void SGD::reset() {
    momentum_initialized_ = false;
    velocity_weights_ = Matrix();
    velocity_bias_ = Matrix();
}

// Adam Implementation
Adam::Adam(double learning_rate, double beta1, double beta2, double epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), 
      epsilon_(epsilon), t_(0), initialized_(false) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("Beta1 must be in [0, 1)");
    }
    if (beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("Beta2 must be in [0, 1)");
    }
}

void Adam::update(Matrix& weights, const Matrix& gradients) {
    if (!initialized_ || m_weights_.rows() != weights.rows() || 
        m_weights_.cols() != weights.cols()) {
        m_weights_ = Matrix::zeros(weights.rows(), weights.cols());
        v_weights_ = Matrix::zeros(weights.rows(), weights.cols());
        initialized_ = true;
    }
    
    t_++;
    
    // Update biased first moment estimate
    m_weights_ = m_weights_ * beta1_ + gradients * (1.0 - beta1_);
    
    // Update biased second raw moment estimate
    Matrix gradients_squared = gradients.hadamard(gradients);
    v_weights_ = v_weights_ * beta2_ + gradients_squared * (1.0 - beta2_);
    
    // Compute bias-corrected first moment estimate
    double bias_correction1 = 1.0 - std::pow(beta1_, t_);
    Matrix m_hat = m_weights_ * (1.0 / bias_correction1);
    
    // Compute bias-corrected second raw moment estimate
    double bias_correction2 = 1.0 - std::pow(beta2_, t_);
    Matrix v_hat = v_weights_ * (1.0 / bias_correction2);
    
    // Update weights
    Matrix denominator = v_hat.apply([this](double x) { return std::sqrt(x) + epsilon_; });
    Matrix update = m_hat.hadamard(denominator.apply([](double x) { return 1.0 / x; }));
    weights -= update * learning_rate_;
}

void Adam::update_bias(Matrix& biases, const Matrix& gradients) {
    if (!initialized_ || m_bias_.rows() != biases.rows() || 
        m_bias_.cols() != biases.cols()) {
        m_bias_ = Matrix::zeros(biases.rows(), biases.cols());
        v_bias_ = Matrix::zeros(biases.rows(), biases.cols());
    }
    
    // Update biased first moment estimate
    m_bias_ = m_bias_ * beta1_ + gradients * (1.0 - beta1_);
    
    // Update biased second raw moment estimate
    Matrix gradients_squared = gradients.hadamard(gradients);
    v_bias_ = v_bias_ * beta2_ + gradients_squared * (1.0 - beta2_);
    
    // Compute bias-corrected first moment estimate
    double bias_correction1 = 1.0 - std::pow(beta1_, t_);
    Matrix m_hat = m_bias_ * (1.0 / bias_correction1);
    
    // Compute bias-corrected second raw moment estimate
    double bias_correction2 = 1.0 - std::pow(beta2_, t_);
    Matrix v_hat = v_bias_ * (1.0 / bias_correction2);
    
    // Update biases
    Matrix denominator = v_hat.apply([this](double x) { return std::sqrt(x) + epsilon_; });
    Matrix update = m_hat.hadamard(denominator.apply([](double x) { return 1.0 / x; }));
    biases -= update * learning_rate_;
}

std::unique_ptr<Optimizer> Adam::clone() const {
    return std::make_unique<Adam>(learning_rate_, beta1_, beta2_, epsilon_);
}

void Adam::reset() {
    t_ = 0;
    initialized_ = false;
    m_weights_ = Matrix();
    v_weights_ = Matrix();
    m_bias_ = Matrix();
    v_bias_ = Matrix();
}

// RMSprop Implementation
RMSprop::RMSprop(double learning_rate, double decay_rate, double epsilon)
    : learning_rate_(learning_rate), decay_rate_(decay_rate), 
      epsilon_(epsilon), initialized_(false) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (decay_rate < 0.0 || decay_rate >= 1.0) {
        throw std::invalid_argument("Decay rate must be in [0, 1)");
    }
}

void RMSprop::update(Matrix& weights, const Matrix& gradients) {
    if (!initialized_ || cache_weights_.rows() != weights.rows() || 
        cache_weights_.cols() != weights.cols()) {
        cache_weights_ = Matrix::zeros(weights.rows(), weights.cols());
        initialized_ = true;
    }
    
    // Update cache
    Matrix gradients_squared = gradients.hadamard(gradients);
    cache_weights_ = cache_weights_ * decay_rate_ + gradients_squared * (1.0 - decay_rate_);
    
    // Update weights
    Matrix denominator = cache_weights_.apply([this](double x) { return std::sqrt(x) + epsilon_; });
    Matrix update = gradients.hadamard(denominator.apply([](double x) { return 1.0 / x; }));
    weights -= update * learning_rate_;
}

void RMSprop::update_bias(Matrix& biases, const Matrix& gradients) {
    if (!initialized_ || cache_bias_.rows() != biases.rows() || 
        cache_bias_.cols() != biases.cols()) {
        cache_bias_ = Matrix::zeros(biases.rows(), biases.cols());
    }
    
    // Update cache
    Matrix gradients_squared = gradients.hadamard(gradients);
    cache_bias_ = cache_bias_ * decay_rate_ + gradients_squared * (1.0 - decay_rate_);
    
    // Update biases
    Matrix denominator = cache_bias_.apply([this](double x) { return std::sqrt(x) + epsilon_; });
    Matrix update = gradients.hadamard(denominator.apply([](double x) { return 1.0 / x; }));
    biases -= update * learning_rate_;
}

std::unique_ptr<Optimizer> RMSprop::clone() const {
    return std::make_unique<RMSprop>(learning_rate_, decay_rate_, epsilon_);
}

void RMSprop::reset() {
    initialized_ = false;
    cache_weights_ = Matrix();
    cache_bias_ = Matrix();
}

// AdaGrad Implementation
AdaGrad::AdaGrad(double learning_rate, double epsilon)
    : learning_rate_(learning_rate), epsilon_(epsilon), initialized_(false) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
}

void AdaGrad::update(Matrix& weights, const Matrix& gradients) {
    if (!initialized_ || cache_weights_.rows() != weights.rows() || 
        cache_weights_.cols() != weights.cols()) {
        cache_weights_ = Matrix::zeros(weights.rows(), weights.cols());
        initialized_ = true;
    }
    
    // Update cache (accumulate squared gradients)
    Matrix gradients_squared = gradients.hadamard(gradients);
    cache_weights_ += gradients_squared;
    
    // Update weights
    Matrix denominator = cache_weights_.apply([this](double x) { return std::sqrt(x) + epsilon_; });
    Matrix update = gradients.hadamard(denominator.apply([](double x) { return 1.0 / x; }));
    weights -= update * learning_rate_;
}

void AdaGrad::update_bias(Matrix& biases, const Matrix& gradients) {
    if (!initialized_ || cache_bias_.rows() != biases.rows() || 
        cache_bias_.cols() != biases.cols()) {
        cache_bias_ = Matrix::zeros(biases.rows(), biases.cols());
    }
    
    // Update cache (accumulate squared gradients)
    Matrix gradients_squared = gradients.hadamard(gradients);
    cache_bias_ += gradients_squared;
    
    // Update biases
    Matrix denominator = cache_bias_.apply([this](double x) { return std::sqrt(x) + epsilon_; });
    Matrix update = gradients.hadamard(denominator.apply([](double x) { return 1.0 / x; }));
    biases -= update * learning_rate_;
}

std::unique_ptr<Optimizer> AdaGrad::clone() const {
    return std::make_unique<AdaGrad>(learning_rate_, epsilon_);
}

void AdaGrad::reset() {
    initialized_ = false;
    cache_weights_ = Matrix();
    cache_bias_ = Matrix();
}

// Factory function
std::unique_ptr<Optimizer> create_optimizer(const std::string& name, double learning_rate) {
    if (name == "sgd" || name == "SGD") {
        return std::make_unique<SGD>(learning_rate);
    } else if (name == "adam" || name == "Adam") {
        return std::make_unique<Adam>(learning_rate);
    } else if (name == "rmsprop" || name == "RMSprop") {
        return std::make_unique<RMSprop>(learning_rate);
    } else if (name == "adagrad" || name == "AdaGrad") {
        return std::make_unique<AdaGrad>(learning_rate);
    } else {
        throw std::invalid_argument("Unknown optimizer: " + name);
    }
}

} // namespace asekioml
