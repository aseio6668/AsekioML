#pragma once

#include "optimizer.hpp"
#include <unordered_map>
#include <cmath>

namespace clmodel {

// AdamW optimizer (Adam with weight decay)
class AdamW : public Optimizer {
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    int timestep_;
    
    std::unordered_map<void*, Matrix> m_; // First moment estimates
    std::unordered_map<void*, Matrix> v_; // Second moment estimates
    
public:
    AdamW(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, 
          double epsilon = 1e-8, double weight_decay = 0.01)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), 
          epsilon_(epsilon), weight_decay_(weight_decay), timestep_(0) {}
    
    void update(Matrix& weights, const Matrix& gradients) override {
        timestep_++;
        
        void* weights_ptr = &weights;
        
        // Initialize moments if first time
        if (m_.find(weights_ptr) == m_.end()) {
            m_[weights_ptr] = Matrix(weights.rows(), weights.cols());
            v_[weights_ptr] = Matrix(weights.rows(), weights.cols());
        }
        
        Matrix& m = m_[weights_ptr];
        Matrix& v = v_[weights_ptr];
        
        // Update biased first moment estimate
        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                m(i, j) = beta1_ * m(i, j) + (1.0 - beta1_) * gradients(i, j);
                v(i, j) = beta2_ * v(i, j) + (1.0 - beta2_) * gradients(i, j) * gradients(i, j);
                
                // Bias correction
                double m_hat = m(i, j) / (1.0 - std::pow(beta1_, timestep_));
                double v_hat = v(i, j) / (1.0 - std::pow(beta2_, timestep_));
                
                // AdamW update with weight decay
                weights(i, j) = weights(i, j) * (1.0 - learning_rate_ * weight_decay_) -
                               learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
      void update_bias(Matrix& biases, const Matrix& gradients) override {
        timestep_++;
        
        void* bias_ptr = &biases;
        
        // Initialize moments if first time
        if (m_.find(bias_ptr) == m_.end()) {
            m_[bias_ptr] = Matrix(biases.rows(), biases.cols());
            v_[bias_ptr] = Matrix(biases.rows(), biases.cols());
        }
        
        Matrix& m = m_[bias_ptr];
        Matrix& v = v_[bias_ptr];
        
        // Update biased first moment estimate
        for (size_t i = 0; i < biases.rows(); ++i) {
            for (size_t j = 0; j < biases.cols(); ++j) {
                m(i, j) = beta1_ * m(i, j) + (1.0 - beta1_) * gradients(i, j);
                v(i, j) = beta2_ * v(i, j) + (1.0 - beta2_) * gradients(i, j) * gradients(i, j);
                
                // Bias correction
                double m_hat = m(i, j) / (1.0 - std::pow(beta1_, timestep_));
                double v_hat = v(i, j) / (1.0 - std::pow(beta2_, timestep_));
                
                // AdamW update (no weight decay for biases)
                biases(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
    
    void reset() override {
        m_.clear();
        v_.clear();
        timestep_ = 0;
    }
    
    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<AdamW>(learning_rate_, beta1_, beta2_, epsilon_, weight_decay_);
    }
    
    std::string name() const override {
        return "AdamW";
    }
    
    void set_learning_rate(double lr) override {
        learning_rate_ = lr;
    }
    
    double get_learning_rate() const override {
        return learning_rate_;
    }
};

// LAMB optimizer (Layer-wise Adaptive Moments optimizer for Large batch training)
class LAMB : public Optimizer {
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    int timestep_;
    
    std::unordered_map<void*, Matrix> m_;
    std::unordered_map<void*, Matrix> v_;
    
    double compute_norm(const Matrix& matrix) {
        double norm = 0.0;
        for (size_t i = 0; i < matrix.rows(); ++i) {
            for (size_t j = 0; j < matrix.cols(); ++j) {
                norm += matrix(i, j) * matrix(i, j);
            }
        }
        return std::sqrt(norm);
    }
    
public:
    LAMB(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999,
         double epsilon = 1e-6, double weight_decay = 0.01)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
          epsilon_(epsilon), weight_decay_(weight_decay), timestep_(0) {}
    
    void update(Matrix& weights, const Matrix& gradients) override {
        timestep_++;
        
        void* weights_ptr = &weights;
        
        if (m_.find(weights_ptr) == m_.end()) {
            m_[weights_ptr] = Matrix(weights.rows(), weights.cols());
            v_[weights_ptr] = Matrix(weights.rows(), weights.cols());
        }
        
        Matrix& m = m_[weights_ptr];
        Matrix& v = v_[weights_ptr];
        
        // Update moments
        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                m(i, j) = beta1_ * m(i, j) + (1.0 - beta1_) * gradients(i, j);
                v(i, j) = beta2_ * v(i, j) + (1.0 - beta2_) * gradients(i, j) * gradients(i, j);
            }
        }
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, timestep_);
        double bias_correction2 = 1.0 - std::pow(beta2_, timestep_);
        
        // Compute update direction
        Matrix update(weights.rows(), weights.cols());
        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                double m_hat = m(i, j) / bias_correction1;
                double v_hat = v(i, j) / bias_correction2;
                update(i, j) = m_hat / (std::sqrt(v_hat) + epsilon_) + weight_decay_ * weights(i, j);
            }
        }
        
        // Layer-wise adaptation
        double weight_norm = compute_norm(weights);
        double update_norm = compute_norm(update);
        
        double trust_ratio = 1.0;
        if (weight_norm > 0 && update_norm > 0) {
            trust_ratio = weight_norm / update_norm;
        }
        
        // Apply update
        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                weights(i, j) -= learning_rate_ * trust_ratio * update(i, j);
            }
        }
    }
      void update_bias(Matrix& biases, const Matrix& gradients) override {
        timestep_++;
        
        void* bias_ptr = &biases;
        
        if (m_.find(bias_ptr) == m_.end()) {
            m_[bias_ptr] = Matrix(biases.rows(), biases.cols());
            v_[bias_ptr] = Matrix(biases.rows(), biases.cols());
        }
        
        Matrix& m = m_[bias_ptr];
        Matrix& v = v_[bias_ptr];
        
        // Update moments
        for (size_t i = 0; i < biases.rows(); ++i) {
            for (size_t j = 0; j < biases.cols(); ++j) {
                m(i, j) = beta1_ * m(i, j) + (1.0 - beta1_) * gradients(i, j);
                v(i, j) = beta2_ * v(i, j) + (1.0 - beta2_) * gradients(i, j) * gradients(i, j);
            }
        }
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, timestep_);
        double bias_correction2 = 1.0 - std::pow(beta2_, timestep_);
        
        // Compute update direction (no weight decay for biases)
        Matrix update(biases.rows(), biases.cols());
        for (size_t i = 0; i < biases.rows(); ++i) {
            for (size_t j = 0; j < biases.cols(); ++j) {
                double m_hat = m(i, j) / bias_correction1;
                double v_hat = v(i, j) / bias_correction2;
                update(i, j) = m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
        
        // Layer-wise adaptation
        double bias_norm = compute_norm(biases);
        double update_norm = compute_norm(update);
        
        double trust_ratio = 1.0;
        if (bias_norm > 0 && update_norm > 0) {
            trust_ratio = bias_norm / update_norm;
        }
        
        // Apply update
        for (size_t i = 0; i < biases.rows(); ++i) {
            for (size_t j = 0; j < biases.cols(); ++j) {
                biases(i, j) -= learning_rate_ * trust_ratio * update(i, j);
            }
        }
    }
    
    void reset() override {
        m_.clear();
        v_.clear();
        timestep_ = 0;
    }
    
    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<LAMB>(learning_rate_, beta1_, beta2_, epsilon_, weight_decay_);
    }
    
    std::string name() const override {
        return "LAMB";
    }
    
    void set_learning_rate(double lr) override {
        learning_rate_ = lr;
    }
    
    double get_learning_rate() const override {
        return learning_rate_;
    }
};

// Lookahead optimizer wrapper
class Lookahead : public Optimizer {
private:
    std::unique_ptr<Optimizer> base_optimizer_;
    double alpha_;  // Slow weights step size
    int k_;         // Update frequency
    int step_count_;
    
    std::unordered_map<void*, Matrix> slow_weights_;
    
public:
    Lookahead(std::unique_ptr<Optimizer> base_optimizer, double alpha = 0.5, int k = 5)
        : base_optimizer_(std::move(base_optimizer)), alpha_(alpha), k_(k), step_count_(0) {}
    
    void update(Matrix& weights, const Matrix& gradients) override {
        // Fast weights update using base optimizer
        base_optimizer_->update(weights, gradients);
        step_count_++;
        
        void* weights_ptr = &weights;
        
        // Initialize slow weights if first time
        if (slow_weights_.find(weights_ptr) == slow_weights_.end()) {
            slow_weights_[weights_ptr] = weights;  // Copy current weights
        }
        
        // Update slow weights every k steps
        if (step_count_ % k_ == 0) {
            Matrix& slow_w = slow_weights_[weights_ptr];
            for (size_t i = 0; i < weights.rows(); ++i) {
                for (size_t j = 0; j < weights.cols(); ++j) {
                    slow_w(i, j) += alpha_ * (weights(i, j) - slow_w(i, j));
                    weights(i, j) = slow_w(i, j);
                }
            }
        }
    }
      void update_bias(Matrix& biases, const Matrix& gradients) override {
        // Fast biases update using base optimizer
        base_optimizer_->update_bias(biases, gradients);
        step_count_++;
        
        void* bias_ptr = &biases;
        
        // Initialize slow biases if first time
        if (slow_weights_.find(bias_ptr) == slow_weights_.end()) {
            slow_weights_[bias_ptr] = biases;  // Copy current biases
        }
        
        // Update slow biases every k steps
        if (step_count_ % k_ == 0) {
            Matrix& slow_b = slow_weights_[bias_ptr];
            for (size_t i = 0; i < biases.rows(); ++i) {
                for (size_t j = 0; j < biases.cols(); ++j) {
                    slow_b(i, j) += alpha_ * (biases(i, j) - slow_b(i, j));
                    biases(i, j) = slow_b(i, j);
                }
            }
        }
    }
    
    void reset() override {
        base_optimizer_->reset();
        slow_weights_.clear();
        step_count_ = 0;
    }
    
    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<Lookahead>(base_optimizer_->clone(), alpha_, k_);
    }
    
    std::string name() const override {
        return "Lookahead(" + base_optimizer_->name() + ")";
    }
    
    void set_learning_rate(double lr) override {
        base_optimizer_->set_learning_rate(lr);
    }
    
    double get_learning_rate() const override {
        return base_optimizer_->get_learning_rate();
    }
};

// Ranger optimizer (RAdam + Lookahead)
class RAdam : public Optimizer {
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    int timestep_;
    
    std::unordered_map<void*, Matrix> m_;
    std::unordered_map<void*, Matrix> v_;
    
public:
    RAdam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), timestep_(0) {}
    
    void update(Matrix& weights, const Matrix& gradients) override {
        timestep_++;
        
        void* weights_ptr = &weights;
        
        if (m_.find(weights_ptr) == m_.end()) {
            m_[weights_ptr] = Matrix(weights.rows(), weights.cols());
            v_[weights_ptr] = Matrix(weights.rows(), weights.cols());
        }
        
        Matrix& m = m_[weights_ptr];
        Matrix& v = v_[weights_ptr];
        
        // Compute rho_inf
        double rho_inf = 2.0 / (1.0 - beta2_) - 1.0;
        
        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                // Update moments
                m(i, j) = beta1_ * m(i, j) + (1.0 - beta1_) * gradients(i, j);
                v(i, j) = beta2_ * v(i, j) + (1.0 - beta2_) * gradients(i, j) * gradients(i, j);
                
                // Bias correction for first moment
                double m_hat = m(i, j) / (1.0 - std::pow(beta1_, timestep_));
                
                // Compute rho_t
                double rho_t = rho_inf - 2.0 * timestep_ * std::pow(beta2_, timestep_) / 
                              (1.0 - std::pow(beta2_, timestep_));
                
                if (rho_t > 4.0) {
                    // Bias correction for second moment
                    double v_hat = std::sqrt(v(i, j) / (1.0 - std::pow(beta2_, timestep_)));
                    
                    // Compute rectification term
                    double r_t = std::sqrt((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / 
                                          ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t));
                    
                    weights(i, j) -= learning_rate_ * r_t * m_hat / (v_hat + epsilon_);
                } else {
                    // Use momentum only
                    weights(i, j) -= learning_rate_ * m_hat;
                }
            }
        }
    }
      void update_bias(Matrix& biases, const Matrix& gradients) override {
        timestep_++;
        
        void* bias_ptr = &biases;
        
        if (m_.find(bias_ptr) == m_.end()) {
            m_[bias_ptr] = Matrix(biases.rows(), biases.cols());
            v_[bias_ptr] = Matrix(biases.rows(), biases.cols());
        }
        
        Matrix& m = m_[bias_ptr];
        Matrix& v = v_[bias_ptr];
        
        // Compute rho_inf
        double rho_inf = 2.0 / (1.0 - beta2_) - 1.0;
        
        for (size_t i = 0; i < biases.rows(); ++i) {
            for (size_t j = 0; j < biases.cols(); ++j) {
                // Update moments
                m(i, j) = beta1_ * m(i, j) + (1.0 - beta1_) * gradients(i, j);
                v(i, j) = beta2_ * v(i, j) + (1.0 - beta2_) * gradients(i, j) * gradients(i, j);
                
                // Bias correction for first moment
                double m_hat = m(i, j) / (1.0 - std::pow(beta1_, timestep_));
                
                // Compute rho_t
                double rho_t = rho_inf - 2.0 * timestep_ * std::pow(beta2_, timestep_) / 
                              (1.0 - std::pow(beta2_, timestep_));
                
                if (rho_t > 4.0) {
                    // Bias correction for second moment
                    double v_hat = std::sqrt(v(i, j) / (1.0 - std::pow(beta2_, timestep_)));
                    
                    // Compute rectification term
                    double r_t = std::sqrt((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / 
                                          ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t));
                    
                    biases(i, j) -= learning_rate_ * r_t * m_hat / (v_hat + epsilon_);
                } else {
                    // Use momentum only
                    biases(i, j) -= learning_rate_ * m_hat;
                }
            }
        }
    }
    
    void reset() override {
        m_.clear();
        v_.clear();
        timestep_ = 0;
    }
    
    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<RAdam>(learning_rate_, beta1_, beta2_, epsilon_);
    }
    
    std::string name() const override {
        return "RAdam";
    }
    
    void set_learning_rate(double lr) override {
        learning_rate_ = lr;
    }
    
    double get_learning_rate() const override {
        return learning_rate_;
    }
};

} // namespace clmodel
