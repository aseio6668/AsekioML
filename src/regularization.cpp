#include "../include/regularization.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <fstream>

namespace asekioml {

// ==================== DropoutLayer Implementation ====================

DropoutLayer::DropoutLayer(float dropout_rate) 
    : dropout_rate_(dropout_rate), training_mode_(true), input_size_(0),
      rng_(std::random_device{}()), dist_(0.0f, 1.0f) {
    if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in range [0.0, 1.0)");
    }
}

void DropoutLayer::set_dropout_rate(float rate) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in range [0.0, 1.0)");
    }
    dropout_rate_ = rate;
}

void DropoutLayer::set_training_mode(bool training) {
    training_mode_ = training;
}

Matrix DropoutLayer::forward(const Matrix& input) {
    if (input_size_ == 0) {
        input_size_ = input.cols();
    }
    
    if (!training_mode_) {
        // During inference, keep all units and scale by keep probability
        return input * (1.0f - dropout_rate_);
    }
    
    // During training, apply dropout mask
    dropout_mask_ = Matrix(input.rows(), input.cols());
    float keep_prob = 1.0f - dropout_rate_;
    
    // Generate random mask
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            dropout_mask_(i, j) = (dist_(rng_) < keep_prob) ? (1.0f / keep_prob) : 0.0f;
        }
    }
    
    // Apply mask to input
    Matrix output(input.rows(), input.cols());
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = input(i, j) * dropout_mask_(i, j);
        }
    }
    
    return output;
}

Matrix DropoutLayer::backward(const Matrix& gradient) {
    if (!training_mode_) {
        // During inference, just pass gradients through scaled
        return gradient * (1.0f - dropout_rate_);
    }
    
    // During training, apply the same mask used in forward pass
    Matrix output_grad(gradient.rows(), gradient.cols());
    for (size_t i = 0; i < gradient.rows(); ++i) {
        for (size_t j = 0; j < gradient.cols(); ++j) {
            output_grad(i, j) = gradient(i, j) * dropout_mask_(i, j);
        }
    }
    
    return output_grad;
}

void DropoutLayer::update_weights(double learning_rate) {
    // Dropout has no trainable parameters
    (void)learning_rate;
}

size_t DropoutLayer::output_size() const {
    return input_size_;
}

size_t DropoutLayer::input_size() const {
    return input_size_;
}

std::unique_ptr<Layer> DropoutLayer::clone() const {
    auto cloned = std::make_unique<DropoutLayer>(dropout_rate_);
    cloned->set_training_mode(training_mode_);
    cloned->input_size_ = input_size_;
    return cloned;
}

std::string DropoutLayer::serialize_to_json() const {
    std::ostringstream ss;
    ss << "{";
    ss << "\"type\":\"Dropout\",";
    ss << "\"dropout_rate\":" << dropout_rate_ << ",";
    ss << "\"input_size\":" << input_size_;
    ss << "}";
    return ss.str();
}

void DropoutLayer::serialize_weights(std::ofstream& file) const {
    // Dropout layers have no weights to serialize
}

void DropoutLayer::deserialize_weights(std::ifstream& file) {
    // Dropout layers have no weights to deserialize
}

size_t DropoutLayer::get_weights_size() const {
    return 0; // No weights in dropout layers
}

// ==================== WeightDecay Implementation ====================

void WeightDecay::apply(const Matrix& weights, Matrix& gradients) const {
    if (lambda_ <= 0.0f) return;
    
    // Add L2 regularization: grad += lambda * weights
    for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
            gradients(i, j) += lambda_ * weights(i, j);
        }
    }
}

float WeightDecay::compute_loss(const Matrix& weights) const {
    if (lambda_ <= 0.0f) return 0.0f;
    
    float l2_norm = 0.0f;
    for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
            l2_norm += weights(i, j) * weights(i, j);
        }
    }
    
    return 0.5f * lambda_ * l2_norm;
}

// ==================== BatchNormLayer Implementation ====================

BatchNormLayer::BatchNormLayer(float momentum, float epsilon)
    : momentum_(momentum), epsilon_(epsilon), training_mode_(true), input_size_(0) {
    if (momentum < 0.0f || momentum > 1.0f) {
        throw std::invalid_argument("Momentum must be in range [0.0, 1.0]");
    }
    if (epsilon <= 0.0f) {
        throw std::invalid_argument("Epsilon must be positive");
    }
}

void BatchNormLayer::set_input_size(size_t size) {
    input_size_ = size;
    
    // Initialize parameters
    gamma_ = Matrix(1, size);
    beta_ = Matrix(1, size);
    gamma_grad_ = Matrix(1, size);
    beta_grad_ = Matrix(1, size);
    running_mean_ = Matrix(1, size);
    running_var_ = Matrix(1, size);
    
    // Initialize gamma to 1, beta to 0
    for (size_t i = 0; i < size; ++i) {
        gamma_(0, i) = 1.0f;
        beta_(0, i) = 0.0f;
        running_mean_(0, i) = 0.0f;
        running_var_(0, i) = 1.0f;
    }
}

Matrix BatchNormLayer::forward(const Matrix& input) {
    if (input_size_ == 0) {
        input_size_ = input.cols();
        set_input_size(input_size_);
    }
    
    last_input_ = input;
    size_t batch_size = input.rows();
    
    Matrix output(batch_size, input_size_);
    
    if (training_mode_) {
        // Compute batch statistics
        Matrix batch_mean(1, input_size_);
        Matrix batch_var(1, input_size_);
        
        // Compute mean
        for (size_t j = 0; j < input_size_; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < batch_size; ++i) {
                sum += static_cast<float>(input(i, j));
            }
            batch_mean(0, j) = sum / batch_size;
        }
        
        // Compute variance
        for (size_t j = 0; j < input_size_; ++j) {
            float sum_sq = 0.0f;
            for (size_t i = 0; i < batch_size; ++i) {
                float diff = static_cast<float>(input(i, j)) - batch_mean(0, j);
                sum_sq += diff * diff;
            }
            batch_var(0, j) = sum_sq / batch_size;
        }
        
        // Update running statistics
        for (size_t j = 0; j < input_size_; ++j) {
            running_mean_(0, j) = momentum_ * running_mean_(0, j) + (1.0f - momentum_) * batch_mean(0, j);
            running_var_(0, j) = momentum_ * running_var_(0, j) + (1.0f - momentum_) * batch_var(0, j);
        }
        
        // Normalize and scale
        normalized_ = Matrix(batch_size, input_size_);
        variance_ = batch_var;
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                float normalized_val = (static_cast<float>(input(i, j)) - batch_mean(0, j)) / std::sqrt(batch_var(0, j) + epsilon_);
                normalized_(i, j) = normalized_val;
                output(i, j) = gamma_(0, j) * normalized_val + beta_(0, j);
            }
        }
    } else {
        // Use running statistics for inference
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                float normalized_val = (static_cast<float>(input(i, j)) - running_mean_(0, j)) / std::sqrt(running_var_(0, j) + epsilon_);
                output(i, j) = gamma_(0, j) * normalized_val + beta_(0, j);
            }
        }
    }
    
    return output;
}

Matrix BatchNormLayer::backward(const Matrix& gradient) {
    size_t batch_size = gradient.rows();
    
    // Initialize gradients
    gamma_grad_ = Matrix(1, input_size_);
    beta_grad_ = Matrix(1, input_size_);
    
    // Compute parameter gradients
    for (size_t j = 0; j < input_size_; ++j) {
        float gamma_grad_sum = 0.0f;
        float beta_grad_sum = 0.0f;
        
        for (size_t i = 0; i < batch_size; ++i) {
            gamma_grad_sum += static_cast<float>(gradient(i, j)) * normalized_(i, j);
            beta_grad_sum += static_cast<float>(gradient(i, j));
        }
        
        gamma_grad_(0, j) = gamma_grad_sum;
        beta_grad_(0, j) = beta_grad_sum;
    }
    
    // Compute input gradients (simplified version)
    Matrix input_grad(batch_size, input_size_);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < input_size_; ++j) {
            input_grad(i, j) = static_cast<float>(gradient(i, j)) * gamma_(0, j) / std::sqrt(variance_(0, j) + epsilon_);
        }
    }
    
    return input_grad;
}

void BatchNormLayer::update_weights(double learning_rate) {
    // Update gamma and beta using simple gradient descent
    for (size_t j = 0; j < input_size_; ++j) {
        gamma_(0, j) -= static_cast<float>(learning_rate) * gamma_grad_(0, j);
        beta_(0, j) -= static_cast<float>(learning_rate) * beta_grad_(0, j);
    }
}

std::unique_ptr<Layer> BatchNormLayer::clone() const {
    auto cloned = std::make_unique<BatchNormLayer>(momentum_, epsilon_);
    cloned->set_training_mode(training_mode_);
    if (input_size_ > 0) {
        cloned->input_size_ = input_size_;
        cloned->set_input_size(input_size_);
        // Copy learned parameters
        cloned->gamma_ = gamma_;
        cloned->beta_ = beta_;
        cloned->running_mean_ = running_mean_;
        cloned->running_var_ = running_var_;
    }
    return cloned;
}

std::string BatchNormLayer::serialize_to_json() const {
    std::ostringstream ss;
    ss << "{";
    ss << "\"type\":\"BatchNorm\",";
    ss << "\"momentum\":" << momentum_ << ",";
    ss << "\"epsilon\":" << epsilon_ << ",";
    ss << "\"input_size\":" << input_size_;
    ss << "}";
    return ss.str();
}

void BatchNormLayer::serialize_weights(std::ofstream& file) const {
    // Write gamma matrix
    uint32_t rows = static_cast<uint32_t>(gamma_.rows());
    uint32_t cols = static_cast<uint32_t>(gamma_.cols());
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    for (size_t i = 0; i < gamma_.rows(); ++i) {
        for (size_t j = 0; j < gamma_.cols(); ++j) {
            double value = gamma_(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
    
    // Write beta matrix
    rows = static_cast<uint32_t>(beta_.rows());
    cols = static_cast<uint32_t>(beta_.cols());
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    for (size_t i = 0; i < beta_.rows(); ++i) {
        for (size_t j = 0; j < beta_.cols(); ++j) {
            double value = beta_(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
    
    // Write running mean
    rows = static_cast<uint32_t>(running_mean_.rows());
    cols = static_cast<uint32_t>(running_mean_.cols());
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    for (size_t i = 0; i < running_mean_.rows(); ++i) {
        for (size_t j = 0; j < running_mean_.cols(); ++j) {
            double value = running_mean_(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
    
    // Write running variance
    rows = static_cast<uint32_t>(running_var_.rows());
    cols = static_cast<uint32_t>(running_var_.cols());
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    for (size_t i = 0; i < running_var_.rows(); ++i) {
        for (size_t j = 0; j < running_var_.cols(); ++j) {
            double value = running_var_(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
}

void BatchNormLayer::deserialize_weights(std::ifstream& file) {
    // Read gamma matrix
    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    gamma_ = Matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            gamma_(i, j) = value;
        }
    }
    
    // Read beta matrix
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    beta_ = Matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            beta_(i, j) = value;
        }
    }
    
    // Read running mean
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    running_mean_ = Matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            running_mean_(i, j) = value;
        }
    }
    
    // Read running variance
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    running_var_ = Matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            running_var_(i, j) = value;
        }
    }
    
    // Reinitialize gradient matrices
    gamma_grad_ = Matrix(gamma_.rows(), gamma_.cols());
    beta_grad_ = Matrix(beta_.rows(), beta_.cols());
}

size_t BatchNormLayer::get_weights_size() const {
    return gamma_.rows() * gamma_.cols() * 4; // gamma, beta, running_mean, running_var
}

} // namespace asekioml
