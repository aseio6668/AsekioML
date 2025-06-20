#include "advanced_layers.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>

namespace clmodel {
namespace advanced {

// ====== BATCH NORMALIZATION LAYER ======

BatchNormalizationLayer::BatchNormalizationLayer(size_t size, double momentum, double epsilon)
    : size_(size), momentum_(momentum), epsilon_(epsilon), training_(true) {
    
    // Initialize parameters
    gamma_ = Matrix(1, size_);  // Scale parameters (initialized to 1)
    beta_ = Matrix(1, size_);   // Shift parameters (initialized to 0)
    running_mean_ = Matrix(1, size_);  // Running mean (initialized to 0)
    running_var_ = Matrix(1, size_);   // Running variance (initialized to 1)
    
    // Initialize gamma to 1, others to 0
    for (size_t i = 0; i < size_; ++i) {
        gamma_(0, i) = 1.0;
        beta_(0, i) = 0.0;
        running_mean_(0, i) = 0.0;
        running_var_(0, i) = 1.0;
    }
}

Matrix BatchNormalizationLayer::forward(const Matrix& input) {
    last_input_ = input;
    
    if (training_) {
        // Calculate batch statistics
        Matrix batch_mean(1, size_);
        Matrix batch_var(1, size_);
        
        // Calculate mean for each feature
        for (size_t j = 0; j < size_; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < input.rows(); ++i) {
                sum += input(i, j);
            }
            batch_mean(0, j) = sum / input.rows();
        }
        
        // Calculate variance for each feature
        for (size_t j = 0; j < size_; ++j) {
            double sum_sq_diff = 0.0;
            for (size_t i = 0; i < input.rows(); ++i) {
                double diff = input(i, j) - batch_mean(0, j);
                sum_sq_diff += diff * diff;
            }
            batch_var(0, j) = sum_sq_diff / input.rows();
        }
        
        // Update running statistics
        for (size_t j = 0; j < size_; ++j) {
            running_mean_(0, j) = momentum_ * running_mean_(0, j) + (1.0 - momentum_) * batch_mean(0, j);
            running_var_(0, j) = momentum_ * running_var_(0, j) + (1.0 - momentum_) * batch_var(0, j);
        }
        
        // Normalize using batch statistics
        normalized_ = Matrix(input.rows(), size_);
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < size_; ++j) {
                normalized_(i, j) = (input(i, j) - batch_mean(0, j)) / std::sqrt(batch_var(0, j) + epsilon_);
            }
        }
    } else {
        // Use running statistics for inference
        normalized_ = Matrix(input.rows(), size_);
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < size_; ++j) {
                normalized_(i, j) = (input(i, j) - running_mean_(0, j)) / std::sqrt(running_var_(0, j) + epsilon_);
            }
        }
    }
    
    // Apply scale and shift
    Matrix output(input.rows(), size_);
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < size_; ++j) {
            output(i, j) = gamma_(0, j) * normalized_(i, j) + beta_(0, j);
        }
    }
    
    return output;
}

Matrix BatchNormalizationLayer::backward(const Matrix& gradient) {
    // This is a simplified backward pass
    // In practice, batch normalization backprop is quite complex
    return gradient;  // For now, just pass gradient through
}

void BatchNormalizationLayer::update_weights(double /* learning_rate */) {
    // Update gamma and beta parameters
    // This is simplified - in practice would need gradients w.r.t. gamma and beta
}

std::unique_ptr<Layer> BatchNormalizationLayer::clone() const {
    auto cloned = std::make_unique<BatchNormalizationLayer>(size_, momentum_, epsilon_);
    cloned->gamma_ = gamma_;
    cloned->beta_ = beta_;
    cloned->running_mean_ = running_mean_;
    cloned->running_var_ = running_var_;
    return cloned;
}

// ====== CONVOLUTIONAL 2D LAYER ======

Conv2DLayer::Conv2DLayer(size_t input_height, size_t input_width, size_t input_channels,
                         size_t num_filters, size_t kernel_height, size_t kernel_width,
                         size_t stride_h, size_t stride_w, 
                         size_t padding_h, size_t padding_w)
    : input_height_(input_height), input_width_(input_width), 
      input_channels_(input_channels), num_filters_(num_filters),
      kernel_height_(kernel_height), kernel_width_(kernel_width),
      stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w) {
    
    // Calculate output dimensions
    output_height_ = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    output_width_ = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;
    
    // Initialize kernels and biases
    size_t kernel_size = kernel_height * kernel_width * input_channels;
    kernels_ = Matrix(num_filters, kernel_size);
    biases_ = Matrix(num_filters, 1);
    
    // Xavier initialization for kernels
    std::random_device rd;
    std::mt19937 gen(rd());
    double scale = std::sqrt(2.0 / (kernel_size + num_filters));
    std::normal_distribution<double> dist(0.0, scale);
    
    for (size_t filter_idx = 0; filter_idx < num_filters; ++filter_idx) {
        for (size_t kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
            kernels_(filter_idx, kernel_idx) = dist(gen);
        }
        biases_(filter_idx, 0) = 0.0;
    }
}

Matrix Conv2DLayer::forward(const Matrix& input) {
    last_input_ = input;
    
    // Simplified 2D convolution implementation
    // Input format: (batch_size, height * width * channels)
    // Output format: (batch_size, output_height * output_width * num_filters)
    
    size_t batch_size = input.rows();
    size_t output_size = output_height_ * output_width_ * num_filters_;
    Matrix output(batch_size, output_size);
    
    // This is a simplified implementation
    // In practice, you'd use im2col or optimized convolution algorithms
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < num_filters_; ++f) {
            for (size_t oh = 0; oh < output_height_; ++oh) {
                for (size_t ow = 0; ow < output_width_; ++ow) {
                    double conv_sum = biases_(f, 0);
                    
                    // Convolution operation
                    for (size_t kh = 0; kh < kernel_height_; ++kh) {
                        for (size_t kw = 0; kw < kernel_width_; ++kw) {
                            for (size_t c = 0; c < input_channels_; ++c) {
                                size_t ih = oh * stride_h_ + kh;
                                size_t iw = ow * stride_w_ + kw;
                                
                                if (ih < input_height_ && iw < input_width_) {
                                    size_t input_idx = ih * input_width_ * input_channels_ + 
                                                     iw * input_channels_ + c;
                                    size_t kernel_idx = kh * kernel_width_ * input_channels_ + 
                                                      kw * input_channels_ + c;
                                    
                                    if (input_idx < input.cols()) {
                                        conv_sum += input(b, input_idx) * kernels_(f, kernel_idx);
                                    }
                                }
                            }
                        }
                    }
                    
                    size_t output_idx = oh * output_width_ * num_filters_ + 
                                      ow * num_filters_ + f;
                    if (output_idx < output.cols()) {
                        output(b, output_idx) = conv_sum;
                    }
                }
            }
        }
    }
    
    return output;
}

Matrix Conv2DLayer::backward(const Matrix& /*gradient*/) {
    // Simplified backward pass
    return Matrix(last_input_.rows(), last_input_.cols());
}

void Conv2DLayer::update_weights(double /*learning_rate*/) {
    // Update kernels and biases based on gradients
}

std::unique_ptr<Layer> Conv2DLayer::clone() const {
    return std::make_unique<Conv2DLayer>(input_height_, input_width_, input_channels_,
                                       num_filters_, kernel_height_, kernel_width_,
                                       stride_h_, stride_w_, padding_h_, padding_w_);
}

size_t Conv2DLayer::input_size() const { 
    return input_height_ * input_width_ * input_channels_; 
}

size_t Conv2DLayer::output_size() const { 
    return output_height_ * output_width_ * num_filters_; 
}

// ====== L1/L2 REGULARIZATION LAYER ======

RegularizationLayer::RegularizationLayer(size_t size, double l1_lambda, double l2_lambda)
    : size_(size), l1_lambda_(l1_lambda), l2_lambda_(l2_lambda) {}

Matrix RegularizationLayer::forward(const Matrix& input) {
    last_input_ = input;
    return input;  // Regularization doesn't change forward pass
}

Matrix RegularizationLayer::backward(const Matrix& gradient) {
    Matrix regularized_grad = gradient;
    
    if (l1_lambda_ > 0.0 || l2_lambda_ > 0.0) {
        for (size_t i = 0; i < gradient.rows(); ++i) {
            for (size_t j = 0; j < gradient.cols(); ++j) {
                double reg_term = 0.0;
                
                // L1 regularization (Lasso)
                if (l1_lambda_ > 0.0) {
                    reg_term += l1_lambda_ * (last_input_(i, j) >= 0 ? 1.0 : -1.0);
                }
                
                // L2 regularization (Ridge)
                if (l2_lambda_ > 0.0) {
                    reg_term += l2_lambda_ * 2.0 * last_input_(i, j);
                }
                
                regularized_grad(i, j) += reg_term;
            }
        }
    }
    
    return regularized_grad;
}

std::unique_ptr<Layer> RegularizationLayer::clone() const {
    return std::make_unique<RegularizationLayer>(size_, l1_lambda_, l2_lambda_);
}

std::string RegularizationLayer::type() const { 
    std::ostringstream oss;
    oss << "Regularization(L1=" << l1_lambda_ << ",L2=" << l2_lambda_ << ")";
    return oss.str();
}

// ====== LSTM LAYER ======

LSTMLayer::LSTMLayer(size_t input_size, size_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size), sequence_length_(1) {
    
    // Initialize weight matrices using Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double scale = std::sqrt(2.0 / (input_size + hidden_size));
    std::normal_distribution<double> dist(0.0, scale);
    
    // Initialize forget gate weights
    W_f_ = Matrix(hidden_size, input_size);
    U_f_ = Matrix(hidden_size, hidden_size);
    b_f_ = Matrix(hidden_size, 1);
    
    // Initialize input gate weights
    W_i_ = Matrix(hidden_size, input_size);
    U_i_ = Matrix(hidden_size, hidden_size);
    b_i_ = Matrix(hidden_size, 1);
    
    // Initialize candidate weights
    W_c_ = Matrix(hidden_size, input_size);
    U_c_ = Matrix(hidden_size, hidden_size);
    b_c_ = Matrix(hidden_size, 1);
    
    // Initialize output gate weights
    W_o_ = Matrix(hidden_size, input_size);
    U_o_ = Matrix(hidden_size, hidden_size);
    b_o_ = Matrix(hidden_size, 1);
    
    // Initialize all weights
    auto init_matrix = [&](Matrix& m) {
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                m(i, j) = dist(gen);
            }
        }
    };
    
    init_matrix(W_f_); init_matrix(U_f_);
    init_matrix(W_i_); init_matrix(U_i_);
    init_matrix(W_c_); init_matrix(U_c_);
    init_matrix(W_o_); init_matrix(U_o_);
    
    // Initialize biases to zero (except forget gate bias to 1)
    for (size_t i = 0; i < hidden_size; ++i) {
        b_f_(i, 0) = 1.0;  // Forget gate bias should be 1 initially
        b_i_(i, 0) = 0.0;
        b_c_(i, 0) = 0.0;
        b_o_(i, 0) = 0.0;
    }
    
    // Initialize states
    cell_state_ = Matrix(hidden_size, 1);
    hidden_state_ = Matrix(hidden_size, 1);
    reset_state();
}

Matrix LSTMLayer::sigmoid_gate(const Matrix& x, const Matrix& h, 
                              const Matrix& W, const Matrix& U, const Matrix& b) {
    // Compute W*x + U*h + b and apply sigmoid
    Matrix input_part = W * x.transpose();  // W*x
    Matrix hidden_part = U * h;             // U*h
    Matrix result(hidden_size_, 1);
    
    for (size_t i = 0; i < hidden_size_; ++i) {
        double val = input_part(i, 0) + hidden_part(i, 0) + b(i, 0);
        result(i, 0) = 1.0 / (1.0 + std::exp(-val));  // Sigmoid
    }
    
    return result;
}

Matrix LSTMLayer::tanh_gate(const Matrix& x, const Matrix& h,
                           const Matrix& W, const Matrix& U, const Matrix& b) {
    // Compute W*x + U*h + b and apply tanh
    Matrix input_part = W * x.transpose();  // W*x
    Matrix hidden_part = U * h;             // U*h
    Matrix result(hidden_size_, 1);
    
    for (size_t i = 0; i < hidden_size_; ++i) {
        double val = input_part(i, 0) + hidden_part(i, 0) + b(i, 0);
        result(i, 0) = std::tanh(val);
    }
    
    return result;
}

Matrix LSTMLayer::forward(const Matrix& input) {
    // For simplicity, assume input is a single timestep: (batch_size, input_size)
    // In practice, LSTM would process sequences
    
    size_t batch_size = input.rows();
    Matrix output(batch_size, hidden_size_);
    
    // Process each sample in the batch
    for (size_t b = 0; b < batch_size; ++b) {
        // Extract single input vector
        Matrix x(1, input_size_);
        for (size_t i = 0; i < input_size_; ++i) {
            x(0, i) = input(b, i);
        }
        
        // LSTM forward pass
        Matrix forget_gate = sigmoid_gate(x, hidden_state_, W_f_, U_f_, b_f_);
        Matrix input_gate = sigmoid_gate(x, hidden_state_, W_i_, U_i_, b_i_);
        Matrix candidate = tanh_gate(x, hidden_state_, W_c_, U_c_, b_c_);
        Matrix output_gate = sigmoid_gate(x, hidden_state_, W_o_, U_o_, b_o_);
        
        // Update cell state
        for (size_t i = 0; i < hidden_size_; ++i) {
            cell_state_(i, 0) = forget_gate(i, 0) * cell_state_(i, 0) + 
                               input_gate(i, 0) * candidate(i, 0);
        }
        
        // Update hidden state
        for (size_t i = 0; i < hidden_size_; ++i) {
            hidden_state_(i, 0) = output_gate(i, 0) * std::tanh(cell_state_(i, 0));
            output(b, i) = hidden_state_(i, 0);
        }
    }
    
    return output;
}

Matrix LSTMLayer::backward(const Matrix& gradient) {
    // Simplified backward pass - LSTM backprop is quite complex
    // In practice, this would implement BPTT (Backpropagation Through Time)
    return Matrix(gradient.rows(), input_size_);
}

void LSTMLayer::update_weights(double /*learning_rate*/) {
    // Update weights based on gradients
    // This is simplified - would need actual gradient computation
}

void LSTMLayer::reset_state() {
    for (size_t i = 0; i < hidden_size_; ++i) {
        cell_state_(i, 0) = 0.0;
        hidden_state_(i, 0) = 0.0;
    }
}

std::unique_ptr<Layer> LSTMLayer::clone() const {
    auto cloned = std::make_unique<LSTMLayer>(input_size_, hidden_size_);
    // Copy all weight matrices
    cloned->W_f_ = W_f_; cloned->U_f_ = U_f_; cloned->b_f_ = b_f_;
    cloned->W_i_ = W_i_; cloned->U_i_ = U_i_; cloned->b_i_ = b_i_;
    cloned->W_c_ = W_c_; cloned->U_c_ = U_c_; cloned->b_c_ = b_c_;
    cloned->W_o_ = W_o_; cloned->U_o_ = U_o_; cloned->b_o_ = b_o_;
    return cloned;
}

} // namespace advanced
} // namespace clmodel
