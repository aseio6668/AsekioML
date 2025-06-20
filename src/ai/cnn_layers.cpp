#include "../../include/ai/cnn_layers.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <limits>

namespace asekioml {
namespace ai {

// ================================================================================================
// Conv2DLayer Implementation
// ================================================================================================

Conv2DLayer::Conv2DLayer(size_t input_channels, size_t output_channels, 
                         size_t kernel_size, size_t stride, size_t padding)
    : input_channels_(input_channels)
    , output_channels_(output_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , input_height_(0)
    , input_width_(0)
    , weights_initialized_(false) {
    
    if (stride_ == 0) {
        stride_ = 1;
    }
}

void Conv2DLayer::set_input_dimensions(size_t height, size_t width) {
    input_height_ = height;
    input_width_ = width;
    
    // Initialize weights if not already done
    if (!weights_initialized_) {
        initialize_weights();
        weights_initialized_ = true;
    }
}

void Conv2DLayer::initialize_weights(const std::string& method) {
    // Weights shape: [output_channels, input_channels, kernel_size, kernel_size]
    std::vector<size_t> weight_shape = {output_channels_, input_channels_, kernel_size_, kernel_size_};
    std::vector<size_t> bias_shape = {output_channels_};
    
    if (method == "xavier" || method == "glorot") {
        // Xavier/Glorot initialization
        double limit = std::sqrt(6.0 / (input_channels_ * kernel_size_ * kernel_size_ + 
                                       output_channels_ * kernel_size_ * kernel_size_));
        weights_ = Tensor::random(weight_shape, -limit, limit);
    } else if (method == "he" || method == "kaiming") {
        // He/Kaiming initialization (better for ReLU)
        double std_dev = std::sqrt(2.0 / (input_channels_ * kernel_size_ * kernel_size_));
        weights_ = Tensor::randn(weight_shape, 0.0, std_dev);
    } else {
        // Default: small random values
        weights_ = Tensor::randn(weight_shape, 0.0, 0.01);
    }
    
    // Initialize biases to zero
    biases_ = Tensor::zeros(bias_shape);
    
    // Initialize gradient tensors
    weight_gradients_ = Tensor::zeros(weight_shape);
    bias_gradients_ = Tensor::zeros(bias_shape);
}

size_t Conv2DLayer::calculate_output_size(size_t input_size, size_t kernel_size, 
                                         size_t stride, size_t padding) const {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

size_t Conv2DLayer::get_output_height() const {
    if (input_height_ == 0) return 0;
    return calculate_output_size(input_height_, kernel_size_, stride_, padding_);
}

size_t Conv2DLayer::get_output_width() const {
    if (input_width_ == 0) return 0;
    return calculate_output_size(input_width_, kernel_size_, stride_, padding_);
}

void Conv2DLayer::validate_input_tensor(const Tensor& input) const {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Conv2D input must be 4D: [batch, channels, height, width]");
    }
    
    if (input.size(1) != input_channels_) {
        throw std::invalid_argument("Input channels mismatch: expected " + 
                                  std::to_string(input_channels_) + 
                                  ", got " + std::to_string(input.size(1)));
    }
}

Tensor Conv2DLayer::im2col(const Tensor& input) const {
    // Convert convolution to matrix multiplication using im2col
    // Input: [batch, channels, height, width]
    // Output: [batch * out_h * out_w, channels * kernel_h * kernel_w]
    
    size_t batch_size = input.size(0);
    size_t channels = input.size(1);
    size_t height = input.size(2);
    size_t width = input.size(3);
    
    size_t out_h = get_output_height();
    size_t out_w = get_output_width();
    
    size_t col_height = batch_size * out_h * out_w;
    size_t col_width = channels * kernel_size_ * kernel_size_;
    
    Tensor col = Tensor::zeros({col_height, col_width});
    
    size_t col_idx = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t y = 0; y < out_h; ++y) {
            for (size_t x = 0; x < out_w; ++x) {
                size_t col_elem = 0;
                
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t ky = 0; ky < kernel_size_; ++ky) {
                        for (size_t kx = 0; kx < kernel_size_; ++kx) {
                            // Calculate input position with padding
                            int in_y = static_cast<int>(y * stride_ + ky) - static_cast<int>(padding_);
                            int in_x = static_cast<int>(x * stride_ + kx) - static_cast<int>(padding_);
                            
                            double value = 0.0;
                            if (in_y >= 0 && in_y < static_cast<int>(height) && 
                                in_x >= 0 && in_x < static_cast<int>(width)) {
                                value = input({b, c, static_cast<size_t>(in_y), static_cast<size_t>(in_x)});
                            }
                            
                            col({col_idx, col_elem}) = value;
                            ++col_elem;
                        }
                    }
                }
                ++col_idx;
            }
        }
    }
    
    return col;
}

Tensor Conv2DLayer::col2im(const Tensor& col, size_t height, size_t width) const {
    // Convert column matrix back to image format (for backward pass)
    size_t batch_size = col.size(0) / (get_output_height() * get_output_width());
    
    Tensor image = Tensor::zeros({batch_size, input_channels_, height, width});
    
    size_t out_h = get_output_height();
    size_t out_w = get_output_width();
    
    size_t col_idx = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t y = 0; y < out_h; ++y) {
            for (size_t x = 0; x < out_w; ++x) {
                size_t col_elem = 0;
                
                for (size_t c = 0; c < input_channels_; ++c) {
                    for (size_t ky = 0; ky < kernel_size_; ++ky) {
                        for (size_t kx = 0; kx < kernel_size_; ++kx) {
                            int in_y = static_cast<int>(y * stride_ + ky) - static_cast<int>(padding_);
                            int in_x = static_cast<int>(x * stride_ + kx) - static_cast<int>(padding_);
                            
                            if (in_y >= 0 && in_y < static_cast<int>(height) && 
                                in_x >= 0 && in_x < static_cast<int>(width)) {
                                image({b, c, static_cast<size_t>(in_y), static_cast<size_t>(in_x)}) += 
                                    col({col_idx, col_elem});
                            }
                            ++col_elem;
                        }
                    }
                }
                ++col_idx;
            }
        }
    }
    
    return image;
}

Tensor Conv2DLayer::apply_convolution(const Tensor& input) const {
    // Use im2col to convert convolution to matrix multiplication
    Tensor col = im2col(input);
    
    // Reshape weights for matrix multiplication
    // Weights: [output_channels, input_channels, kernel_h, kernel_w]
    // -> [output_channels, input_channels * kernel_h * kernel_w]
    size_t weight_rows = output_channels_;
    size_t weight_cols = input_channels_ * kernel_size_ * kernel_size_;
    Tensor weights_2d = weights_.reshape({weight_rows, weight_cols});
    
    // Perform matrix multiplication: weights_2d @ col.T
    Tensor col_t = col.transpose();
    Tensor output_2d = weights_2d.matmul(col_t);
    
    // Add bias
    for (size_t i = 0; i < output_channels_; ++i) {
        for (size_t j = 0; j < output_2d.size(1); ++j) {
            output_2d({i, j}) += biases_({i});
        }
    }
    
    // Reshape output back to 4D tensor
    size_t batch_size = input.size(0);
    size_t out_h = get_output_height();
    size_t out_w = get_output_width();
    
    Tensor output = output_2d.transpose().reshape({batch_size, output_channels_, out_h, out_w});
    
    return output;
}

Tensor Conv2DLayer::forward_tensor(const Tensor& input) {
    validate_input_tensor(input);
    
    // Store input for backward pass
    last_input_ = input;
    
    // Set input dimensions if not set
    if (input_height_ == 0 || input_width_ == 0) {
        set_input_dimensions(input.size(2), input.size(3));
    }
    
    return apply_convolution(input);
}

Tensor Conv2DLayer::backward_tensor(const Tensor& gradient) {
    // Compute gradients for weights, biases, and input
    
    // Gradient w.r.t. bias: sum over all spatial locations and batch
    for (size_t c = 0; c < output_channels_; ++c) {
        double bias_grad = 0.0;
        for (size_t b = 0; b < gradient.size(0); ++b) {
            for (size_t h = 0; h < gradient.size(2); ++h) {
                for (size_t w = 0; w < gradient.size(3); ++w) {
                    bias_grad += gradient({b, c, h, w});
                }
            }
        }
        bias_gradients_({c}) = bias_grad;
    }
    
    // Gradient w.r.t. weights using im2col
    Tensor input_col = im2col(last_input_);
    
    // Reshape gradient for matrix operations
    size_t batch_size = gradient.size(0);
    size_t out_h = gradient.size(2);
    size_t out_w = gradient.size(3);
    
    Tensor grad_2d = gradient.reshape({batch_size * out_h * out_w, output_channels_});
    Tensor grad_2d_t = grad_2d.transpose();
    
    // Weight gradients: grad_2d_t @ input_col
    Tensor weight_grad_2d = grad_2d_t.matmul(input_col);
    weight_gradients_ = weight_grad_2d.reshape({output_channels_, input_channels_, kernel_size_, kernel_size_});
    
    // Gradient w.r.t. input
    Tensor weights_2d = weights_.reshape({output_channels_, input_channels_ * kernel_size_ * kernel_size_});
    Tensor weights_2d_t = weights_2d.transpose();
    Tensor input_grad_col = weights_2d_t.matmul(grad_2d_t);
    Tensor input_grad_col_t = input_grad_col.transpose();
    
    // Convert back to image format
    Tensor input_gradient = col2im(input_grad_col_t, last_input_.size(2), last_input_.size(3));
    
    return input_gradient;
}

Matrix Conv2DLayer::forward(const Matrix& input) {
    // Convert Matrix to Tensor, perform forward pass, convert back
    // Assume input is flattened from [channels, height, width]
    
    if (!weights_initialized_) {
        throw std::runtime_error("Conv2D layer dimensions not set. Call set_input_dimensions first.");
    }
    
    // For now, assume single batch and known dimensions
    // This is a simplified implementation for backward compatibility
    size_t total_input_size = input_channels_ * input_height_ * input_width_;
    if (input.cols() != total_input_size) {
        throw std::invalid_argument("Input size mismatch in Conv2D forward pass");
    }
    
    // Convert to tensor (batch size = 1)
    Tensor input_tensor({1, input_channels_, input_height_, input_width_});
    for (size_t i = 0; i < total_input_size; ++i) {
        size_t c = i / (input_height_ * input_width_);
        size_t remaining = i % (input_height_ * input_width_);
        size_t h = remaining / input_width_;
        size_t w = remaining % input_width_;
        input_tensor({0, c, h, w}) = input(0, i);
    }
    
    Tensor output_tensor = forward_tensor(input_tensor);
    
    // Convert back to Matrix
    size_t total_output_size = output_tensor.size(1) * output_tensor.size(2) * output_tensor.size(3);
    Matrix output(1, total_output_size);
    
    for (size_t i = 0; i < total_output_size; ++i) {
        size_t c = i / (output_tensor.size(2) * output_tensor.size(3));
        size_t remaining = i % (output_tensor.size(2) * output_tensor.size(3));
        size_t h = remaining / output_tensor.size(3);
        size_t w = remaining % output_tensor.size(3);
        output(0, i) = output_tensor({0, c, h, w});
    }
    
    return output;
}

Matrix Conv2DLayer::backward(const Matrix& gradient) {
    // Convert Matrix gradient to Tensor, perform backward pass, convert back
    size_t total_output_size = output_channels_ * get_output_height() * get_output_width();
    
    if (gradient.cols() != total_output_size) {
        throw std::invalid_argument("Gradient size mismatch in Conv2D backward pass");
    }
    
    // Convert to tensor
    Tensor grad_tensor({1, output_channels_, get_output_height(), get_output_width()});
    for (size_t i = 0; i < total_output_size; ++i) {
        size_t c = i / (get_output_height() * get_output_width());
        size_t remaining = i % (get_output_height() * get_output_width());
        size_t h = remaining / get_output_width();
        size_t w = remaining % get_output_width();
        grad_tensor({0, c, h, w}) = gradient(0, i);
    }
    
    Tensor input_grad_tensor = backward_tensor(grad_tensor);
    
    // Convert back to Matrix
    size_t total_input_size = input_channels_ * input_height_ * input_width_;
    Matrix input_gradient(1, total_input_size);
    
    for (size_t i = 0; i < total_input_size; ++i) {
        size_t c = i / (input_height_ * input_width_);
        size_t remaining = i % (input_height_ * input_width_);
        size_t h = remaining / input_width_;
        size_t w = remaining % input_width_;
        input_gradient(0, i) = input_grad_tensor({0, c, h, w});
    }
    
    return input_gradient;
}

void Conv2DLayer::update_weights(double learning_rate) {
    // Update weights and biases using computed gradients
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_.data()[i] -= learning_rate * weight_gradients_.data()[i];
    }
    
    for (size_t i = 0; i < biases_.size(); ++i) {
        biases_.data()[i] -= learning_rate * bias_gradients_.data()[i];
    }
}

size_t Conv2DLayer::input_size() const {
    return input_channels_ * input_height_ * input_width_;
}

size_t Conv2DLayer::output_size() const {
    return output_channels_ * get_output_height() * get_output_width();
}

std::unique_ptr<Layer> Conv2DLayer::clone() const {
    auto cloned = std::make_unique<Conv2DLayer>(input_channels_, output_channels_, 
                                               kernel_size_, stride_, padding_);
    cloned->input_height_ = input_height_;
    cloned->input_width_ = input_width_;
    if (weights_initialized_) {
        cloned->weights_ = weights_;
        cloned->biases_ = biases_;
        cloned->weights_initialized_ = true;
    }
    return cloned;
}

std::string Conv2DLayer::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"Conv2D\","
        << "\"input_channels\":" << input_channels_ << ","
        << "\"output_channels\":" << output_channels_ << ","
        << "\"kernel_size\":" << kernel_size_ << ","
        << "\"stride\":" << stride_ << ","
        << "\"padding\":" << padding_ << ","
        << "\"input_height\":" << input_height_ << ","
        << "\"input_width\":" << input_width_
        << "}";
    return oss.str();
}

void Conv2DLayer::serialize_weights(std::ofstream& file) const {
    // Write weights and biases in binary format
    size_t weights_size = weights_.size();
    size_t biases_size = biases_.size();
    
    file.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));
    file.write(reinterpret_cast<const char*>(weights_.raw_data()), weights_size * sizeof(double));
    
    file.write(reinterpret_cast<const char*>(&biases_size), sizeof(biases_size));
    file.write(reinterpret_cast<const char*>(biases_.raw_data()), biases_size * sizeof(double));
}

void Conv2DLayer::deserialize_weights(std::ifstream& file) {
    size_t weights_size, biases_size;
    
    file.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
    if (weights_size != weights_.size()) {
        throw std::runtime_error("Weight size mismatch during deserialization");
    }
    file.read(reinterpret_cast<char*>(weights_.raw_data()), weights_size * sizeof(double));
    
    file.read(reinterpret_cast<char*>(&biases_size), sizeof(biases_size));
    if (biases_size != biases_.size()) {
        throw std::runtime_error("Bias size mismatch during deserialization");
    }
    file.read(reinterpret_cast<char*>(biases_.raw_data()), biases_size * sizeof(double));
}

size_t Conv2DLayer::get_weights_size() const {
    return weights_.size() + biases_.size();
}

// ================================================================================================
// MaxPool2DLayer Implementation
// ================================================================================================

MaxPool2DLayer::MaxPool2DLayer(size_t kernel_size, size_t stride, size_t padding)
    : kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)  // Default stride = kernel_size
    , padding_(padding)
    , input_channels_(0)
    , input_height_(0)
    , input_width_(0) {
}

void MaxPool2DLayer::set_input_dimensions(size_t channels, size_t height, size_t width) {
    input_channels_ = channels;
    input_height_ = height;
    input_width_ = width;
}

size_t MaxPool2DLayer::calculate_output_size(size_t input_size, size_t kernel_size, 
                                           size_t stride, size_t padding) const {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

size_t MaxPool2DLayer::get_output_height() const {
    if (input_height_ == 0) return 0;
    return calculate_output_size(input_height_, kernel_size_, stride_, padding_);
}

size_t MaxPool2DLayer::get_output_width() const {
    if (input_width_ == 0) return 0;
    return calculate_output_size(input_width_, kernel_size_, stride_, padding_);
}

void MaxPool2DLayer::validate_input_tensor(const Tensor& input) const {
    if (input.ndim() != 4) {
        throw std::invalid_argument("MaxPool2D input must be 4D: [batch, channels, height, width]");
    }
}

Tensor MaxPool2DLayer::forward_tensor(const Tensor& input) {
    validate_input_tensor(input);
    
    last_input_ = input;
    
    if (input_channels_ == 0) {
        set_input_dimensions(input.size(1), input.size(2), input.size(3));
    }
    
    size_t batch_size = input.size(0);
    size_t channels = input.size(1);
    size_t height = input.size(2);
    size_t width = input.size(3);
    
    size_t out_h = get_output_height();
    size_t out_w = get_output_width();
    
    Tensor output({batch_size, channels, out_h, out_w});
    
    // Store indices of max values for backward pass
    max_indices_ = Tensor::zeros({batch_size, channels, out_h, out_w});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t y = 0; y < out_h; ++y) {
                for (size_t x = 0; x < out_w; ++x) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t max_idx = 0;
                    
                    for (size_t ky = 0; ky < kernel_size_; ++ky) {
                        for (size_t kx = 0; kx < kernel_size_; ++kx) {
                            int in_y = static_cast<int>(y * stride_ + ky) - static_cast<int>(padding_);
                            int in_x = static_cast<int>(x * stride_ + kx) - static_cast<int>(padding_);
                            
                            if (in_y >= 0 && in_y < static_cast<int>(height) && 
                                in_x >= 0 && in_x < static_cast<int>(width)) {
                                double val = input({b, c, static_cast<size_t>(in_y), static_cast<size_t>(in_x)});
                                if (val > max_val) {
                                    max_val = val;
                                    // Store flattened index for backward pass
                                    max_idx = static_cast<size_t>(in_y) * width + static_cast<size_t>(in_x);
                                }
                            }
                        }
                    }
                    
                    output({b, c, y, x}) = max_val;
                    max_indices_({b, c, y, x}) = static_cast<double>(max_idx);
                }
            }
        }
    }
    
    return output;
}

Tensor MaxPool2DLayer::backward_tensor(const Tensor& gradient) {
    size_t batch_size = last_input_.size(0);
    size_t channels = last_input_.size(1);
    size_t height = last_input_.size(2);
    size_t width = last_input_.size(3);
    
    Tensor input_gradient = Tensor::zeros({batch_size, channels, height, width});
    
    size_t out_h = gradient.size(2);
    size_t out_w = gradient.size(3);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t y = 0; y < out_h; ++y) {
                for (size_t x = 0; x < out_w; ++x) {
                    // Get the index where the max value came from
                    size_t max_idx = static_cast<size_t>(max_indices_({b, c, y, x}));
                    size_t max_h = max_idx / width;
                    size_t max_w = max_idx % width;
                    
                    // Add gradient to the position that produced the max
                    input_gradient({b, c, max_h, max_w}) += gradient({b, c, y, x});
                }
            }
        }
    }
    
    return input_gradient;
}

Matrix MaxPool2DLayer::forward(const Matrix& input) {
    // Simplified implementation for backward compatibility
    if (input_channels_ == 0) {
        throw std::runtime_error("MaxPool2D layer dimensions not set. Call set_input_dimensions first.");
    }
    
    size_t total_input_size = input_channels_ * input_height_ * input_width_;
    if (input.cols() != total_input_size) {
        throw std::invalid_argument("Input size mismatch in MaxPool2D forward pass");
    }
    
    // Convert to tensor
    Tensor input_tensor({1, input_channels_, input_height_, input_width_});
    for (size_t i = 0; i < total_input_size; ++i) {
        size_t c = i / (input_height_ * input_width_);
        size_t remaining = i % (input_height_ * input_width_);
        size_t h = remaining / input_width_;
        size_t w = remaining % input_width_;
        input_tensor({0, c, h, w}) = input(0, i);
    }
    
    Tensor output_tensor = forward_tensor(input_tensor);
    
    // Convert back to Matrix
    size_t total_output_size = output_tensor.size(1) * output_tensor.size(2) * output_tensor.size(3);
    Matrix output(1, total_output_size);
    
    for (size_t i = 0; i < total_output_size; ++i) {
        size_t c = i / (output_tensor.size(2) * output_tensor.size(3));
        size_t remaining = i % (output_tensor.size(2) * output_tensor.size(3));
        size_t h = remaining / output_tensor.size(3);
        size_t w = remaining % output_tensor.size(3);
        output(0, i) = output_tensor({0, c, h, w});
    }
    
    return output;
}

Matrix MaxPool2DLayer::backward(const Matrix& gradient) {
    size_t total_output_size = input_channels_ * get_output_height() * get_output_width();
    
    if (gradient.cols() != total_output_size) {
        throw std::invalid_argument("Gradient size mismatch in MaxPool2D backward pass");
    }
    
    // Convert to tensor
    Tensor grad_tensor({1, input_channels_, get_output_height(), get_output_width()});
    for (size_t i = 0; i < total_output_size; ++i) {
        size_t c = i / (get_output_height() * get_output_width());
        size_t remaining = i % (get_output_height() * get_output_width());
        size_t h = remaining / get_output_width();
        size_t w = remaining % get_output_width();
        grad_tensor({0, c, h, w}) = gradient(0, i);
    }
    
    Tensor input_grad_tensor = backward_tensor(grad_tensor);
    
    // Convert back to Matrix
    size_t total_input_size = input_channels_ * input_height_ * input_width_;
    Matrix input_gradient(1, total_input_size);
    
    for (size_t i = 0; i < total_input_size; ++i) {
        size_t c = i / (input_height_ * input_width_);
        size_t remaining = i % (input_height_ * input_width_);
        size_t h = remaining / input_width_;
        size_t w = remaining % input_width_;
        input_gradient(0, i) = input_grad_tensor({0, c, h, w});
    }
    
    return input_gradient;
}

size_t MaxPool2DLayer::input_size() const {
    return input_channels_ * input_height_ * input_width_;
}

size_t MaxPool2DLayer::output_size() const {
    return input_channels_ * get_output_height() * get_output_width();
}

std::unique_ptr<Layer> MaxPool2DLayer::clone() const {
    auto cloned = std::make_unique<MaxPool2DLayer>(kernel_size_, stride_, padding_);
    cloned->input_channels_ = input_channels_;
    cloned->input_height_ = input_height_;
    cloned->input_width_ = input_width_;
    return cloned;
}

std::string MaxPool2DLayer::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"MaxPool2D\","
        << "\"kernel_size\":" << kernel_size_ << ","
        << "\"stride\":" << stride_ << ","
        << "\"padding\":" << padding_ << ","
        << "\"input_channels\":" << input_channels_ << ","
        << "\"input_height\":" << input_height_ << ","
        << "\"input_width\":" << input_width_
        << "}";
    return oss.str();
}

// ================================================================================================
// AvgPool2DLayer Implementation
// ================================================================================================

AvgPool2DLayer::AvgPool2DLayer(size_t kernel_size, size_t stride, size_t padding)
    : kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)
    , padding_(padding)
    , input_channels_(0)
    , input_height_(0)
    , input_width_(0) {
}

void AvgPool2DLayer::set_input_dimensions(size_t channels, size_t height, size_t width) {
    input_channels_ = channels;
    input_height_ = height;
    input_width_ = width;
}

size_t AvgPool2DLayer::calculate_output_size(size_t input_size, size_t kernel_size, 
                                           size_t stride, size_t padding) const {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

size_t AvgPool2DLayer::get_output_height() const {
    if (input_height_ == 0) return 0;
    return calculate_output_size(input_height_, kernel_size_, stride_, padding_);
}

size_t AvgPool2DLayer::get_output_width() const {
    if (input_width_ == 0) return 0;
    return calculate_output_size(input_width_, kernel_size_, stride_, padding_);
}

void AvgPool2DLayer::validate_input_tensor(const Tensor& input) const {
    if (input.ndim() != 4) {
        throw std::invalid_argument("AvgPool2D input must be 4D: [batch, channels, height, width]");
    }
}

Tensor AvgPool2DLayer::forward_tensor(const Tensor& input) {
    validate_input_tensor(input);
    
    last_input_ = input;
    
    if (input_channels_ == 0) {
        set_input_dimensions(input.size(1), input.size(2), input.size(3));
    }
    
    size_t batch_size = input.size(0);
    size_t channels = input.size(1);
    size_t height = input.size(2);
    size_t width = input.size(3);
    
    size_t out_h = get_output_height();
    size_t out_w = get_output_width();
    
    Tensor output({batch_size, channels, out_h, out_w});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t y = 0; y < out_h; ++y) {
                for (size_t x = 0; x < out_w; ++x) {
                    double sum = 0.0;
                    size_t count = 0;
                    
                    for (size_t ky = 0; ky < kernel_size_; ++ky) {
                        for (size_t kx = 0; kx < kernel_size_; ++kx) {
                            int in_y = static_cast<int>(y * stride_ + ky) - static_cast<int>(padding_);
                            int in_x = static_cast<int>(x * stride_ + kx) - static_cast<int>(padding_);
                            
                            if (in_y >= 0 && in_y < static_cast<int>(height) && 
                                in_x >= 0 && in_x < static_cast<int>(width)) {
                                sum += input({b, c, static_cast<size_t>(in_y), static_cast<size_t>(in_x)});
                                ++count;
                            }
                        }
                    }
                    
                    output({b, c, y, x}) = (count > 0) ? sum / count : 0.0;
                }
            }
        }
    }
    
    return output;
}

Tensor AvgPool2DLayer::backward_tensor(const Tensor& gradient) {
    size_t batch_size = last_input_.size(0);
    size_t channels = last_input_.size(1);
    size_t height = last_input_.size(2);
    size_t width = last_input_.size(3);
    
    Tensor input_gradient = Tensor::zeros({batch_size, channels, height, width});
    
    size_t out_h = gradient.size(2);
    size_t out_w = gradient.size(3);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t y = 0; y < out_h; ++y) {
                for (size_t x = 0; x < out_w; ++x) {
                    // Count valid positions in the pooling window
                    size_t count = 0;
                    for (size_t ky = 0; ky < kernel_size_; ++ky) {
                        for (size_t kx = 0; kx < kernel_size_; ++kx) {
                            int in_y = static_cast<int>(y * stride_ + ky) - static_cast<int>(padding_);
                            int in_x = static_cast<int>(x * stride_ + kx) - static_cast<int>(padding_);
                            
                            if (in_y >= 0 && in_y < static_cast<int>(height) && 
                                in_x >= 0 && in_x < static_cast<int>(width)) {
                                ++count;
                            }
                        }
                    }
                    
                    // Distribute gradient equally among all positions
                    double grad_contribution = (count > 0) ? gradient({b, c, y, x}) / count : 0.0;
                    
                    for (size_t ky = 0; ky < kernel_size_; ++ky) {
                        for (size_t kx = 0; kx < kernel_size_; ++kx) {
                            int in_y = static_cast<int>(y * stride_ + ky) - static_cast<int>(padding_);
                            int in_x = static_cast<int>(x * stride_ + kx) - static_cast<int>(padding_);
                            
                            if (in_y >= 0 && in_y < static_cast<int>(height) && 
                                in_x >= 0 && in_x < static_cast<int>(width)) {
                                input_gradient({b, c, static_cast<size_t>(in_y), static_cast<size_t>(in_x)}) += grad_contribution;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return input_gradient;
}

Matrix AvgPool2DLayer::forward(const Matrix& input) {
    // Similar implementation as MaxPool2D but with averaging
    if (input_channels_ == 0) {
        throw std::runtime_error("AvgPool2D layer dimensions not set. Call set_input_dimensions first.");
    }
    
    size_t total_input_size = input_channels_ * input_height_ * input_width_;
    if (input.cols() != total_input_size) {
        throw std::invalid_argument("Input size mismatch in AvgPool2D forward pass");
    }
    
    // Convert to tensor
    Tensor input_tensor({1, input_channels_, input_height_, input_width_});
    for (size_t i = 0; i < total_input_size; ++i) {
        size_t c = i / (input_height_ * input_width_);
        size_t remaining = i % (input_height_ * input_width_);
        size_t h = remaining / input_width_;
        size_t w = remaining % input_width_;
        input_tensor({0, c, h, w}) = input(0, i);
    }
    
    Tensor output_tensor = forward_tensor(input_tensor);
    
    // Convert back to Matrix
    size_t total_output_size = output_tensor.size(1) * output_tensor.size(2) * output_tensor.size(3);
    Matrix output(1, total_output_size);
    
    for (size_t i = 0; i < total_output_size; ++i) {
        size_t c = i / (output_tensor.size(2) * output_tensor.size(3));
        size_t remaining = i % (output_tensor.size(2) * output_tensor.size(3));
        size_t h = remaining / output_tensor.size(3);
        size_t w = remaining % output_tensor.size(3);
        output(0, i) = output_tensor({0, c, h, w});
    }
    
    return output;
}

Matrix AvgPool2DLayer::backward(const Matrix& gradient) {
    // Similar to MaxPool2D backward implementation
    size_t total_output_size = input_channels_ * get_output_height() * get_output_width();
    
    if (gradient.cols() != total_output_size) {
        throw std::invalid_argument("Gradient size mismatch in AvgPool2D backward pass");
    }
    
    // Convert to tensor
    Tensor grad_tensor({1, input_channels_, get_output_height(), get_output_width()});
    for (size_t i = 0; i < total_output_size; ++i) {
        size_t c = i / (get_output_height() * get_output_width());
        size_t remaining = i % (get_output_height() * get_output_width());
        size_t h = remaining / get_output_width();
        size_t w = remaining % get_output_width();
        grad_tensor({0, c, h, w}) = gradient(0, i);
    }
    
    Tensor input_grad_tensor = backward_tensor(grad_tensor);
    
    // Convert back to Matrix
    size_t total_input_size = input_channels_ * input_height_ * input_width_;
    Matrix input_gradient(1, total_input_size);
    
    for (size_t i = 0; i < total_input_size; ++i) {
        size_t c = i / (input_height_ * input_width_);
        size_t remaining = i % (input_height_ * input_width_);
        size_t h = remaining / input_width_;
        size_t w = remaining % input_width_;
        input_gradient(0, i) = input_grad_tensor({0, c, h, w});
    }
    
    return input_gradient;
}

size_t AvgPool2DLayer::input_size() const {
    return input_channels_ * input_height_ * input_width_;
}

size_t AvgPool2DLayer::output_size() const {
    return input_channels_ * get_output_height() * get_output_width();
}

std::unique_ptr<Layer> AvgPool2DLayer::clone() const {
    auto cloned = std::make_unique<AvgPool2DLayer>(kernel_size_, stride_, padding_);
    cloned->input_channels_ = input_channels_;
    cloned->input_height_ = input_height_;
    cloned->input_width_ = input_width_;
    return cloned;
}

std::string AvgPool2DLayer::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"AvgPool2D\","
        << "\"kernel_size\":" << kernel_size_ << ","
        << "\"stride\":" << stride_ << ","
        << "\"padding\":" << padding_ << ","
        << "\"input_channels\":" << input_channels_ << ","
        << "\"input_height\":" << input_height_ << ","
        << "\"input_width\":" << input_width_
        << "}";
    return oss.str();
}

// ================================================================================================
// FlattenLayer Implementation
// ================================================================================================

Tensor FlattenLayer::forward_tensor(const Tensor& input) {
    last_input_ = input;
    input_shape_ = input.shape();
    
    // Flatten to 2D: [batch_size, flattened_features]
    size_t batch_size = (input.ndim() > 0) ? input.size(0) : 1;
    size_t flattened_size = input.size() / batch_size;
    
    return input.reshape({batch_size, flattened_size});
}

Tensor FlattenLayer::backward_tensor(const Tensor& gradient) {
    // Reshape gradient back to original input shape
    return gradient.reshape(input_shape_);
}

Matrix FlattenLayer::forward(const Matrix& input) {
    // For Matrix input, just return as-is (already flattened)
    return input;
}

Matrix FlattenLayer::backward(const Matrix& gradient) {
    // For Matrix gradient, just return as-is
    return gradient;
}

size_t FlattenLayer::input_size() const {
    if (input_shape_.empty()) return 0;
    
    size_t size = 1;
    for (size_t i = 1; i < input_shape_.size(); ++i) {  // Skip batch dimension
        size *= input_shape_[i];
    }
    return size;
}

size_t FlattenLayer::output_size() const {
    return input_size();  // Same as input size (just reshaped)
}

std::unique_ptr<Layer> FlattenLayer::clone() const {
    auto cloned = std::make_unique<FlattenLayer>();
    cloned->input_shape_ = input_shape_;
    return cloned;
}

std::string FlattenLayer::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"Flatten\"";
    
    if (!input_shape_.empty()) {
        oss << ",\"input_shape\":[";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            if (i > 0) oss << ",";
            oss << input_shape_[i];
        }
        oss << "]";
    }
    
    oss << "}";
    return oss.str();
}

} // namespace ai
} // namespace asekioml
