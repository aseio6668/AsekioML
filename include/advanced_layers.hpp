#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include <memory>
#include <string>

namespace asekioml {
namespace advanced {

/**
 * @brief Batch Normalization layer for normalizing activations
 * 
 * Applies batch normalization to improve training stability and speed.
 * Normalizes inputs to have zero mean and unit variance.
 */
class BatchNormalizationLayer : public Layer {
private:
    size_t size_;
    double momentum_;
    double epsilon_;
    bool training_;
    
    Matrix gamma_;        // Scale parameters
    Matrix beta_;         // Shift parameters
    Matrix running_mean_; // Running mean for inference
    Matrix running_var_;  // Running variance for inference
    Matrix last_input_;
    Matrix normalized_;   // Cached normalized values for backward pass
    
public:
    /**
     * @brief Constructor for BatchNormalizationLayer
     * @param size Number of features to normalize
     * @param momentum Momentum for updating running statistics (default: 0.9)
     * @param epsilon Small value for numerical stability (default: 1e-5)
     */
    BatchNormalizationLayer(size_t size, double momentum = 0.9, double epsilon = 1e-5);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return size_; }
    size_t output_size() const override { return size_; }
    std::string type() const override { return "BatchNormalization"; }
    
    // Set training mode
    void set_training(bool training) { training_ = training; }
    bool is_training() const { return training_; }
    
    // Serialization methods
    std::string serialize_to_json() const override { return "{}"; }
    void serialize_weights(std::ofstream& file) const override {}
    void deserialize_weights(std::ifstream& file) override {}
    size_t get_weights_size() const override { return size_ * 4; } // gamma, beta, mean, var
};

/**
 * @brief 2D Convolutional layer (alternative implementation)
 * 
 * Note: This is a different implementation from the one in cnn_layers.hpp
 * This version uses a different constructor signature and internal structure.
 */
class Conv2DLayer : public Layer {
private:
    size_t input_height_;
    size_t input_width_;
    size_t input_channels_;
    size_t num_filters_;
    size_t kernel_height_;
    size_t kernel_width_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
    size_t output_height_;
    size_t output_width_;
    
    Matrix kernels_;     // Filter weights
    Matrix biases_;      // Filter biases
    Matrix last_input_;  // Cached input for backward pass
    
public:
    /**
     * @brief Constructor for Conv2DLayer
     * @param input_height Height of input feature maps
     * @param input_width Width of input feature maps
     * @param input_channels Number of input channels
     * @param num_filters Number of output filters/channels
     * @param kernel_height Height of convolution kernels
     * @param kernel_width Width of convolution kernels
     * @param stride_h Vertical stride (default: 1)
     * @param stride_w Horizontal stride (default: 1)
     * @param padding_h Vertical padding (default: 0)
     * @param padding_w Horizontal padding (default: 0)
     */
    Conv2DLayer(size_t input_height, size_t input_width, size_t input_channels,
                size_t num_filters, size_t kernel_height, size_t kernel_width,
                size_t stride_h = 1, size_t stride_w = 1, 
                size_t padding_h = 0, size_t padding_w = 0);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "Conv2D"; }
    
    // Getters
    size_t get_output_height() const { return output_height_; }
    size_t get_output_width() const { return output_width_; }
    size_t get_num_filters() const { return num_filters_; }
    
    // Serialization methods
    std::string serialize_to_json() const override { return "{}"; }
    void serialize_weights(std::ofstream& file) const override {}
    void deserialize_weights(std::ifstream& file) override {}
    size_t get_weights_size() const override { 
        return kernels_.rows() * kernels_.cols() + biases_.rows() * biases_.cols(); 
    }
};

/**
 * @brief L1/L2 Regularization layer
 * 
 * Applies L1 (Lasso) and/or L2 (Ridge) regularization to prevent overfitting.
 * This layer doesn't change the forward pass but adds regularization terms 
 * to the gradients during backward pass.
 */
class RegularizationLayer : public Layer {
private:
    size_t size_;
    double l1_lambda_;    // L1 regularization strength
    double l2_lambda_;    // L2 regularization strength
    Matrix last_input_;   // Cached input for regularization computation
    
public:
    /**
     * @brief Constructor for RegularizationLayer
     * @param size Size of the input/output
     * @param l1_lambda L1 regularization coefficient (default: 0.0)
     * @param l2_lambda L2 regularization coefficient (default: 0.0)
     */
    RegularizationLayer(size_t size, double l1_lambda = 0.0, double l2_lambda = 0.0);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override {}  // No weights to update
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return size_; }
    size_t output_size() const override { return size_; }
    std::string type() const override;
    
    // Regularization parameter setters/getters
    void set_l1_lambda(double l1_lambda) { l1_lambda_ = l1_lambda; }
    void set_l2_lambda(double l2_lambda) { l2_lambda_ = l2_lambda; }
    double get_l1_lambda() const { return l1_lambda_; }
    double get_l2_lambda() const { return l2_lambda_; }
    
    // Serialization methods
    std::string serialize_to_json() const override { return "{}"; }
    void serialize_weights(std::ofstream& file) const override {}
    void deserialize_weights(std::ifstream& file) override {}
    size_t get_weights_size() const override { return 0; } // No weights
};

/**
 * @brief Long Short-Term Memory (LSTM) layer
 * 
 * Implements LSTM recurrent neural network layer for sequence processing.
 * Includes forget gate, input gate, candidate values, and output gate.
 */
class LSTMLayer : public Layer {
private:
    size_t input_size_;
    size_t hidden_size_;
    size_t sequence_length_;
    
    // LSTM gate weight matrices
    Matrix W_f_, U_f_, b_f_;  // Forget gate
    Matrix W_i_, U_i_, b_i_;  // Input gate
    Matrix W_c_, U_c_, b_c_;  // Candidate values
    Matrix W_o_, U_o_, b_o_;  // Output gate
    
    // LSTM states
    Matrix cell_state_;      // Cell state (C_t)
    Matrix hidden_state_;    // Hidden state (h_t)
    Matrix last_input_;      // Cached input for backward pass
    
    // Helper methods for gate computations
    Matrix sigmoid_gate(const Matrix& x, const Matrix& h, 
                       const Matrix& W, const Matrix& U, const Matrix& b);
    Matrix tanh_gate(const Matrix& x, const Matrix& h,
                    const Matrix& W, const Matrix& U, const Matrix& b);
    
public:
    /**
     * @brief Constructor for LSTMLayer
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     */
    LSTMLayer(size_t input_size, size_t hidden_size);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return input_size_; }
    size_t output_size() const override { return hidden_size_; }
    std::string type() const override { return "LSTM"; }
    
    // LSTM-specific methods
    void reset_state();
    void set_sequence_length(size_t length) { sequence_length_ = length; }
    
    // State access
    const Matrix& get_cell_state() const { return cell_state_; }
    const Matrix& get_hidden_state() const { return hidden_state_; }
    
    // Serialization methods
    std::string serialize_to_json() const override { return "{}"; }
    void serialize_weights(std::ofstream& file) const override {}
    void deserialize_weights(std::ifstream& file) override {}
    size_t get_weights_size() const override { 
        return (W_f_.rows() * W_f_.cols() + U_f_.rows() * U_f_.cols() + b_f_.rows()) * 4; 
    }
};

} // namespace advanced
} // namespace asekioml
