#pragma once

#include "../tensor.hpp"
#include "../layer.hpp"
#include <memory>

namespace clmodel {
namespace ai {

/**
 * @brief 2D Convolutional layer for image processing
 */
class Conv2DLayer : public Layer {
private:
    size_t input_channels_;
    size_t output_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t input_height_;
    size_t input_width_;
    
    // Weights: [output_channels, input_channels, kernel_size, kernel_size]
    Tensor weights_;
    Tensor biases_;
    Tensor last_input_;
    Tensor weight_gradients_;
    Tensor bias_gradients_;
    
    bool weights_initialized_;
    
public:
    Conv2DLayer(size_t input_channels, size_t output_channels, 
                size_t kernel_size, size_t stride = 1, size_t padding = 0);
    
    // Set input dimensions (required before first forward pass)
    void set_input_dimensions(size_t height, size_t width);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "Conv2D"; }
    
    // Tensor-based interface (preferred for CNN operations)
    Tensor forward_tensor(const Tensor& input);
    Tensor backward_tensor(const Tensor& gradient);
    
    // Weight initialization
    void initialize_weights(const std::string& method = "xavier");
    
    // Getters for inspection
    const Tensor& weights() const { return weights_; }
    const Tensor& biases() const { return biases_; }
    size_t get_output_height() const;
    size_t get_output_width() const;
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    void validate_input_tensor(const Tensor& input) const;
    Tensor apply_convolution(const Tensor& input) const;
    size_t calculate_output_size(size_t input_size, size_t kernel_size, 
                                size_t stride, size_t padding) const;
    
    // Helper methods for convolution
    Tensor im2col(const Tensor& input) const;
    Tensor col2im(const Tensor& col, size_t height, size_t width) const;
};

/**
 * @brief 2D Max Pooling layer
 */
class MaxPool2DLayer : public Layer {
private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t input_channels_;
    size_t input_height_;
    size_t input_width_;
    
    Tensor last_input_;
    Tensor max_indices_; // For backward pass
    
public:
    MaxPool2DLayer(size_t kernel_size, size_t stride = 0, size_t padding = 0);
    
    void set_input_dimensions(size_t channels, size_t height, size_t width);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override {} // No weights to update
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "MaxPool2D"; }
    
    // Tensor-based interface
    Tensor forward_tensor(const Tensor& input);
    Tensor backward_tensor(const Tensor& gradient);
    
    // Getters
    size_t get_output_height() const;
    size_t get_output_width() const;
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override {} // No weights
    void deserialize_weights(std::ifstream& file) override {} // No weights
    size_t get_weights_size() const override { return 0; }
    
private:
    void validate_input_tensor(const Tensor& input) const;
    size_t calculate_output_size(size_t input_size, size_t kernel_size, 
                                size_t stride, size_t padding) const;
};

/**
 * @brief 2D Average Pooling layer
 */
class AvgPool2DLayer : public Layer {
private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t input_channels_;
    size_t input_height_;
    size_t input_width_;
    
    Tensor last_input_;
    
public:
    AvgPool2DLayer(size_t kernel_size, size_t stride = 0, size_t padding = 0);
    
    void set_input_dimensions(size_t channels, size_t height, size_t width);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override {} // No weights to update
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "AvgPool2D"; }
    
    // Tensor-based interface
    Tensor forward_tensor(const Tensor& input);
    Tensor backward_tensor(const Tensor& gradient);
    
    // Getters
    size_t get_output_height() const;
    size_t get_output_width() const;
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override {} // No weights
    void deserialize_weights(std::ifstream& file) override {} // No weights
    size_t get_weights_size() const override { return 0; }
    
private:
    void validate_input_tensor(const Tensor& input) const;
    size_t calculate_output_size(size_t input_size, size_t kernel_size, 
                                size_t stride, size_t padding) const;
};

/**
 * @brief Flatten layer to convert multi-dimensional tensors to vectors
 */
class FlattenLayer : public Layer {
private:
    std::vector<size_t> input_shape_;
    Tensor last_input_;
    
public:
    FlattenLayer() = default;
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override {} // No weights to update
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "Flatten"; }
    
    // Tensor-based interface
    Tensor forward_tensor(const Tensor& input);
    Tensor backward_tensor(const Tensor& gradient);
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override {} // No weights
    void deserialize_weights(std::ifstream& file) override {} // No weights
    size_t get_weights_size() const override { return 0; }
};

} // namespace ai
} // namespace clmodel
