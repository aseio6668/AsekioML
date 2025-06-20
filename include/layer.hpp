#pragma once

#include "matrix.hpp"
#include "activation.hpp"
#include <memory>

namespace asekioml {

// Abstract base class for neural network layers
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient) = 0;
    virtual void update_weights(double /* learning_rate */) {}
    virtual std::unique_ptr<Layer> clone() const = 0;
    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;
    virtual std::string type() const = 0;
    
    // Serialization interface
    virtual std::string serialize_to_json() const = 0;
    virtual void serialize_weights(std::ofstream& file) const {}
    virtual void deserialize_weights(std::ifstream& file) {}
    virtual size_t get_weights_size() const { return 0; }
};

// Dense (Fully Connected) Layer
class DenseLayer : public Layer {
private:
    Matrix weights_;
    Matrix biases_;
    Matrix last_input_;
    Matrix weight_gradients_;
    Matrix bias_gradients_;
    size_t input_size_;
    size_t output_size_;

public:
    DenseLayer(size_t input_size, size_t output_size);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return input_size_; }
    size_t output_size() const override { return output_size_; }
    std::string type() const override { return "Dense"; }
    
    // Getters for weights and biases (for inspection/debugging)
    const Matrix& weights() const { return weights_; }
    const Matrix& biases() const { return biases_; }
      // Initialize weights with different strategies
    void initialize_xavier();
    void initialize_he();
    void initialize_random(double min = -1.0, double max = 1.0);
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

// Activation Layer
class ActivationLayer : public Layer {
private:
    std::unique_ptr<ActivationFunction> activation_func_;
    Matrix last_input_;
    size_t size_;

public:
    explicit ActivationLayer(std::unique_ptr<ActivationFunction> activation_func, size_t size);
    explicit ActivationLayer(const std::string& activation_name, size_t size);
    
    // Copy constructor and assignment operator
    ActivationLayer(const ActivationLayer& other);
    ActivationLayer& operator=(const ActivationLayer& other);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return size_; }
    size_t output_size() const override { return size_; }
    std::string type() const override { return "Activation_" + activation_func_->name(); }
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

// Dropout Layer is now defined in regularization.hpp

} // namespace asekioml
