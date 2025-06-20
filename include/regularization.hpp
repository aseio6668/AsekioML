#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include <random>
#include <memory>

namespace clmodel {

/**
 * @brief Dropout layer for regularization
 * 
 * Dropout randomly sets a fraction of input units to 0 at each update during training,
 * which helps prevent overfitting. During inference, all units are kept and outputs
 * are scaled by the keep probability.
 */
class DropoutLayer : public Layer {
private:
    float dropout_rate_;      // Probability of dropping a unit (0.0 to 1.0)
    bool training_mode_;      // Whether the layer is in training or inference mode
    size_t input_size_;       // Size of input features
    Matrix dropout_mask_;     // Binary mask for dropped units
    mutable std::mt19937 rng_;     // Random number generator
    mutable std::uniform_real_distribution<float> dist_;

public:
    /**
     * @brief Construct a new Dropout Layer
     * @param dropout_rate Probability of dropping a unit (0.0 to 1.0)
     */
    explicit DropoutLayer(float dropout_rate = 0.5f);
    
    /**
     * @brief Set training mode
     * @param training True for training mode, false for inference
     */
    void set_training_mode(bool training);
    
    /**
     * @brief Get current training mode
     * @return True if in training mode, false if in inference mode
     */
    bool is_training() const { return training_mode_; }
    
    /**
     * @brief Get dropout rate
     * @return Current dropout rate
     */
    float get_dropout_rate() const { return dropout_rate_; }
      /**
     * @brief Set dropout rate
     * @param rate New dropout rate (0.0 to 1.0)
     */
    void set_dropout_rate(float rate);
    
    /**
     * @brief Set input size for the layer
     * @param size Input feature size
     */
    void set_input_size(size_t size) { input_size_ = size; }
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::string type() const override { return "Dropout"; }    size_t output_size() const override;
    size_t input_size() const override;
    std::unique_ptr<Layer> clone() const override;
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

/**
 * @brief Weight decay regularization for optimizers
 * 
 * Adds L2 regularization term to the loss function by penalizing large weights.
 * This is typically implemented as weight decay in the optimizer update rule.
 */
class WeightDecay {
private:
    float lambda_;  // Regularization strength

public:
    /**
     * @brief Construct weight decay regularizer
     * @param lambda Regularization strength (typically 1e-4 to 1e-2)
     */
    explicit WeightDecay(float lambda = 1e-4f) : lambda_(lambda) {}
    
    /**
     * @brief Apply weight decay to gradients
     * @param weights Current weights
     * @param gradients Gradients to modify
     */
    void apply(const Matrix& weights, Matrix& gradients) const;
    
    /**
     * @brief Get regularization strength
     * @return Current lambda value
     */
    float get_lambda() const { return lambda_; }
    
    /**
     * @brief Set regularization strength
     * @param lambda New lambda value
     */
    void set_lambda(float lambda) { lambda_ = lambda; }
    
    /**
     * @brief Compute L2 regularization loss
     * @param weights Model weights
     * @return Regularization loss contribution
     */
    float compute_loss(const Matrix& weights) const;
};

/**
 * @brief Batch Normalization layer
 * 
 * Normalizes inputs to have zero mean and unit variance, with learnable
 * scale and shift parameters. Maintains running statistics for inference.
 */
class BatchNormLayer : public Layer {
private:
    Matrix gamma_;           // Scale parameter (learnable)
    Matrix beta_;            // Shift parameter (learnable)
    Matrix gamma_grad_;      // Gradients for gamma
    Matrix beta_grad_;       // Gradients for beta
    
    Matrix running_mean_;    // Running mean for inference
    Matrix running_var_;     // Running variance for inference
    
    Matrix last_input_;      // Cached input for backward pass
    Matrix normalized_;      // Cached normalized values
    Matrix variance_;        // Cached variance
    
    float momentum_;         // Momentum for running statistics
    float epsilon_;          // Small constant for numerical stability
    bool training_mode_;     // Training vs inference mode
    size_t input_size_;      // Size of input features

public:
    /**
     * @brief Construct Batch Normalization layer
     * @param momentum Momentum for running statistics (default: 0.9)
     * @param epsilon Small constant for numerical stability (default: 1e-5)
     */
    explicit BatchNormLayer(float momentum = 0.9f, float epsilon = 1e-5f);
    
    /**
     * @brief Set training mode
     * @param training True for training mode, false for inference
     */
    void set_training_mode(bool training) { training_mode_ = training; }
      /**
     * @brief Get current training mode
     * @return True if in training mode
     */
    bool is_training() const { return training_mode_; }
    
    /**
     * @brief Set input size and initialize parameters
     * @param size Input feature size
     */
    void set_input_size(size_t size);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::string type() const override { return "BatchNorm"; }
    size_t output_size() const override { return input_size_; }    size_t input_size() const override { return input_size_; }
    std::unique_ptr<Layer> clone() const override;
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

} // namespace clmodel
