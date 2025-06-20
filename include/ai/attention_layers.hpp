#pragma once

#include "../tensor.hpp"
#include "../layer.hpp"
#include "compute_engine.hpp"
#include <memory>
#include <vector>

namespace asekioml {
namespace ai {

/**
 * @brief Multi-Head Attention mechanism for transformer architectures
 * 
 * Implements the attention mechanism from "Attention Is All You Need" paper.
 * Supports both self-attention and cross-attention patterns.
 */
class MultiHeadAttentionLayer : public Layer {
private:
    size_t model_dim_;        // Model dimension (d_model)
    size_t num_heads_;        // Number of attention heads
    size_t head_dim_;         // Dimension per head (d_model / num_heads)
    size_t max_seq_length_;   // Maximum sequence length
    
    // Learnable parameters
    Tensor query_weights_;    // [model_dim, model_dim]
    Tensor key_weights_;      // [model_dim, model_dim]
    Tensor value_weights_;    // [model_dim, model_dim]
    Tensor output_weights_;   // [model_dim, model_dim]
    
    Tensor query_bias_;       // [model_dim]
    Tensor key_bias_;         // [model_dim]
    Tensor value_bias_;       // [model_dim]
    Tensor output_bias_;      // [model_dim]
    
    // Gradients
    Tensor query_weights_grad_;
    Tensor key_weights_grad_;
    Tensor value_weights_grad_;
    Tensor output_weights_grad_;
    Tensor query_bias_grad_;
    Tensor key_bias_grad_;
    Tensor value_bias_grad_;
    Tensor output_bias_grad_;
    
    // Cached values for backward pass
    Tensor last_query_;
    Tensor last_key_;
    Tensor last_value_;
    Tensor last_attention_weights_;
    Tensor last_attention_output_;
    
    // Configuration
    double dropout_rate_ = 0.1;
    bool use_bias_ = true;
    bool scale_attention_ = true;
    bool causal_mask_ = false;
    
    bool weights_initialized_ = false;
    
public:
    MultiHeadAttentionLayer(size_t model_dim, size_t num_heads, 
                           size_t max_seq_length = 512, bool use_bias = true);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return model_dim_; }
    size_t output_size() const override { return model_dim_; }
    std::string type() const override { return "MultiHeadAttention"; }
    
    // Tensor-based interface (preferred)
    Tensor forward_tensor(const Tensor& query, const Tensor& key, const Tensor& value,
                         const Tensor* attention_mask = nullptr);
    Tensor forward_tensor_self_attention(const Tensor& input,
                                       const Tensor* attention_mask = nullptr);
    
    std::tuple<Tensor, Tensor, Tensor> backward_tensor(const Tensor& grad_output);
    
    // Configuration
    void set_dropout_rate(double rate) { dropout_rate_ = rate; }
    void enable_causal_mask(bool enable) { causal_mask_ = enable; }
    void set_scale_attention(bool scale) { scale_attention_ = scale; }
    
    // Weight initialization
    void initialize_weights(const std::string& method = "xavier");
    
    // Getters for inspection
    const Tensor& get_query_weights() const { return query_weights_; }
    const Tensor& get_key_weights() const { return key_weights_; }
    const Tensor& get_value_weights() const { return value_weights_; }
    const Tensor& get_output_weights() const { return output_weights_; }
    const Tensor& get_last_attention_weights() const { return last_attention_weights_; }
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    void validate_dimensions() const;
    void validate_input_tensor(const Tensor& tensor, const std::string& name) const;
    
    // Core attention computation
    Tensor compute_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                           const Tensor* mask = nullptr);
    
    // Helper functions
    Tensor split_heads(const Tensor& x) const;
    Tensor combine_heads(const Tensor& x) const;
    Tensor apply_attention_mask(const Tensor& attention_scores, const Tensor& mask) const;
    Tensor create_causal_mask(size_t seq_length) const;
    
    // Linear transformations
    Tensor linear_transform(const Tensor& input, const Tensor& weights, const Tensor& bias) const;
    
    // Softmax with numerical stability
    Tensor stable_softmax(const Tensor& x, int dim = -1) const;
    
    // Gradient computation helpers
    std::tuple<Tensor, Tensor, Tensor> compute_linear_gradients(
        const Tensor& input, const Tensor& weights, const Tensor& grad_output) const;
};

/**
 * @brief Positional encoding for transformer architectures
 */
class PositionalEncoding {
private:
    Tensor encoding_table_;
    size_t max_length_;
    size_t model_dim_;
    
public:
    PositionalEncoding(size_t max_length, size_t model_dim);
    
    Tensor get_encoding(size_t sequence_length) const;
    void add_to_tensor(Tensor& input) const;
    
    // Different encoding types
    static Tensor sinusoidal_encoding(size_t max_length, size_t model_dim);
    static Tensor learned_encoding(size_t max_length, size_t model_dim);
};

/**
 * @brief Layer normalization for transformer blocks
 */
class LayerNormalizationLayer : public Layer {
private:
    size_t feature_dim_;
    double epsilon_;
    
    Tensor gamma_;  // Scale parameter
    Tensor beta_;   // Shift parameter
    
    Tensor gamma_grad_;
    Tensor beta_grad_;
    
    // Cached values for backward pass
    Tensor last_input_;
    Tensor last_mean_;
    Tensor last_variance_;
    Tensor last_normalized_;
    
    bool weights_initialized_ = false;
    
public:
    LayerNormalizationLayer(size_t feature_dim, double epsilon = 1e-6);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return feature_dim_; }
    size_t output_size() const override { return feature_dim_; }
    std::string type() const override { return "LayerNormalization"; }
    
    // Tensor-based interface
    Tensor forward_tensor(const Tensor& input);
    Tensor backward_tensor(const Tensor& grad_output);
    
    // Weight initialization
    void initialize_weights();
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    void validate_input_tensor(const Tensor& input) const;
    std::tuple<Tensor, Tensor> compute_mean_variance(const Tensor& input) const;
};

/**
 * @brief Feed-forward network for transformer blocks
 */
class TransformerFeedForwardLayer : public Layer {
private:
    size_t input_dim_;
    size_t hidden_dim_;
    
    Tensor weights1_;      // [input_dim, hidden_dim]
    Tensor bias1_;         // [hidden_dim]
    Tensor weights2_;      // [hidden_dim, input_dim]
    Tensor bias2_;         // [input_dim]
    
    Tensor weights1_grad_;
    Tensor bias1_grad_;
    Tensor weights2_grad_;
    Tensor bias2_grad_;
    
    // Cached values
    Tensor last_input_;
    Tensor last_hidden_;
    
    double dropout_rate_ = 0.1;
    std::string activation_ = "relu";
    bool weights_initialized_ = false;
    
public:
    TransformerFeedForwardLayer(size_t input_dim, size_t hidden_dim, 
                               const std::string& activation = "relu");
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return input_dim_; }
    size_t output_size() const override { return input_dim_; }
    std::string type() const override { return "TransformerFeedForward"; }
    
    // Tensor-based interface
    Tensor forward_tensor(const Tensor& input);
    Tensor backward_tensor(const Tensor& grad_output);
    
    // Configuration
    void set_dropout_rate(double rate) { dropout_rate_ = rate; }
    void set_activation(const std::string& activation) { activation_ = activation; }
    
    // Weight initialization
    void initialize_weights(const std::string& method = "xavier");
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    Tensor apply_activation(const Tensor& input) const;
    Tensor apply_activation_derivative(const Tensor& input) const;
};

/**
 * @brief Complete transformer block combining attention, normalization, and feed-forward
 */
class TransformerBlock : public Layer {
private:
    std::unique_ptr<MultiHeadAttentionLayer> attention_;
    std::unique_ptr<LayerNormalizationLayer> norm1_;
    std::unique_ptr<LayerNormalizationLayer> norm2_;
    std::unique_ptr<TransformerFeedForwardLayer> feed_forward_;
    
    size_t model_dim_;
    double dropout_rate_ = 0.1;
    bool pre_norm_ = false; // Pre-norm vs post-norm architecture
    
    // Cached values
    Tensor last_input_;
    Tensor last_attention_output_;
    Tensor last_norm1_output_;
    Tensor last_ff_output_;
    
public:
    TransformerBlock(size_t model_dim, size_t num_heads, size_t ff_hidden_dim,
                    bool pre_norm = false);
    
    // Layer interface implementation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return model_dim_; }
    size_t output_size() const override { return model_dim_; }
    std::string type() const override { return "TransformerBlock"; }
    
    // Tensor-based interface
    Tensor forward_tensor(const Tensor& input, const Tensor* attention_mask = nullptr);
    Tensor backward_tensor(const Tensor& grad_output);
    
    // Configuration
    void set_dropout_rate(double rate);
    void enable_causal_mask(bool enable);
    
    // Access to sub-layers
    MultiHeadAttentionLayer& attention() { return *attention_; }
    LayerNormalizationLayer& norm1() { return *norm1_; }
    LayerNormalizationLayer& norm2() { return *norm2_; }
    TransformerFeedForwardLayer& feed_forward() { return *feed_forward_; }
    
    // Serialization methods
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

} // namespace ai
} // namespace asekioml
