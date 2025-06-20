#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include <vector>
#include <memory>
#include <complex>

namespace asekioml {
namespace ai {

/**
 * @brief Transformer architecture components for language models
 */
class MultiHeadAttentionLayer : public Layer {
private:
    size_t d_model_;      // Model dimension
    size_t num_heads_;    // Number of attention heads
    size_t d_k_;          // Key/Query dimension per head
    size_t d_v_;          // Value dimension per head
    
    // Weight matrices for Q, K, V projections
    Matrix W_q_, W_k_, W_v_, W_o_;
    Matrix last_input_;
    
public:
    MultiHeadAttentionLayer(size_t d_model, size_t num_heads);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return d_model_; }
    size_t output_size() const override { return d_model_; }
    std::string type() const override { return "MultiHeadAttention"; }
    
    // Serialization
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V);
    Matrix apply_causal_mask(const Matrix& attention_scores);
};

/**
 * @brief Positional encoding for transformer models
 */
class PositionalEncodingLayer : public Layer {
private:
    size_t max_sequence_length_;
    size_t d_model_;
    Matrix encoding_matrix_;
    
public:
    PositionalEncodingLayer(size_t max_sequence_length, size_t d_model);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return d_model_; }
    size_t output_size() const override { return d_model_; }
    std::string type() const override { return "PositionalEncoding"; }
    
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    void generate_positional_encodings();
};

/**
 * @brief Convolutional layers for image/video processing
 */
class Conv2DLayer : public Layer {
private:
    size_t input_channels_, output_channels_;
    size_t kernel_size_, stride_, padding_;
    size_t input_height_, input_width_;
    
    Matrix kernels_;  // Shape: [output_channels, input_channels, kernel_size, kernel_size]
    Matrix biases_;   // Shape: [output_channels]
    Matrix last_input_;
    
public:
    Conv2DLayer(size_t input_channels, size_t output_channels, 
                size_t kernel_size, size_t stride = 1, size_t padding = 0,
                size_t input_height = 0, size_t input_width = 0);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "Conv2D"; }
    
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    Matrix im2col(const Matrix& input);
    Matrix col2im(const Matrix& col, size_t height, size_t width);
};

/**
 * @brief Transposed convolution for generative models (upsampling)
 */
class ConvTranspose2DLayer : public Layer {
private:
    size_t input_channels_, output_channels_;
    size_t kernel_size_, stride_, padding_;
    Matrix kernels_, biases_;
    Matrix last_input_;
    
public:
    ConvTranspose2DLayer(size_t input_channels, size_t output_channels,
                        size_t kernel_size, size_t stride = 1, size_t padding = 0);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "ConvTranspose2D"; }
    
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

/**
 * @brief Spectral convolution for audio processing
 */
class SpectralConvLayer : public Layer {
private:
    size_t fft_size_;
    size_t overlap_;
    Matrix frequency_kernels_;
    Matrix last_input_;
    
public:
    SpectralConvLayer(size_t fft_size, size_t num_filters, size_t overlap = 0);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return fft_size_; }
    size_t output_size() const override;
    std::string type() const override { return "SpectralConv"; }
    
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
    
private:
    Matrix apply_fft(const Matrix& input);
    Matrix apply_ifft(const Matrix& frequency_domain);
};

/**
 * @brief Cross-attention for multi-modal fusion
 */
class CrossAttentionLayer : public Layer {
private:
    size_t d_model_;
    size_t num_heads_;
    Matrix W_q_, W_k_, W_v_, W_o_;
    Matrix last_query_, last_key_value_;
    
public:
    CrossAttentionLayer(size_t d_model, size_t num_heads);
    
    // Special forward for cross-attention (query from one modality, key-value from another)
    Matrix forward_cross(const Matrix& query, const Matrix& key_value);
    Matrix forward(const Matrix& input) override; // Default implementation
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return d_model_; }
    size_t output_size() const override { return d_model_; }
    std::string type() const override { return "CrossAttention"; }
    
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

/**
 * @brief Variational Autoencoder layer for generative modeling
 */
class VAELayer : public Layer {
private:
    size_t latent_dim_;
    size_t input_dim_;
    Matrix mu_weights_, mu_bias_;
    Matrix logvar_weights_, logvar_bias_;
    Matrix last_input_;
    Matrix last_mu_, last_logvar_;
    
public:
    VAELayer(size_t input_dim, size_t latent_dim);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
    
    size_t input_size() const override { return input_dim_; }
    size_t output_size() const override { return latent_dim_; }
    std::string type() const override { return "VAE"; }
    
    // VAE-specific methods
    Matrix sample_latent(const Matrix& mu, const Matrix& logvar);
    double kl_divergence_loss();
    
    std::string serialize_to_json() const override;
    void serialize_weights(std::ofstream& file) const override;
    void deserialize_weights(std::ifstream& file) override;
    size_t get_weights_size() const override;
};

} // namespace ai
} // namespace asekioml
