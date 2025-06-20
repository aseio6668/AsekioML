#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include "regularization.hpp"  // For DropoutLayer
#include <memory>
#include <vector>
#include <string>  // For std::to_string

namespace asekioml {
namespace advanced {

// Batch Normalization Layer
class BatchNormalizationLayer : public Layer {
private:
    Matrix gamma_;          // Scale parameter
    Matrix beta_;           // Shift parameter
    Matrix running_mean_;   // Running average of means
    Matrix running_var_;    // Running average of variances
    Matrix last_input_;
    Matrix normalized_;
    double momentum_;
    double epsilon_;
    bool training_;
    size_t size_;

public:
    BatchNormalizationLayer(size_t size, double momentum = 0.99, double epsilon = 1e-5);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return size_; }
    size_t output_size() const override { return size_; }
    std::string type() const override { return "BatchNorm"; }
    std::string serialize_to_json() const override { return "{\"type\":\"BatchNorm\",\"size\":" + std::to_string(size_) + "}"; }
    
    void set_training(bool training) { training_ = training; }
};

// Layer Normalization
class LayerNormalizationLayer : public Layer {
private:
    Matrix gamma_;
    Matrix beta_;
    Matrix last_input_;
    double epsilon_;
    size_t size_;

public:
    LayerNormalizationLayer(size_t size, double epsilon = 1e-5);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return size_; }
    size_t output_size() const override { return size_; }
    std::string type() const override { return "LayerNorm"; }
    std::string serialize_to_json() const override { return "{\"type\":\"LayerNorm\",\"size\":" + std::to_string(size_) + "}"; }
};

// Residual Connection Layer
class ResidualLayer : public Layer {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    Matrix residual_input_;
    size_t size_;

public:
    ResidualLayer(size_t size);
    
    void add_layer(std::unique_ptr<Layer> layer);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return size_; }
    size_t output_size() const override { return size_; }
    std::string type() const override { return "Residual"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"Residual\",\"size\":" + std::to_string(size_) + "}"; 
    }
};

// Attention Mechanism (Self-Attention)
class AttentionLayer : public Layer {
private:
    Matrix W_query_;
    Matrix W_key_;
    Matrix W_value_;
    Matrix W_output_;
    
    Matrix last_queries_;
    Matrix last_keys_;
    Matrix last_values_;
    Matrix attention_weights_;
    
    size_t d_model_;
    size_t d_k_;
    size_t num_heads_;

public:
    AttentionLayer(size_t d_model, size_t num_heads = 8);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return d_model_; }
    size_t output_size() const override { return d_model_; }
    std::string type() const override { return "Attention"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"Attention\",\"d_model\":" + std::to_string(d_model_) + 
               ",\"num_heads\":" + std::to_string(num_heads_) + "}"; 
    }

private:
    Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V);
    Matrix multi_head_attention(const Matrix& input);
};

// Convolutional Layer (1D for sequence data)
class Conv1DLayer : public Layer {
private:
    Matrix kernels_;        // Shape: (num_filters, kernel_size, input_channels)
    Matrix biases_;         // Shape: (num_filters,)
    Matrix last_input_;
    
    size_t input_channels_;
    size_t output_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;

public:
    Conv1DLayer(size_t input_channels, size_t output_channels, 
               size_t kernel_size, size_t stride = 1, size_t padding = 0);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return input_channels_; }
    size_t output_size() const override { return output_channels_; }
    std::string type() const override { return "Conv1D"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"Conv1D\",\"input_channels\":" + std::to_string(input_channels_) + 
               ",\"output_channels\":" + std::to_string(output_channels_) + 
               ",\"kernel_size\":" + std::to_string(kernel_size_) + "}"; 
    }

private:
    Matrix im2col(const Matrix& input) const;
    Matrix col2im(const Matrix& cols, size_t input_length) const;
};

// 2D Convolutional Layer (for images and 2D data)
class Conv2DLayer : public Layer {
private:
    Matrix kernels_;        // Shape: (num_filters, kernel_height * kernel_width * input_channels)
    Matrix biases_;         // Shape: (num_filters,)
    Matrix last_input_;
    
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

public:
    Conv2DLayer(size_t input_height, size_t input_width, size_t input_channels,
               size_t num_filters, size_t kernel_height, size_t kernel_width,
               size_t stride_h = 1, size_t stride_w = 1, 
               size_t padding_h = 0, size_t padding_w = 0);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override;
    size_t output_size() const override;
    std::string type() const override { return "Conv2D"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"Conv2D\",\"filters\":" + std::to_string(num_filters_) + 
               ",\"kernel\":[" + std::to_string(kernel_height_) + "," + std::to_string(kernel_width_) + "]}"; 
    }
    
    // Getters for layer configuration
    size_t get_output_height() const { return output_height_; }
    size_t get_output_width() const { return output_width_; }
    size_t get_num_filters() const { return num_filters_; }
};

// LSTM Layer
class LSTMLayer : public Layer {
private:
    // Weight matrices for forget, input, candidate, and output gates
    Matrix W_f_, U_f_, b_f_;  // Forget gate
    Matrix W_i_, U_i_, b_i_;  // Input gate
    Matrix W_c_, U_c_, b_c_;  // Candidate
    Matrix W_o_, U_o_, b_o_;  // Output gate
    
    // State variables
    Matrix cell_state_;
    Matrix hidden_state_;
    
    // For backpropagation
    std::vector<Matrix> hidden_states_;
    std::vector<Matrix> cell_states_;
    std::vector<Matrix> forget_gates_;
    std::vector<Matrix> input_gates_;
    std::vector<Matrix> candidate_values_;
    std::vector<Matrix> output_gates_;
    
    size_t input_size_;
    size_t hidden_size_;
    size_t sequence_length_;

public:
    LSTMLayer(size_t input_size, size_t hidden_size);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return input_size_; }
    size_t output_size() const override { return hidden_size_; }
    std::string type() const override { return "LSTM"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"LSTM\",\"input_size\":" + std::to_string(input_size_) + 
               ",\"hidden_size\":" + std::to_string(hidden_size_) + "}"; 
    }
    
    void reset_state();

private:
    Matrix sigmoid_gate(const Matrix& x, const Matrix& h, 
                       const Matrix& W, const Matrix& U, const Matrix& b);
    Matrix tanh_gate(const Matrix& x, const Matrix& h,
                    const Matrix& W, const Matrix& U, const Matrix& b);
};

// Transformer Block
class TransformerBlock : public Layer {
private:
    std::unique_ptr<AttentionLayer> attention_;
    std::unique_ptr<LayerNormalizationLayer> norm1_;
    std::unique_ptr<DenseLayer> feed_forward1_;
    std::unique_ptr<DenseLayer> feed_forward2_;
    std::unique_ptr<LayerNormalizationLayer> norm2_;
    std::unique_ptr<DropoutLayer> dropout1_;
    std::unique_ptr<DropoutLayer> dropout2_;
    
    size_t d_model_;
    size_t d_ff_;

public:
    TransformerBlock(size_t d_model, size_t d_ff, size_t num_heads = 8, double dropout_rate = 0.1);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return d_model_; }
    size_t output_size() const override { return d_model_; }
    std::string type() const override { return "Transformer"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"Transformer\",\"d_model\":" + std::to_string(d_model_) + 
               ",\"d_ff\":" + std::to_string(d_ff_) + "}"; 
    }
};

// Embedding Layer
class EmbeddingLayer : public Layer {
private:
    Matrix embeddings_;     // Shape: (vocab_size, embedding_dim)
    Matrix last_input_indices_;
    size_t vocab_size_;
    size_t embedding_dim_;

public:
    EmbeddingLayer(size_t vocab_size, size_t embedding_dim);
    
    Matrix forward(const Matrix& input) override;  // Input should be indices
    Matrix backward(const Matrix& gradient) override;
    void update_weights(double learning_rate) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return vocab_size_; }
    size_t output_size() const override { return embedding_dim_; }
    std::string type() const override { return "Embedding"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"Embedding\",\"vocab_size\":" + std::to_string(vocab_size_) + 
               ",\"embedding_dim\":" + std::to_string(embedding_dim_) + "}"; 
    }
    
    // Utility functions
    Matrix get_embeddings() const { return embeddings_; }
    void load_pretrained_embeddings(const Matrix& pretrained);
};

// Positional Encoding (for Transformers)
class PositionalEncodingLayer : public Layer {
private:
    Matrix positional_encodings_;
    size_t max_length_;
    size_t d_model_;

public:
    PositionalEncodingLayer(size_t max_length, size_t d_model);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return d_model_; }
    size_t output_size() const override { return d_model_; }
    std::string type() const override { return "PositionalEncoding"; }
    std::string serialize_to_json() const override { 
        return "{\"type\":\"PositionalEncoding\",\"max_length\":" + std::to_string(max_length_) + 
               ",\"d_model\":" + std::to_string(d_model_) + "}"; 
    }

private:
    void compute_positional_encodings();
};

// L1/L2 Regularization Layer
class RegularizationLayer : public Layer {
private:
    double l1_lambda_;
    double l2_lambda_;
    Matrix last_input_;
    size_t size_;

public:
    RegularizationLayer(size_t size, double l1_lambda = 0.0, double l2_lambda = 0.0);
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
    std::unique_ptr<Layer> clone() const override;
      size_t input_size() const override { return size_; }
    size_t output_size() const override { return size_; }
    std::string type() const override;
    std::string serialize_to_json() const override { 
        return "{\"type\":\"Regularization\",\"size\":" + std::to_string(size_) + 
               ",\"l1\":" + std::to_string(l1_lambda_) + ",\"l2\":" + std::to_string(l2_lambda_) + "}"; 
    }
    
    // Regularization parameter accessors
    double get_l1_lambda() const { return l1_lambda_; }
    double get_l2_lambda() const { return l2_lambda_; }
    void set_l1_lambda(double lambda) { l1_lambda_ = lambda; }
    void set_l2_lambda(double lambda) { l2_lambda_ = lambda; }
};

} // namespace advanced
} // namespace clmodel
