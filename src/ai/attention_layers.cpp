#include "../../include/ai/attention_layers.hpp"
#include "../../include/ai/memory_manager.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {
namespace ai {

// ================================================================================================
// MultiHeadAttentionLayer Implementation
// ================================================================================================

MultiHeadAttentionLayer::MultiHeadAttentionLayer(size_t model_dim, size_t num_heads,
                                                size_t max_seq_length, bool use_bias)
    : model_dim_(model_dim)
    , num_heads_(num_heads)
    , max_seq_length_(max_seq_length)
    , use_bias_(use_bias) {
    
    if (model_dim_ % num_heads_ != 0) {
        throw std::invalid_argument("Model dimension must be divisible by number of heads");
    }
    
    head_dim_ = model_dim_ / num_heads_;
    validate_dimensions();
    
    // Initialize weight tensors
    query_weights_ = Tensor::zeros({model_dim_, model_dim_});
    key_weights_ = Tensor::zeros({model_dim_, model_dim_});
    value_weights_ = Tensor::zeros({model_dim_, model_dim_});
    output_weights_ = Tensor::zeros({model_dim_, model_dim_});
    
    if (use_bias_) {
        query_bias_ = Tensor::zeros({model_dim_});
        key_bias_ = Tensor::zeros({model_dim_});
        value_bias_ = Tensor::zeros({model_dim_});
        output_bias_ = Tensor::zeros({model_dim_});
    }
    
    // Initialize gradient tensors
    query_weights_grad_ = Tensor::zeros({model_dim_, model_dim_});
    key_weights_grad_ = Tensor::zeros({model_dim_, model_dim_});
    value_weights_grad_ = Tensor::zeros({model_dim_, model_dim_});
    output_weights_grad_ = Tensor::zeros({model_dim_, model_dim_});
    
    if (use_bias_) {
        query_bias_grad_ = Tensor::zeros({model_dim_});
        key_bias_grad_ = Tensor::zeros({model_dim_});
        value_bias_grad_ = Tensor::zeros({model_dim_});
        output_bias_grad_ = Tensor::zeros({model_dim_});
    }
}

void MultiHeadAttentionLayer::initialize_weights(const std::string& method) {
    double scale = 1.0;
    
    if (method == "xavier" || method == "glorot") {
        scale = std::sqrt(6.0 / (2.0 * model_dim_));
    } else if (method == "he" || method == "kaiming") {
        scale = std::sqrt(2.0 / model_dim_);
    } else {
        scale = 0.02; // Default small random
    }
    
    // Initialize weight matrices
    query_weights_ = Tensor::randn({model_dim_, model_dim_}, 0.0, scale);
    key_weights_ = Tensor::randn({model_dim_, model_dim_}, 0.0, scale);
    value_weights_ = Tensor::randn({model_dim_, model_dim_}, 0.0, scale);
    output_weights_ = Tensor::randn({model_dim_, model_dim_}, 0.0, scale);
    
    // Initialize biases to zero
    if (use_bias_) {
        query_bias_ = Tensor::zeros({model_dim_});
        key_bias_ = Tensor::zeros({model_dim_});
        value_bias_ = Tensor::zeros({model_dim_});
        output_bias_ = Tensor::zeros({model_dim_});
    }
    
    weights_initialized_ = true;
}

void MultiHeadAttentionLayer::validate_dimensions() const {
    if (model_dim_ == 0 || num_heads_ == 0) {
        throw std::invalid_argument("Model dimension and number of heads must be positive");
    }
    
    if (model_dim_ % num_heads_ != 0) {
        throw std::invalid_argument("Model dimension must be divisible by number of heads");
    }
    
    if (head_dim_ == 0) {
        throw std::invalid_argument("Head dimension cannot be zero");
    }
}

void MultiHeadAttentionLayer::validate_input_tensor(const Tensor& tensor, const std::string& name) const {
    if (tensor.ndim() < 3) {
        throw std::invalid_argument(name + " must be at least 3D: [batch, seq_len, model_dim]");
    }
    
    if (tensor.size(tensor.ndim() - 1) != model_dim_) {
        throw std::invalid_argument(name + " last dimension must match model dimension");
    }
}

Tensor MultiHeadAttentionLayer::linear_transform(const Tensor& input, const Tensor& weights, const Tensor& bias) const {
    // input: [batch, seq_len, model_dim]
    // weights: [model_dim, model_dim]
    // output: [batch, seq_len, model_dim]
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto input_dim = input.size(2);
    auto output_dim = weights.size(1);
    
    // Reshape input to [batch * seq_len, model_dim] for matrix multiplication
    Tensor input_2d = input.reshape({batch_size * seq_len, input_dim});
    
    // Matrix multiplication
    Tensor output_2d = input_2d.matmul(weights);
    
    // Add bias if provided
    if (use_bias_ && bias.size() > 0) {
        for (size_t i = 0; i < output_2d.size(0); ++i) {
            for (size_t j = 0; j < output_dim; ++j) {
                output_2d({i, j}) += bias({j});
            }
        }
    }
    
    // Reshape back to [batch, seq_len, output_dim]
    return output_2d.reshape({batch_size, seq_len, output_dim});
}

Tensor MultiHeadAttentionLayer::split_heads(const Tensor& x) const {
    // x: [batch, seq_len, model_dim]
    // output: [batch, num_heads, seq_len, head_dim]
    
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    
    // Reshape to [batch, seq_len, num_heads, head_dim]
    Tensor reshaped = x.reshape({batch_size, seq_len, num_heads_, head_dim_});
    
    // Transpose to [batch, num_heads, seq_len, head_dim]
    return reshaped.transpose({0, 2, 1, 3});
}

Tensor MultiHeadAttentionLayer::combine_heads(const Tensor& x) const {
    // x: [batch, num_heads, seq_len, head_dim]
    // output: [batch, seq_len, model_dim]
    
    auto batch_size = x.size(0);
    auto seq_len = x.size(2);
    
    // Transpose to [batch, seq_len, num_heads, head_dim]
    Tensor transposed = x.transpose({0, 2, 1, 3});
    
    // Reshape to [batch, seq_len, model_dim]
    return transposed.reshape({batch_size, seq_len, model_dim_});
}

Tensor MultiHeadAttentionLayer::stable_softmax(const Tensor& x, int dim) const {
    // For numerical stability, subtract max before exp
    auto max_vals = x.max(dim, true);
    auto shifted = x - max_vals;
    
    // Apply exp
    Tensor exp_vals = shifted;
    for (size_t i = 0; i < exp_vals.size(); ++i) {
        exp_vals.data()[i] = std::exp(exp_vals.data()[i]);
    }
    
    // Sum and normalize
    auto sum_vals = exp_vals.sum(dim, true);
    return exp_vals / sum_vals;
}

Tensor MultiHeadAttentionLayer::create_causal_mask(size_t seq_length) const {
    Tensor mask({seq_length, seq_length});
    
    for (size_t i = 0; i < seq_length; ++i) {
        for (size_t j = 0; j < seq_length; ++j) {
            // Allow attention to current and previous positions
            mask({i, j}) = (j <= i) ? 0.0 : -1e9;
        }
    }
    
    return mask;
}

Tensor MultiHeadAttentionLayer::apply_attention_mask(const Tensor& attention_scores, const Tensor& mask) const {
    // attention_scores: [batch, num_heads, seq_len, seq_len]
    // mask: [seq_len, seq_len] or [batch, seq_len, seq_len]
    
    Tensor masked_scores = attention_scores;
    
    auto batch_size = attention_scores.size(0);
    auto num_heads = attention_scores.size(1);
    auto seq_len = attention_scores.size(2);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t mask_i = (mask.size(0) == 1) ? 0 : i;
                    size_t mask_j = (mask.size(1) == 1) ? 0 : j;
                    
                    if (mask.ndim() == 2) {
                        masked_scores({b, h, i, j}) += mask({mask_i, mask_j});
                    } else if (mask.ndim() == 3) {
                        masked_scores({b, h, i, j}) += mask({b, mask_i, mask_j});
                    }
                }
            }
        }
    }
    
    return masked_scores;
}

Tensor MultiHeadAttentionLayer::compute_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                                 const Tensor* mask) {
    // query, key, value: [batch, num_heads, seq_len, head_dim]
    
    auto batch_size = query.size(0);
    auto seq_len_q = query.size(2);
    auto seq_len_kv = key.size(2);
    
    // Compute attention scores: Q @ K^T
    // query: [batch, num_heads, seq_len_q, head_dim]
    // key^T: [batch, num_heads, head_dim, seq_len_kv]
    Tensor key_transposed = key.transpose({0, 1, 3, 2});
    
    // This is a simplified matrix multiplication - in practice you'd want to optimize this
    Tensor attention_scores = Tensor::zeros({batch_size, num_heads_, seq_len_q, seq_len_kv});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
            for (size_t i = 0; i < seq_len_q; ++i) {
                for (size_t j = 0; j < seq_len_kv; ++j) {
                    double score = 0.0;
                    for (size_t k = 0; k < head_dim_; ++k) {
                        score += query({b, h, i, k}) * key_transposed({b, h, k, j});
                    }
                    attention_scores({b, h, i, j}) = score;
                }
            }
        }
    }
    
    // Scale by sqrt(head_dim) for stability
    if (scale_attention_) {
        double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_dim_));
        attention_scores = attention_scores * scale_factor;
    }
    
    // Apply causal mask if enabled
    if (causal_mask_) {
        auto causal_mask_tensor = create_causal_mask(seq_len_q);
        attention_scores = apply_attention_mask(attention_scores, causal_mask_tensor);
    }
    
    // Apply custom mask if provided
    if (mask) {
        attention_scores = apply_attention_mask(attention_scores, *mask);
    }
    
    // Apply softmax to get attention weights
    Tensor attention_weights = stable_softmax(attention_scores, -1);
    
    // Store for backward pass
    last_attention_weights_ = attention_weights;
    
    // Apply attention to values: attention_weights @ V
    Tensor attention_output = Tensor::zeros({batch_size, num_heads_, seq_len_q, head_dim_});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
            for (size_t i = 0; i < seq_len_q; ++i) {
                for (size_t k = 0; k < head_dim_; ++k) {
                    double weighted_value = 0.0;
                    for (size_t j = 0; j < seq_len_kv; ++j) {
                        weighted_value += attention_weights({b, h, i, j}) * value({b, h, j, k});
                    }
                    attention_output({b, h, i, k}) = weighted_value;
                }
            }
        }
    }
    
    return attention_output;
}

Tensor MultiHeadAttentionLayer::forward_tensor(const Tensor& query, const Tensor& key, const Tensor& value,
                                              const Tensor* attention_mask) {
    if (!weights_initialized_) {
        initialize_weights();
    }
    
    validate_input_tensor(query, "query");
    validate_input_tensor(key, "key");
    validate_input_tensor(value, "value");
    
    // Store inputs for backward pass
    last_query_ = query;
    last_key_ = key;
    last_value_ = value;
    
    // Linear transformations
    Tensor Q = linear_transform(query, query_weights_, query_bias_);
    Tensor K = linear_transform(key, key_weights_, key_bias_);
    Tensor V = linear_transform(value, value_weights_, value_bias_);
    
    // Split into multiple heads
    Q = split_heads(Q);
    K = split_heads(K);
    V = split_heads(V);
    
    // Compute attention
    Tensor attention_output = compute_attention(Q, K, V, attention_mask);
    
    // Combine heads
    attention_output = combine_heads(attention_output);
    
    // Final linear transformation
    Tensor output = linear_transform(attention_output, output_weights_, output_bias_);
    
    last_attention_output_ = attention_output;
    return output;
}

Tensor MultiHeadAttentionLayer::forward_tensor_self_attention(const Tensor& input,
                                                            const Tensor* attention_mask) {
    return forward_tensor(input, input, input, attention_mask);
}

Matrix MultiHeadAttentionLayer::forward(const Matrix& input) {
    // Convert Matrix to Tensor for processing
    if (input.rows() != 1) {
        throw std::invalid_argument("Matrix input must have batch size 1");
    }
    
    size_t total_size = input.cols();
    size_t seq_len = total_size / model_dim_;
    
    if (total_size % model_dim_ != 0) {
        throw std::invalid_argument("Input size must be divisible by model dimension");
    }
    
    // Reshape to [1, seq_len, model_dim]
    Tensor input_tensor({1, seq_len, model_dim_});
    for (size_t i = 0; i < total_size; ++i) {
        input_tensor.data()[i] = input(0, i);
    }
    
    // Forward pass
    Tensor output_tensor = forward_tensor_self_attention(input_tensor);
    
    // Convert back to Matrix
    Matrix output(1, total_size);
    for (size_t i = 0; i < total_size; ++i) {
        output(0, i) = output_tensor.data()[i];
    }
    
    return output;
}

Matrix MultiHeadAttentionLayer::backward(const Matrix& gradient) {
    // Convert Matrix gradient to Tensor and use tensor-based backward
    if (gradient.rows() != 1) {
        throw std::invalid_argument("Matrix gradient must have batch size 1");
    }
    
    size_t total_size = gradient.cols();
    size_t seq_len = total_size / model_dim_;
    
    if (total_size % model_dim_ != 0) {
        throw std::invalid_argument("Matrix gradient size not compatible with model dimension");
    }
    
    // Convert to tensor
    Tensor grad_tensor({1, seq_len, model_dim_});
    for (size_t i = 0; i < total_size; ++i) {
        grad_tensor.data()[i] = gradient(0, i);
    }
    
    // Use tensor backward (for self-attention, query = key = value gradients are summed)
    auto [grad_query, grad_key, grad_value] = backward_tensor(grad_tensor);
    Tensor grad_input = grad_query + grad_key + grad_value;
    
    // Convert back to Matrix
    Matrix result(1, total_size);
    for (size_t i = 0; i < total_size; ++i) {
        result(0, i) = grad_input.data()[i];
    }
    
    return result;
}

std::tuple<Tensor, Tensor, Tensor> MultiHeadAttentionLayer::backward_tensor(const Tensor& grad_output) {
    // grad_output: [batch, seq_len, model_dim] - gradient w.r.t. output
    
    if (!weights_initialized_) {
        throw std::runtime_error("Cannot compute gradients: weights not initialized");
    }
    
    auto batch_size = grad_output.size(0);
    auto seq_len = grad_output.size(1);
    
    // Gradient w.r.t. attention output (before final linear transformation)
    Tensor grad_attention_output = linear_transform(grad_output, output_weights_.transpose({1, 0}), Tensor());
    
    // Compute gradient w.r.t. output weights and bias
    Tensor last_attention_output_2d = last_attention_output_.reshape({batch_size * seq_len, model_dim_});
    Tensor grad_output_2d = grad_output.reshape({batch_size * seq_len, model_dim_});
    
    output_weights_grad_ = last_attention_output_2d.transpose({1, 0}).matmul(grad_output_2d);
    
    if (use_bias_) {
        output_bias_grad_ = grad_output_2d.sum(0);
    }
    
    // Split attention output gradient into heads
    Tensor grad_attention_split = split_heads(grad_attention_output);
    // grad_attention_split: [batch, num_heads, seq_len, head_dim]
    
    // Gradient w.r.t. attention computation
    // attention_output = attention_weights @ V
    // grad_V = attention_weights^T @ grad_attention_output
    // grad_attention_weights = grad_attention_output @ V^T
    
    Tensor grad_value_heads = Tensor::zeros({batch_size, num_heads_, seq_len, head_dim_});
    Tensor grad_attention_weights = Tensor::zeros({batch_size, num_heads_, seq_len, seq_len});
    
    // Compute gradients w.r.t. values and attention weights
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
            // grad_V = attention_weights^T @ grad_attention_output
            for (size_t j = 0; j < seq_len; ++j) {
                for (size_t k = 0; k < head_dim_; ++k) {
                    double grad_val = 0.0;
                    for (size_t i = 0; i < seq_len; ++i) {
                        grad_val += last_attention_weights_({b, h, i, j}) * grad_attention_split({b, h, i, k});
                    }
                    grad_value_heads({b, h, j, k}) = grad_val;
                }
            }
            
            // grad_attention_weights = grad_attention_output @ V^T
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    double grad_att = 0.0;
                    for (size_t k = 0; k < head_dim_; ++k) {
                        // Note: using last_value_ split into heads
                        Tensor value_heads = split_heads(linear_transform(last_value_, value_weights_, value_bias_));
                        grad_att += grad_attention_split({b, h, i, k}) * value_heads({b, h, j, k});
                    }
                    grad_attention_weights({b, h, i, j}) = grad_att;
                }
            }
        }
    }
    
    // Gradient w.r.t. attention scores (before softmax)
    // grad_scores = softmax_backward(grad_attention_weights, attention_weights)
    Tensor grad_attention_scores = Tensor::zeros({batch_size, num_heads_, seq_len, seq_len});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
            // Softmax backward: grad_input = softmax * (grad_output - sum(grad_output * softmax))
            for (size_t i = 0; i < seq_len; ++i) {
                double sum_grad_att = 0.0;
                for (size_t j = 0; j < seq_len; ++j) {
                    sum_grad_att += grad_attention_weights({b, h, i, j}) * last_attention_weights_({b, h, i, j});
                }
                
                for (size_t j = 0; j < seq_len; ++j) {
                    grad_attention_scores({b, h, i, j}) = last_attention_weights_({b, h, i, j}) * 
                        (grad_attention_weights({b, h, i, j}) - sum_grad_att);
                }
            }
        }
    }
    
    // Apply attention scaling
    if (scale_attention_) {
        double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_dim_));
        grad_attention_scores = grad_attention_scores * scale_factor;
    }
    
    // Gradient w.r.t. Q and K from attention scores
    // scores = Q @ K^T
    // grad_Q = grad_scores @ K
    // grad_K = grad_scores^T @ Q
    
    Tensor query_heads = split_heads(linear_transform(last_query_, query_weights_, query_bias_));
    Tensor key_heads = split_heads(linear_transform(last_key_, key_weights_, key_bias_));
    
    Tensor grad_query_heads = Tensor::zeros({batch_size, num_heads_, seq_len, head_dim_});
    Tensor grad_key_heads = Tensor::zeros({batch_size, num_heads_, seq_len, head_dim_});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
            // grad_Q = grad_scores @ K
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t k = 0; k < head_dim_; ++k) {
                    double grad_q = 0.0;
                    for (size_t j = 0; j < seq_len; ++j) {
                        grad_q += grad_attention_scores({b, h, i, j}) * key_heads({b, h, j, k});
                    }
                    grad_query_heads({b, h, i, k}) = grad_q;
                }
            }
            
            // grad_K = grad_scores^T @ Q
            for (size_t j = 0; j < seq_len; ++j) {
                for (size_t k = 0; k < head_dim_; ++k) {
                    double grad_k = 0.0;
                    for (size_t i = 0; i < seq_len; ++i) {
                        grad_k += grad_attention_scores({b, h, i, j}) * query_heads({b, h, i, k});
                    }
                    grad_key_heads({b, h, j, k}) = grad_k;
                }
            }
        }
    }
    
    // Combine heads back
    Tensor grad_query_combined = combine_heads(grad_query_heads);
    Tensor grad_key_combined = combine_heads(grad_key_heads);
    Tensor grad_value_combined = combine_heads(grad_value_heads);
    
    // Gradient w.r.t. linear transformation weights and inputs
    auto [grad_query_input, grad_query_weights, grad_query_bias] = 
        compute_linear_gradients(last_query_, query_weights_, grad_query_combined);
    auto [grad_key_input, grad_key_weights, grad_key_bias] = 
        compute_linear_gradients(last_key_, key_weights_, grad_key_combined);
    auto [grad_value_input, grad_value_weights, grad_value_bias] = 
        compute_linear_gradients(last_value_, value_weights_, grad_value_combined);
    
    // Accumulate weight gradients
    query_weights_grad_ = grad_query_weights;
    key_weights_grad_ = grad_key_weights;
    value_weights_grad_ = grad_value_weights;
    
    if (use_bias_) {
        query_bias_grad_ = grad_query_bias;
        key_bias_grad_ = grad_key_bias;
        value_bias_grad_ = grad_value_bias;
    }
    
    return std::make_tuple(grad_query_input, grad_key_input, grad_value_input);
}

std::tuple<Tensor, Tensor, Tensor> MultiHeadAttentionLayer::compute_linear_gradients(
    const Tensor& input, const Tensor& weights, const Tensor& grad_output) const {
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto input_dim = input.size(2);
    auto output_dim = weights.size(1);
    
    // Reshape for matrix operations
    Tensor input_2d = input.reshape({batch_size * seq_len, input_dim});
    Tensor grad_output_2d = grad_output.reshape({batch_size * seq_len, output_dim});
    
    // Gradient w.r.t. weights: input^T @ grad_output
    Tensor grad_weights = input_2d.transpose({1, 0}).matmul(grad_output_2d);
    
    // Gradient w.r.t. bias: sum over batch and sequence dimensions
    Tensor grad_bias = grad_output_2d.sum(0);
    
    // Gradient w.r.t. input: grad_output @ weights^T
    Tensor grad_input_2d = grad_output_2d.matmul(weights.transpose({1, 0}));
    Tensor grad_input = grad_input_2d.reshape({batch_size, seq_len, input_dim});
      return std::make_tuple(grad_input, grad_weights, grad_bias);
}

void MultiHeadAttentionLayer::update_weights(double learning_rate) {
    // Update query weights and bias
    for (size_t i = 0; i < query_weights_.size(); ++i) {
        query_weights_.data()[i] -= learning_rate * query_weights_grad_.data()[i];
    }
    
    // Update key weights and bias
    for (size_t i = 0; i < key_weights_.size(); ++i) {
        key_weights_.data()[i] -= learning_rate * key_weights_grad_.data()[i];
    }
    
    // Update value weights and bias
    for (size_t i = 0; i < value_weights_.size(); ++i) {
        value_weights_.data()[i] -= learning_rate * value_weights_grad_.data()[i];
    }
    
    // Update output weights and bias
    for (size_t i = 0; i < output_weights_.size(); ++i) {
        output_weights_.data()[i] -= learning_rate * output_weights_grad_.data()[i];
    }
    
    if (use_bias_) {
        // Update query bias
        for (size_t i = 0; i < query_bias_.size(); ++i) {
            query_bias_.data()[i] -= learning_rate * query_bias_grad_.data()[i];
        }
        
        // Update key bias
        for (size_t i = 0; i < key_bias_.size(); ++i) {
            key_bias_.data()[i] -= learning_rate * key_bias_grad_.data()[i];
        }
        
        // Update value bias
        for (size_t i = 0; i < value_bias_.size(); ++i) {
            value_bias_.data()[i] -= learning_rate * value_bias_grad_.data()[i];
        }
        
        // Update output bias
        for (size_t i = 0; i < output_bias_.size(); ++i) {
            output_bias_.data()[i] -= learning_rate * output_bias_grad_.data()[i];
        }
    }
}

std::unique_ptr<Layer> MultiHeadAttentionLayer::clone() const {
    auto cloned = std::make_unique<MultiHeadAttentionLayer>(model_dim_, num_heads_, max_seq_length_, use_bias_);
    
    if (weights_initialized_) {
        cloned->query_weights_ = query_weights_;
        cloned->key_weights_ = key_weights_;
        cloned->value_weights_ = value_weights_;
        cloned->output_weights_ = output_weights_;
        
        if (use_bias_) {
            cloned->query_bias_ = query_bias_;
            cloned->key_bias_ = key_bias_;
            cloned->value_bias_ = value_bias_;
            cloned->output_bias_ = output_bias_;
        }
        
        cloned->weights_initialized_ = true;
    }
    
    cloned->dropout_rate_ = dropout_rate_;
    cloned->scale_attention_ = scale_attention_;
    cloned->causal_mask_ = causal_mask_;
    
    return cloned;
}

std::string MultiHeadAttentionLayer::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"MultiHeadAttention\","
        << "\"model_dim\":" << model_dim_ << ","
        << "\"num_heads\":" << num_heads_ << ","
        << "\"head_dim\":" << head_dim_ << ","
        << "\"max_seq_length\":" << max_seq_length_ << ","
        << "\"use_bias\":" << (use_bias_ ? "true" : "false") << ","
        << "\"dropout_rate\":" << dropout_rate_ << ","
        << "\"scale_attention\":" << (scale_attention_ ? "true" : "false") << ","
        << "\"causal_mask\":" << (causal_mask_ ? "true" : "false")
        << "}";
    return oss.str();
}

void MultiHeadAttentionLayer::serialize_weights(std::ofstream& file) const {
    // Write all weight matrices and biases
    auto write_tensor = [&](const Tensor& tensor) {
        size_t size = tensor.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(tensor.raw_data()), size * sizeof(double));
    };
    
    write_tensor(query_weights_);
    write_tensor(key_weights_);
    write_tensor(value_weights_);
    write_tensor(output_weights_);
    
    if (use_bias_) {
        write_tensor(query_bias_);
        write_tensor(key_bias_);
        write_tensor(value_bias_);
        write_tensor(output_bias_);
    }
}

void MultiHeadAttentionLayer::deserialize_weights(std::ifstream& file) {
    auto read_tensor = [&](Tensor& tensor) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (size != tensor.size()) {
            throw std::runtime_error("Weight size mismatch during deserialization");
        }
        file.read(reinterpret_cast<char*>(tensor.raw_data()), size * sizeof(double));
    };
    
    read_tensor(query_weights_);
    read_tensor(key_weights_);
    read_tensor(value_weights_);
    read_tensor(output_weights_);
    
    if (use_bias_) {
        read_tensor(query_bias_);
        read_tensor(key_bias_);
        read_tensor(value_bias_);
        read_tensor(output_bias_);
    }
}

size_t MultiHeadAttentionLayer::get_weights_size() const {
    size_t total_size = 4 * model_dim_ * model_dim_; // 4 weight matrices
    
    if (use_bias_) {
        total_size += 4 * model_dim_; // 4 bias vectors
    }
    
    return total_size;
}

// ================================================================================================
// Simplified implementations for other classes (LayerNormalization, etc.)
// ================================================================================================

LayerNormalizationLayer::LayerNormalizationLayer(size_t feature_dim, double epsilon)
    : feature_dim_(feature_dim), epsilon_(epsilon) {
    
    gamma_ = Tensor::ones({feature_dim_});
    beta_ = Tensor::zeros({feature_dim_});
    gamma_grad_ = Tensor::zeros({feature_dim_});
    beta_grad_ = Tensor::zeros({feature_dim_});
    weights_initialized_ = true;
}

void LayerNormalizationLayer::initialize_weights() {
    gamma_ = Tensor::ones({feature_dim_});
    beta_ = Tensor::zeros({feature_dim_});
    gamma_grad_ = Tensor::zeros({feature_dim_});
    beta_grad_ = Tensor::zeros({feature_dim_});
    weights_initialized_ = true;
}

Tensor LayerNormalizationLayer::forward_tensor(const Tensor& input) {
    validate_input_tensor(input);
    last_input_ = input;
    
    // Compute mean and variance along the last dimension
    auto mean_var = compute_mean_variance(input);
    last_mean_ = std::get<0>(mean_var);
    last_variance_ = std::get<1>(mean_var);
    
    // Normalize
    Tensor normalized = Tensor::zeros(input.shape());
    auto feature_dim = input.size(input.ndim() - 1);
    
    for (size_t i = 0; i < input.size(); ++i) {
        size_t feature_idx = i % feature_dim;
        size_t sample_idx = i / feature_dim;
        
        double mean_val = last_mean_.data()[sample_idx];
        double var_val = last_variance_.data()[sample_idx];
        double norm_val = (input.data()[i] - mean_val) / std::sqrt(var_val + epsilon_);
        
        normalized.data()[i] = gamma_.data()[feature_idx] * norm_val + beta_.data()[feature_idx];
    }
    
    last_normalized_ = normalized;
    return normalized;
}

std::tuple<Tensor, Tensor> LayerNormalizationLayer::compute_mean_variance(const Tensor& input) const {
    auto batch_size = input.size() / feature_dim_;
    
    Tensor mean({batch_size});
    Tensor variance({batch_size});
    
    // Compute mean for each sample
    for (size_t b = 0; b < batch_size; ++b) {
        double sum = 0.0;
        for (size_t f = 0; f < feature_dim_; ++f) {
            sum += input.data()[b * feature_dim_ + f];
        }
        mean.data()[b] = sum / static_cast<double>(feature_dim_);
    }
    
    // Compute variance for each sample
    for (size_t b = 0; b < batch_size; ++b) {
        double sum_sq_diff = 0.0;
        double mean_val = mean.data()[b];
        
        for (size_t f = 0; f < feature_dim_; ++f) {
            double diff = input.data()[b * feature_dim_ + f] - mean_val;
            sum_sq_diff += diff * diff;
        }
        variance.data()[b] = sum_sq_diff / static_cast<double>(feature_dim_);
    }
    
    return std::make_tuple(mean, variance);
}

void LayerNormalizationLayer::validate_input_tensor(const Tensor& input) const {
    if (input.size() % feature_dim_ != 0) {
        throw std::invalid_argument("Input size must be divisible by feature dimension");
    }
}

Tensor LayerNormalizationLayer::backward_tensor(const Tensor& grad_output) {
    if (!weights_initialized_) {
        throw std::runtime_error("Cannot compute gradients: weights not initialized");
    }
    
    auto batch_size = last_input_.size(0);
    auto seq_len = last_input_.size(1);
    auto feature_dim = last_input_.size(2);
    
    // Gradient w.r.t. normalized input
    Tensor grad_normalized = Tensor::zeros(last_normalized_.shape());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_normalized.data()[i] = grad_output.data()[i] * gamma_.data()[i % feature_dim_];
    }
    
    // Gradient w.r.t. gamma and beta
    gamma_grad_ = Tensor::zeros({feature_dim_});
    beta_grad_ = Tensor::zeros({feature_dim_});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t f = 0; f < feature_dim; ++f) {
                gamma_grad_({f}) += grad_output({b, s, f}) * last_normalized_({b, s, f});
                beta_grad_({f}) += grad_output({b, s, f});
            }
        }
    }
    
    // Gradient w.r.t. input (complex layer norm backward pass)
    Tensor grad_input = Tensor::zeros(last_input_.shape());
    double N = static_cast<double>(feature_dim);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            double mean = last_mean_({b, s});
            double var = last_variance_({b, s});
            double std_inv = 1.0 / std::sqrt(var + epsilon_);
            
            // Compute gradient sums
            double grad_mean = 0.0;
            double grad_var = 0.0;
            
            for (size_t f = 0; f < feature_dim; ++f) {
                double x_centered = last_input_({b, s, f}) - mean;
                grad_mean += grad_normalized({b, s, f});
                grad_var += grad_normalized({b, s, f}) * x_centered;
            }
            
            grad_mean *= -std_inv;
            grad_var *= -0.5 * std_inv * std_inv * std_inv;
            
            // Compute final gradients
            for (size_t f = 0; f < feature_dim; ++f) {
                double x_centered = last_input_({b, s, f}) - mean;
                grad_input({b, s, f}) = std_inv * grad_normalized({b, s, f}) + 
                                      (grad_var * 2.0 * x_centered + grad_mean) / N;
            }
        }
    }
    
    return grad_input;
}

Matrix LayerNormalizationLayer::forward(const Matrix& input) {
    // Convert to tensor and back - simplified implementation
    Tensor input_tensor = Tensor::from_matrix(input);
    Tensor output_tensor = forward_tensor(input_tensor);
    return output_tensor.to_matrix();
}

Matrix LayerNormalizationLayer::backward(const Matrix& gradient) {
    // Simplified - return identity for now
    return gradient;
}

void LayerNormalizationLayer::update_weights(double learning_rate) {
    for (size_t i = 0; i < gamma_.size(); ++i) {
        gamma_.data()[i] -= learning_rate * gamma_grad_.data()[i];
        beta_.data()[i] -= learning_rate * beta_grad_.data()[i];
    }
}

std::unique_ptr<Layer> LayerNormalizationLayer::clone() const {
    auto cloned = std::make_unique<LayerNormalizationLayer>(feature_dim_, epsilon_);
    cloned->gamma_ = gamma_;
    cloned->beta_ = beta_;
    return cloned;
}

std::string LayerNormalizationLayer::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"LayerNormalization\","
        << "\"feature_dim\":" << feature_dim_ << ","
        << "\"epsilon\":" << epsilon_
        << "}";
    return oss.str();
}

void LayerNormalizationLayer::serialize_weights(std::ofstream& file) const {
    size_t size = gamma_.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(gamma_.raw_data()), size * sizeof(double));
    file.write(reinterpret_cast<const char*>(beta_.raw_data()), size * sizeof(double));
}

void LayerNormalizationLayer::deserialize_weights(std::ifstream& file) {
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (size != gamma_.size()) {
        throw std::runtime_error("Weight size mismatch during deserialization");
    }
    file.read(reinterpret_cast<char*>(gamma_.raw_data()), size * sizeof(double));
    file.read(reinterpret_cast<char*>(beta_.raw_data()), size * sizeof(double));
}

size_t LayerNormalizationLayer::get_weights_size() const {
    return 2 * feature_dim_; // gamma and beta
}

// ================================================================================================
// PositionalEncoding Implementation
// ================================================================================================

PositionalEncoding::PositionalEncoding(size_t max_length, size_t model_dim)
    : max_length_(max_length), model_dim_(model_dim) {
    encoding_table_ = sinusoidal_encoding(max_length, model_dim);
}

Tensor PositionalEncoding::sinusoidal_encoding(size_t max_length, size_t model_dim) {
    Tensor encoding({max_length, model_dim});
    
    for (size_t pos = 0; pos < max_length; ++pos) {
        for (size_t i = 0; i < model_dim; ++i) {
            double angle = static_cast<double>(pos) / std::pow(10000.0, (2.0 * (i / 2)) / static_cast<double>(model_dim));
            
            if (i % 2 == 0) {
                encoding({pos, i}) = std::sin(angle);
            } else {
                encoding({pos, i}) = std::cos(angle);
            }
        }
    }
    
    return encoding;
}

Tensor PositionalEncoding::get_encoding(size_t sequence_length) const {
    if (sequence_length > max_length_) {
        throw std::invalid_argument("Sequence length exceeds maximum length");
    }
    
    return encoding_table_.slice({{0, sequence_length}, {0, model_dim_}});
}

void PositionalEncoding::add_to_tensor(Tensor& input) const {
    auto seq_len = input.size(input.ndim() - 2);
    auto encoding = get_encoding(seq_len);
    
    // Add positional encoding to input (broadcasting)
    for (size_t i = 0; i < input.size(); ++i) {
        size_t pos_idx = (i / model_dim_) % seq_len;
        size_t dim_idx = i % model_dim_;
        input.data()[i] += encoding({pos_idx, dim_idx});
    }
}

// ================================================================================================
// TransformerFeedForwardLayer Implementation
// ================================================================================================

TransformerFeedForwardLayer::TransformerFeedForwardLayer(size_t input_dim, size_t hidden_dim,
                                                        const std::string& activation)
    : input_dim_(input_dim)
    , hidden_dim_(hidden_dim)
    , activation_(activation) {
    
    // Initialize weight tensors
    weights1_ = Tensor::zeros({input_dim_, hidden_dim_});
    bias1_ = Tensor::zeros({hidden_dim_});
    weights2_ = Tensor::zeros({hidden_dim_, input_dim_});
    bias2_ = Tensor::zeros({input_dim_});
    
    // Initialize gradient tensors
    weights1_grad_ = Tensor::zeros({input_dim_, hidden_dim_});
    bias1_grad_ = Tensor::zeros({hidden_dim_});
    weights2_grad_ = Tensor::zeros({hidden_dim_, input_dim_});
    bias2_grad_ = Tensor::zeros({input_dim_});
}

void TransformerFeedForwardLayer::initialize_weights(const std::string& method) {
    double scale1 = 1.0, scale2 = 1.0;
    
    if (method == "xavier" || method == "glorot") {
        scale1 = std::sqrt(6.0 / (input_dim_ + hidden_dim_));
        scale2 = std::sqrt(6.0 / (hidden_dim_ + input_dim_));
    } else if (method == "he" || method == "kaiming") {
        scale1 = std::sqrt(2.0 / input_dim_);
        scale2 = std::sqrt(2.0 / hidden_dim_);
    } else {
        scale1 = scale2 = 0.02;
    }
    
    weights1_ = Tensor::randn({input_dim_, hidden_dim_}, 0.0, scale1);
    weights2_ = Tensor::randn({hidden_dim_, input_dim_}, 0.0, scale2);
    bias1_ = Tensor::zeros({hidden_dim_});
    bias2_ = Tensor::zeros({input_dim_});
    
    weights_initialized_ = true;
}

Tensor TransformerFeedForwardLayer::apply_activation(const Tensor& input) const {
    Tensor output = input;
    
    if (activation_ == "relu") {
        for (size_t i = 0; i < output.size(); ++i) {
            output.data()[i] = std::max(0.0, output.data()[i]);
        }
    } else if (activation_ == "gelu") {
        for (size_t i = 0; i < output.size(); ++i) {
            double x = output.data()[i];
            output.data()[i] = 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        }
    } else if (activation_ == "swish") {
        for (size_t i = 0; i < output.size(); ++i) {
            double x = output.data()[i];
            output.data()[i] = x / (1.0 + std::exp(-x));
        }
    }
    
    return output;
}

Tensor TransformerFeedForwardLayer::apply_activation_derivative(const Tensor& input) const {
    Tensor output = input;
    
    if (activation_ == "relu") {
        for (size_t i = 0; i < output.size(); ++i) {
            output.data()[i] = (input.data()[i] > 0.0) ? 1.0 : 0.0;
        }
    } else if (activation_ == "gelu") {
        for (size_t i = 0; i < output.size(); ++i) {
            double x = input.data()[i];
            double tanh_term = std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x));
            double sech2_term = 1.0 - tanh_term * tanh_term;
            output.data()[i] = 0.5 * (1.0 + tanh_term) + 
                              0.5 * x * sech2_term * std::sqrt(2.0 / M_PI) * (1.0 + 0.134145 * x * x);
        }
    } else if (activation_ == "swish") {
        for (size_t i = 0; i < output.size(); ++i) {
            double x = input.data()[i];
            double sigmoid = 1.0 / (1.0 + std::exp(-x));
            output.data()[i] = sigmoid + x * sigmoid * (1.0 - sigmoid);
        }
    }
    
    return output;
}

Tensor TransformerFeedForwardLayer::forward_tensor(const Tensor& input) {
    if (!weights_initialized_) {
        initialize_weights();
    }
    
    last_input_ = input;
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto input_dim = input.size(2);
    
    // First linear transformation: input -> hidden
    Tensor input_2d = input.reshape({batch_size * seq_len, input_dim});
    Tensor hidden_2d = input_2d.matmul(weights1_);
    
    // Add bias
    for (size_t i = 0; i < hidden_2d.size(0); ++i) {
        for (size_t j = 0; j < hidden_dim_; ++j) {
            hidden_2d({i, j}) += bias1_({j});
        }
    }
    
    Tensor hidden = hidden_2d.reshape({batch_size, seq_len, hidden_dim_});
    
    // Apply activation
    last_hidden_ = apply_activation(hidden);
    
    // Second linear transformation: hidden -> output
    Tensor hidden_activated_2d = last_hidden_.reshape({batch_size * seq_len, hidden_dim_});
    Tensor output_2d = hidden_activated_2d.matmul(weights2_);
    
    // Add bias
    for (size_t i = 0; i < output_2d.size(0); ++i) {
        for (size_t j = 0; j < input_dim_; ++j) {
            output_2d({i, j}) += bias2_({j});
        }
    }
    
    return output_2d.reshape({batch_size, seq_len, input_dim_});
}

Tensor TransformerFeedForwardLayer::backward_tensor(const Tensor& grad_output) {
    if (!weights_initialized_) {
        throw std::runtime_error("Cannot compute gradients: weights not initialized");
    }
    
    auto batch_size = grad_output.size(0);
    auto seq_len = grad_output.size(1);
    
    // Gradient w.r.t. second linear layer
    Tensor grad_output_2d = grad_output.reshape({batch_size * seq_len, input_dim_});
    Tensor hidden_activated_2d = last_hidden_.reshape({batch_size * seq_len, hidden_dim_});
    
    // Gradients for weights2 and bias2
    weights2_grad_ = hidden_activated_2d.transpose({1, 0}).matmul(grad_output_2d);
    bias2_grad_ = grad_output_2d.sum(0);
    
    // Gradient w.r.t. hidden (activated)
    Tensor grad_hidden_activated_2d = grad_output_2d.matmul(weights2_.transpose({1, 0}));
    Tensor grad_hidden_activated = grad_hidden_activated_2d.reshape({batch_size, seq_len, hidden_dim_});
    
    // Gradient w.r.t. hidden (before activation)
    Tensor hidden_before_activation = last_hidden_; // This should be stored separately, but for now...
    Tensor activation_grad = apply_activation_derivative(hidden_before_activation);
    Tensor grad_hidden = grad_hidden_activated * activation_grad;
    
    // Gradient w.r.t. first linear layer
    Tensor grad_hidden_2d = grad_hidden.reshape({batch_size * seq_len, hidden_dim_});
    Tensor input_2d = last_input_.reshape({batch_size * seq_len, input_dim_});
    
    // Gradients for weights1 and bias1
    weights1_grad_ = input_2d.transpose({1, 0}).matmul(grad_hidden_2d);
    bias1_grad_ = grad_hidden_2d.sum(0);
    
    // Gradient w.r.t. input
    Tensor grad_input_2d = grad_hidden_2d.matmul(weights1_.transpose({1, 0}));
    return grad_input_2d.reshape({batch_size, seq_len, input_dim_});
}

Matrix TransformerFeedForwardLayer::forward(const Matrix& input) {
    // Convert to tensor
    auto input_tensor = Tensor::from_matrix(input);
    auto output_tensor = forward_tensor(input_tensor);
    return output_tensor.to_matrix();
}

Matrix TransformerFeedForwardLayer::backward(const Matrix& gradient) {
    // Convert to tensor and use tensor backward
    auto grad_tensor = Tensor::from_matrix(gradient);
    auto grad_input_tensor = backward_tensor(grad_tensor);
    return grad_input_tensor.to_matrix();
}

void TransformerFeedForwardLayer::update_weights(double learning_rate) {
    for (size_t i = 0; i < weights1_.size(); ++i) {
        weights1_.data()[i] -= learning_rate * weights1_grad_.data()[i];
        weights2_.data()[i] -= learning_rate * weights2_grad_.data()[i];
    }
    
    for (size_t i = 0; i < bias1_.size(); ++i) {
        bias1_.data()[i] -= learning_rate * bias1_grad_.data()[i];
        bias2_.data()[i] -= learning_rate * bias2_grad_.data()[i];
    }
}

std::unique_ptr<Layer> TransformerFeedForwardLayer::clone() const {
    auto cloned = std::make_unique<TransformerFeedForwardLayer>(input_dim_, hidden_dim_, activation_);
    
    if (weights_initialized_) {
        cloned->weights1_ = weights1_;
        cloned->bias1_ = bias1_;
        cloned->weights2_ = weights2_;
        cloned->bias2_ = bias2_;
        cloned->weights_initialized_ = true;
    }
    
    cloned->dropout_rate_ = dropout_rate_;
    
    return cloned;
}

std::string TransformerFeedForwardLayer::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"TransformerFeedForward\","
        << "\"input_dim\":" << input_dim_ << ","
        << "\"hidden_dim\":" << hidden_dim_ << ","
        << "\"activation\":\"" << activation_ << "\","
        << "\"dropout_rate\":" << dropout_rate_
        << "}";
    return oss.str();
}

void TransformerFeedForwardLayer::serialize_weights(std::ofstream& file) const {
    auto write_tensor = [&](const Tensor& tensor) {
        size_t size = tensor.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(tensor.raw_data()), size * sizeof(double));
    };
    
    write_tensor(weights1_);
    write_tensor(bias1_);
    write_tensor(weights2_);
    write_tensor(bias2_);
}

void TransformerFeedForwardLayer::deserialize_weights(std::ifstream& file) {
    auto read_tensor = [&](Tensor& tensor) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        tensor = Tensor(std::vector<size_t>{size});
        file.read(reinterpret_cast<char*>(tensor.raw_data()), size * sizeof(double));
    };
    
    read_tensor(weights1_);
    read_tensor(bias1_);
    read_tensor(weights2_);
    read_tensor(bias2_);
    
    weights_initialized_ = true;
}

size_t TransformerFeedForwardLayer::get_weights_size() const {
    return weights1_.size() + bias1_.size() + weights2_.size() + bias2_.size();
}

// ================================================================================================
// TransformerBlock Implementation
// ================================================================================================

TransformerBlock::TransformerBlock(size_t model_dim, size_t num_heads, size_t ff_hidden_dim, bool pre_norm)
    : model_dim_(model_dim)
    , pre_norm_(pre_norm) {
    
    attention_ = std::make_unique<MultiHeadAttentionLayer>(model_dim, num_heads);
    norm1_ = std::make_unique<LayerNormalizationLayer>(model_dim);
    norm2_ = std::make_unique<LayerNormalizationLayer>(model_dim);
    feed_forward_ = std::make_unique<TransformerFeedForwardLayer>(model_dim, ff_hidden_dim);
    
    // Initialize all sub-layers
    attention_->initialize_weights();
    norm1_->initialize_weights();
    norm2_->initialize_weights();
    feed_forward_->initialize_weights();
}

Tensor TransformerBlock::forward_tensor(const Tensor& input, const Tensor* attention_mask) {
    last_input_ = input;
    
    if (pre_norm_) {
        // Pre-normalization architecture
        // Attention block
        Tensor norm1_output = norm1_->forward_tensor(input);
        Tensor attention_output = attention_->forward_tensor_self_attention(norm1_output, attention_mask);
        Tensor residual1 = input + attention_output; // Residual connection
        
        // Feed-forward block
        Tensor norm2_output = norm2_->forward_tensor(residual1);
        Tensor ff_output = feed_forward_->forward_tensor(norm2_output);
        Tensor output = residual1 + ff_output; // Residual connection
        
        last_attention_output_ = attention_output;
        last_norm1_output_ = norm1_output;
        last_ff_output_ = ff_output;
        
        return output;
    } else {
        // Post-normalization architecture (original transformer)
        // Attention block
        Tensor attention_output = attention_->forward_tensor_self_attention(input, attention_mask);
        Tensor residual1 = input + attention_output; // Residual connection
        Tensor norm1_output = norm1_->forward_tensor(residual1);
        
        // Feed-forward block
        Tensor ff_output = feed_forward_->forward_tensor(norm1_output);
        Tensor residual2 = norm1_output + ff_output; // Residual connection
        Tensor output = norm2_->forward_tensor(residual2);
        
        last_attention_output_ = attention_output;
        last_norm1_output_ = norm1_output;
        last_ff_output_ = ff_output;
        
        return output;
    }
}

Tensor TransformerBlock::backward_tensor(const Tensor& grad_output) {
    // This is a simplified backward pass - in practice, you'd need to track
    // the exact computation graph and compute gradients properly
    
    if (pre_norm_) {        // Pre-norm backward pass
        auto grad_ff = feed_forward_->backward_tensor(grad_output);
        auto grad_norm2 = norm2_->backward_tensor(grad_ff);
        auto grad_residual1 = grad_output + grad_norm2;
        
        auto [grad_q, grad_k, grad_v] = attention_->backward_tensor(grad_residual1);
        auto grad_attention_total = grad_q + grad_k + grad_v; // For self-attention
        auto grad_norm1 = norm1_->backward_tensor(grad_attention_total);
        auto grad_input = grad_residual1 + grad_norm1;
        
        return grad_input;
    } else {
        // Post-norm backward pass
        auto grad_norm2 = norm2_->backward_tensor(grad_output);
        auto grad_ff = feed_forward_->backward_tensor(grad_norm2);
        auto grad_norm1 = norm1_->backward_tensor(grad_norm2 + grad_ff);
        
        auto [grad_q, grad_k, grad_v] = attention_->backward_tensor(grad_norm1);
        auto grad_attention_total = grad_q + grad_k + grad_v; // For self-attention
        auto grad_input = grad_norm1 + grad_attention_total;
        
        return grad_input;
    }
}

Matrix TransformerBlock::forward(const Matrix& input) {
    auto input_tensor = Tensor::from_matrix(input);
    auto output_tensor = forward_tensor(input_tensor);
    return output_tensor.to_matrix();
}

Matrix TransformerBlock::backward(const Matrix& gradient) {
    auto grad_tensor = Tensor::from_matrix(gradient);
    auto grad_input_tensor = backward_tensor(grad_tensor);
    return grad_input_tensor.to_matrix();
}

void TransformerBlock::update_weights(double learning_rate) {
    attention_->update_weights(learning_rate);
    norm1_->update_weights(learning_rate);
    norm2_->update_weights(learning_rate);
    feed_forward_->update_weights(learning_rate);
}

void TransformerBlock::set_dropout_rate(double rate) {
    dropout_rate_ = rate;
    attention_->set_dropout_rate(rate);
    feed_forward_->set_dropout_rate(rate);
}

void TransformerBlock::enable_causal_mask(bool enable) {
    attention_->enable_causal_mask(enable);
}

std::unique_ptr<Layer> TransformerBlock::clone() const {
    auto cloned = std::make_unique<TransformerBlock>(model_dim_, 
                                                    attention_->input_size(), 
                                                    feed_forward_->input_size(), 
                                                    pre_norm_);
    
    cloned->attention_ = std::unique_ptr<MultiHeadAttentionLayer>(
        dynamic_cast<MultiHeadAttentionLayer*>(attention_->clone().release()));
    cloned->norm1_ = std::unique_ptr<LayerNormalizationLayer>(
        dynamic_cast<LayerNormalizationLayer*>(norm1_->clone().release()));
    cloned->norm2_ = std::unique_ptr<LayerNormalizationLayer>(
        dynamic_cast<LayerNormalizationLayer*>(norm2_->clone().release()));
    cloned->feed_forward_ = std::unique_ptr<TransformerFeedForwardLayer>(
        dynamic_cast<TransformerFeedForwardLayer*>(feed_forward_->clone().release()));
    
    cloned->dropout_rate_ = dropout_rate_;
    
    return cloned;
}

std::string TransformerBlock::serialize_to_json() const {
    std::ostringstream oss;
    oss << "{"
        << "\"type\":\"TransformerBlock\","
        << "\"model_dim\":" << model_dim_ << ","
        << "\"pre_norm\":" << (pre_norm_ ? "true" : "false") << ","
        << "\"dropout_rate\":" << dropout_rate_ << ","
        << "\"attention\":" << attention_->serialize_to_json() << ","
        << "\"norm1\":" << norm1_->serialize_to_json() << ","
        << "\"norm2\":" << norm2_->serialize_to_json() << ","
        << "\"feed_forward\":" << feed_forward_->serialize_to_json()
        << "}";
    return oss.str();
}

void TransformerBlock::serialize_weights(std::ofstream& file) const {
    attention_->serialize_weights(file);
    norm1_->serialize_weights(file);
    norm2_->serialize_weights(file);
    feed_forward_->serialize_weights(file);
}

void TransformerBlock::deserialize_weights(std::ifstream& file) {
    attention_->deserialize_weights(file);
    norm1_->deserialize_weights(file);
    norm2_->deserialize_weights(file);
    feed_forward_->deserialize_weights(file);
}

size_t TransformerBlock::get_weights_size() const {
    return attention_->get_weights_size() + 
           norm1_->get_weights_size() + 
           norm2_->get_weights_size() + 
           feed_forward_->get_weights_size();
}
} // namespace ai
} // namespace asekioml
