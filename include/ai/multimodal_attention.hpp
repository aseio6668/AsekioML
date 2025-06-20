#pragma once

#include "../tensor.hpp"
#include "attention_layers.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace asekioml {
namespace ai {

/**
 * @brief Modality types for cross-modal processing
 */
enum class Modality {
    TEXT,
    IMAGE, 
    AUDIO,
    VIDEO
};

/**
 * @brief Multi-modal tensor with modality metadata
 */
class MultiModalTensor {
public:
    MultiModalTensor(const Tensor& data, Modality modality, const std::string& description = "");
    
    // Getters
    const Tensor& data() const { return data_; }
    Tensor& data() { return data_; }
    Modality modality() const { return modality_; }
    const std::string& description() const { return description_; }
    
    // Modality-specific shape interpretation
    std::vector<size_t> get_sequence_shape() const;  // [batch, seq_len, features]
    std::vector<size_t> get_spatial_shape() const;   // [batch, height, width, channels]
    std::vector<size_t> get_temporal_shape() const;  // [batch, time, features]    // Convenience methods
    bool is_text() const { return modality_ == Modality::TEXT; }
    bool is_image() const { return modality_ == Modality::IMAGE; }
    bool is_audio() const { return modality_ == Modality::AUDIO; }
    bool is_video() const { return modality_ == Modality::VIDEO; }

private:
    Tensor data_;
    Modality modality_;
    std::string description_;
};

/**
 * @brief Cross-modal attention mechanism for different modality pairs
 */
class CrossModalAttention {
public:
    /**
     * @brief Initialize cross-modal attention
     * @param source_dim Dimension of source modality features
     * @param target_dim Dimension of target modality features  
     * @param hidden_dim Hidden dimension for attention computation
     * @param num_heads Number of attention heads
     */
    CrossModalAttention(size_t source_dim, size_t target_dim, size_t hidden_dim, size_t num_heads = 8);
      /**
     * @brief Forward pass with cross-modal attention
     * @param source Source modality tensor [batch, source_seq, source_dim]
     * @param target Target modality tensor [batch, target_seq, target_dim]
     * @param source_modality Type of source modality
     * @param target_modality Type of target modality
     * @return Attended target features [batch, target_seq, hidden_dim]
     */    Tensor forward(const Tensor& source, const Tensor& target, 
                  Modality source_modality, Modality target_modality);
      /**
     * @brief Bidirectional cross-modal attention
     * @param modal1 First modality tensor
     * @param modal2 Second modality tensor  
     * @param modality1 Type of first modality
     * @param modality2 Type of second modality
     * @return Pair of attended features for both modalities
     */    std::pair<Tensor, Tensor> bidirectional_attention(const Tensor& modal1, const Tensor& modal2,
                                                      Modality modality1, Modality modality2);
    
    /**
     * @brief Get attention weights for visualization
     * @return Attention weight matrix [batch, num_heads, target_seq, source_seq]
     */
    Tensor get_attention_weights() const;

private:
    size_t source_dim_;
    size_t target_dim_;
    size_t hidden_dim_;
    size_t num_heads_;
    
    // Projection matrices for different modality combinations
    std::unordered_map<std::string, Tensor> projection_matrices_;
      // Core attention mechanism
    std::unique_ptr<MultiHeadAttentionLayer> attention_layer_;
      // Last computed attention weights
    mutable Tensor last_attention_weights_;
    
    // Helper methods
    std::string get_modality_pair_key(Modality source, Modality target) const;
    Tensor project_modality_features(const Tensor& features, Modality source, Modality target);
    void initialize_projection_matrices();
};

/**
 * @brief Multi-modal fusion strategies
 */
enum class FusionStrategy {
    EARLY_FUSION,      // Concatenate features before processing
    LATE_FUSION,       // Process separately, combine outputs
    INTERMEDIATE_FUSION, // Multiple fusion points throughout network
    ATTENTION_FUSION   // Use attention to selectively combine features
};

/**
 * @brief Multi-modal feature fusion network
 */
class MultiModalFusion {
public:    /**
     * @brief Initialize fusion network
     * @param strategy Fusion strategy to use
     * @param modality_dims Dimensions for each modality
     * @param output_dim Output dimension after fusion
     */
    MultiModalFusion(FusionStrategy strategy, 
                    const std::unordered_map<Modality, size_t>& modality_dims,
                    size_t output_dim);
      /**
     * @brief Fuse multiple modality features
     * @param modality_features Map of modality to feature tensors
     * @return Fused feature representation
     */
    Tensor fuse(const std::unordered_map<Modality, Tensor>& modality_features);
      /**
     * @brief Fuse with attention weights
     * @param modality_features Map of modality to feature tensors
     * @param attention_weights Optional attention weights for each modality
     * @return Fused feature representation with attention scores
     */
    std::pair<Tensor, Tensor> fuse_with_attention(        const std::unordered_map<Modality, Tensor>& modality_features,
        const std::unordered_map<Modality, Tensor>& attention_weights = {});

private:
    FusionStrategy strategy_;
    std::unordered_map<Modality, size_t> modality_dims_;
    size_t output_dim_;    
    // Fusion-specific components
    std::unique_ptr<CrossModalAttention> fusion_attention_;
    std::unordered_map<Modality, Tensor> projection_weights_;
    Tensor fusion_weights_;
    
    // Strategy-specific implementations
    Tensor early_fusion(const std::unordered_map<Modality, Tensor>& features);
    Tensor late_fusion(const std::unordered_map<Modality, Tensor>& features);
    Tensor intermediate_fusion(const std::unordered_map<Modality, Tensor>& features);
    Tensor attention_fusion(const std::unordered_map<Modality, Tensor>& features);
    
    void initialize_fusion_components();
};

/**
 * @brief Multi-modal transformer with cross-attention
 */
class MultiModalTransformer {
public:    /**
     * @brief Initialize multi-modal transformer
     * @param modality_configs Configuration for each modality
     * @param hidden_dim Hidden dimension for transformer
     * @param num_layers Number of transformer layers
     * @param num_heads Number of attention heads
     */
    MultiModalTransformer(const std::unordered_map<Modality, size_t>& modality_configs,
                         size_t hidden_dim, size_t num_layers, size_t num_heads);
    
    /**
     * @brief Forward pass through multi-modal transformer
     * @param inputs Map of modality to input tensors
     * @return Map of modality to output tensors
     */    std::unordered_map<Modality, Tensor> forward(
        const std::unordered_map<Modality, Tensor>& inputs);
    
    /**
     * @brief Generate output for specific modality conditioned on others
     * @param target_modality Modality to generate
     * @param conditioning_modalities Input modalities for conditioning
     * @param max_length Maximum sequence length to generate
     * @return Generated sequence for target modality
     */    Tensor generate_conditioned(Modality target_modality,
                               const std::unordered_map<Modality, Tensor>& conditioning_modalities,
                               size_t max_length);

private:
    std::unordered_map<Modality, size_t> modality_configs_;
    size_t hidden_dim_;
    size_t num_layers_;
    size_t num_heads_;
    
    // Per-modality input projections
    std::unordered_map<Modality, Tensor> input_projections_;
    
    // Cross-modal attention layers
    std::vector<std::unique_ptr<CrossModalAttention>> cross_attention_layers_;
      // Self-attention layers for each modality
    std::unordered_map<Modality, std::vector<std::unique_ptr<MultiHeadAttentionLayer>>> self_attention_layers_;
    
    // Layer normalization and feed-forward networks
    std::unordered_map<Modality, std::vector<Tensor>> layer_norms_;
    std::unordered_map<Modality, std::vector<Tensor>> feed_forward_weights_;
    
    void initialize_layers();
    Tensor apply_layer_norm(const Tensor& input, const Tensor& norm_weights);
    Tensor apply_feed_forward(const Tensor& input, const Tensor& weights);
};

/**
 * @brief Utility functions for multi-modal processing
 */
namespace MultiModalUtils {    /**
     * @brief Convert single-modal tensor to multi-modal tensor
     */    MultiModalTensor to_multimodal(const Tensor& tensor, Modality modality, 
                                  const std::string& description = "");
    
    /**
     * @brief Align sequences of different modalities to same length
     */
    std::unordered_map<Modality, Tensor> align_sequences(
        const std::unordered_map<Modality, Tensor>& sequences);
    
    /**
     * @brief Compute cross-modal similarity matrix
     */
    Tensor compute_cross_modal_similarity(const Tensor& features1, const Tensor& features2);
    
    /**
     * @brief Interpolate between modalities in shared embedding space
     */
    Tensor interpolate_modalities(const Tensor& modal1_features, const Tensor& modal2_features, 
                                 double interpolation_factor);
}

} // namespace ai
} // namespace asekioml
