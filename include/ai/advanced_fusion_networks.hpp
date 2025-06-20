#pragma once

#include "multimodal_attention.hpp"
#include "attention_layers.hpp"
#include "../tensor.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace asekioml {
namespace ai {

/**
 * @brief Custom hash function for std::pair<Modality, Modality>
 */
struct PairModalityHash {
    std::size_t operator()(const std::pair<Modality, Modality>& pair) const {
        auto h1 = std::hash<int>{}(static_cast<int>(pair.first));
        auto h2 = std::hash<int>{}(static_cast<int>(pair.second));
        return h1 ^ (h2 << 1);  // Combine hashes
    }
};

/**
 * @brief Advanced fusion network architectures for multi-modal learning
 */

/**
 * @brief Types of feature fusion at different network levels
 */
enum class FusionLevel {
    INPUT_LEVEL,        // Fusion at input (early fusion)
    FEATURE_LEVEL,      // Fusion at intermediate feature maps
    DECISION_LEVEL,     // Fusion at decision/output level (late fusion)
    HIERARCHICAL        // Multi-level fusion throughout network
};

/**
 * @brief Advanced multi-modal fusion architectures
 */
class AdvancedMultiModalFusion {
public:
    /**
     * @brief Configuration for fusion network
     */    struct FusionConfig {
        std::unordered_map<Modality, size_t> input_dims;
        std::unordered_map<Modality, std::vector<size_t>> intermediate_dims;
        size_t output_dim;
        FusionLevel fusion_level;
        size_t num_fusion_layers;
        double dropout_rate = 0.1;
        bool use_residual_connections = true;
        bool use_layer_norm = true;
    };

    AdvancedMultiModalFusion(const FusionConfig& config);

    /**
     * @brief Forward pass through advanced fusion network
     * @param inputs Map of modality to feature tensors
     * @param fusion_weights Optional per-modality fusion weights
     * @return Fused representation with intermediate features
     */    struct FusionOutput {
        Tensor fused_features;
        std::unordered_map<Modality, std::vector<Tensor>> intermediate_features;
        std::unordered_map<Modality, Tensor> attention_maps;
        Tensor fusion_confidence;
    };

    FusionOutput forward(const std::unordered_map<Modality, Tensor>& inputs,
                        const std::unordered_map<Modality, float>& fusion_weights = {});

    /**
     * @brief Hierarchical fusion with multiple fusion points
     */    FusionOutput hierarchical_fusion(const std::unordered_map<Modality, Tensor>& inputs);    /**
     * @brief Adaptive fusion that learns optimal fusion weights
     */
    FusionOutput adaptive_fusion(const std::unordered_map<Modality, Tensor>& inputs);

private:
    FusionConfig config_;
    std::vector<std::unique_ptr<CrossModalAttention>> cross_attention_layers_;
    std::unordered_map<Modality, std::vector<Tensor>> projection_layers_;
    std::vector<Tensor> fusion_weights_;
    std::vector<std::unique_ptr<LayerNormalizationLayer>> layer_norms_;
    
    void initialize_fusion_layers();    Tensor apply_fusion_layer(const std::unordered_map<Modality, Tensor>& features, size_t layer_idx);
    Tensor compute_fusion_confidence(const std::unordered_map<Modality, Tensor>& features);
};

/**
 * @brief Unified multi-modal representation learning
 */
class UnifiedMultiModalEncoder {
public:
    /**
     * @brief Configuration for unified encoder
     */
    struct EncoderConfig {
        std::unordered_map<Modality, size_t> modality_dims;
        size_t unified_dim;
        size_t num_encoder_layers;
        size_t num_attention_heads;
        double temperature = 0.07;  // For contrastive learning
        bool use_momentum_encoder = true;
        double momentum = 0.999;
    };

    UnifiedMultiModalEncoder(const EncoderConfig& config);

    /**
     * @brief Encode multiple modalities into unified representation space
     * @param inputs Map of modality to feature tensors
     * @return Unified representations and modality-specific features
     */
    struct EncodingOutput {
        std::unordered_map<Modality, Tensor> unified_representations;
        std::unordered_map<Modality, Tensor> modality_specific_features;
        Tensor cross_modal_similarity;
        std::unordered_map<std::pair<Modality, Modality>, Tensor, PairModalityHash> pairwise_similarities;
    };

    EncodingOutput encode(const std::unordered_map<Modality, Tensor>& inputs);

    /**
     * @brief Compute cross-modal contrastive loss
     */
    float compute_contrastive_loss(const EncodingOutput& encoding,
                                  const std::vector<bool>& positive_pairs);

    /**
     * @brief Update momentum encoder (for self-supervised learning)
     */
    void update_momentum_encoder();

private:
    EncoderConfig config_;
    std::unordered_map<Modality, std::vector<Tensor>> modality_encoders_;
    std::unordered_map<Modality, std::vector<Tensor>> momentum_encoders_;
    std::unique_ptr<MultiModalTransformer> unified_transformer_;
    
    void initialize_encoders();
    Tensor project_to_unified_space(const Tensor& features, Modality modality);
    float compute_similarity(const Tensor& repr1, const Tensor& repr2);
};

/**
 * @brief Cross-modal consistency losses and training utilities
 */
class CrossModalConsistency {
public:
    /**
     * @brief Types of consistency losses
     */
    enum class ConsistencyType {
        SEMANTIC_CONSISTENCY,    // Same semantic content across modalities
        TEMPORAL_CONSISTENCY,    // Temporal alignment for video/audio
        SPATIAL_CONSISTENCY,     // Spatial alignment for vision/language
        FEATURE_CONSISTENCY      // Feature-level consistency
    };

    /**
     * @brief Configuration for consistency training
     */
    struct ConsistencyConfig {
        std::vector<ConsistencyType> loss_types;
        std::unordered_map<ConsistencyType, float> loss_weights;
        float margin = 0.2;  // For ranking losses
        float temperature = 0.1;  // For softmax-based losses
    };

    CrossModalConsistency(const ConsistencyConfig& config);

    /**
     * @brief Compute cross-modal consistency losses
     * @param representations Map of modality to feature representations
     * @param ground_truth Optional ground truth alignments
     * @return Total consistency loss and individual loss components
     */
    struct ConsistencyLoss {
        float total_loss;
        std::unordered_map<ConsistencyType, float> individual_losses;
        std::unordered_map<std::pair<Modality, Modality>, float, PairModalityHash> pairwise_losses;
    };

    ConsistencyLoss compute_consistency_loss(
        const std::unordered_map<Modality, Tensor>& representations,
        const std::unordered_map<std::pair<Modality, Modality>, std::vector<bool>, PairModalityHash>& ground_truth = {});

    /**
     * @brief Semantic consistency loss (ensures same semantic content)
     */
    float semantic_consistency_loss(const std::unordered_map<Modality, Tensor>& representations);

    /**
     * @brief Temporal consistency loss (for sequential data)
     */
    float temporal_consistency_loss(const std::unordered_map<Modality, Tensor>& representations);

    /**
     * @brief Spatial consistency loss (for spatial alignment)
     */
    float spatial_consistency_loss(const std::unordered_map<Modality, Tensor>& representations);

private:
    ConsistencyConfig config_;
    
    float contrastive_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative);
    float triplet_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative);
    float info_nce_loss(const std::vector<Tensor>& representations);
};

/**
 * @brief Advanced multi-modal training utilities
 */
class MultiModalTrainer {
public:
    /**
     * @brief Training configuration
     */
    struct TrainingConfig {
        float learning_rate = 1e-4;
        float weight_decay = 1e-5;
        size_t batch_size = 32;
        size_t max_epochs = 100;
        float gradient_clip_norm = 1.0;
        bool use_curriculum_learning = false;
        bool use_mixed_precision = false;
        size_t warmup_steps = 1000;
    };

    MultiModalTrainer(const TrainingConfig& config);

    /**
     * @brief Training step for multi-modal networks
     * @param model The multi-modal model to train
     * @param batch Batch of multi-modal data
     * @param loss_fn Loss function for the task
     * @return Training metrics
     */
    struct TrainingMetrics {
        float loss;
        float accuracy;
        std::unordered_map<Modality, float> modality_losses;
        std::unordered_map<std::string, float> additional_metrics;
    };

    TrainingMetrics training_step(
        AdvancedMultiModalFusion& model,
        const std::unordered_map<Modality, Tensor>& batch,
        std::function<float(const Tensor&, const Tensor&)> loss_fn);

    /**
     * @brief Curriculum learning schedule
     */
    void update_curriculum_schedule(size_t epoch);

    /**
     * @brief Learning rate scheduling
     */
    void update_learning_rate(size_t step);

private:
    TrainingConfig config_;
    float current_learning_rate_;
    size_t current_step_;
    
    void clip_gradients(float max_norm);
    void apply_gradient_updates();
};

/**
 * @brief Multi-modal data augmentation
 */
class MultiModalAugmentation {
public:
    /**
     * @brief Augmentation techniques for different modalities
     */
    enum class AugmentationType {
        CROSS_MODAL_MIXUP,       // Mix features across modalities
        MODALITY_DROPOUT,        // Randomly drop modalities
        TEMPORAL_SHIFTING,       // Shift temporal sequences
        NOISE_INJECTION,         // Add controlled noise
        FEATURE_MASKING          // Mask parts of features
    };

    /**
     * @brief Configuration for augmentation
     */
    struct AugmentationConfig {
        std::vector<AugmentationType> enabled_augmentations;
        std::unordered_map<AugmentationType, float> augmentation_probs;
        float mixup_alpha = 0.2;
        float dropout_rate = 0.1;
        float noise_std = 0.01;
    };

    MultiModalAugmentation(const AugmentationConfig& config);

    /**
     * @brief Apply augmentations to multi-modal batch
     */
    std::unordered_map<Modality, Tensor> augment_batch(
        const std::unordered_map<Modality, Tensor>& batch);

    /**
     * @brief Cross-modal mixup augmentation
     */
    std::unordered_map<Modality, Tensor> cross_modal_mixup(
        const std::unordered_map<Modality, Tensor>& batch1,
        const std::unordered_map<Modality, Tensor>& batch2,
        float lambda);

private:
    AugmentationConfig config_;
    
    Tensor apply_modality_dropout(const Tensor& features, float dropout_rate);
    Tensor apply_temporal_shifting(const Tensor& features, int max_shift);
    Tensor apply_noise_injection(const Tensor& features, float noise_std);
    Tensor apply_feature_masking(const Tensor& features, float mask_ratio);
};

/**
 * @brief Evaluation metrics for multi-modal models
 */
class MultiModalEvaluator {
public:
    /**
     * @brief Evaluation metrics
     */
    struct EvaluationMetrics {
        // Cross-modal retrieval metrics
        float text_to_image_recall_at_k;
        float image_to_text_recall_at_k;
        float cross_modal_map;  // Mean Average Precision
        
        // Fusion quality metrics
        float fusion_accuracy;
        float modality_contribution_balance;
        std::unordered_map<Modality, float> individual_modality_performance;
        
        // Representation quality metrics
        float representation_separability;
        float cross_modal_alignment;
        float intra_modal_clustering;
    };

    /**
     * @brief Evaluate multi-modal model performance
     */
    EvaluationMetrics evaluate(
        AdvancedMultiModalFusion& model,
        const std::vector<std::unordered_map<Modality, Tensor>>& test_data,
        const std::vector<Tensor>& ground_truth);

    /**
     * @brief Compute cross-modal retrieval metrics
     */
    float compute_recall_at_k(const std::unordered_map<Modality, Tensor>& queries,
                             const std::unordered_map<Modality, Tensor>& database,
                             int k = 5);

private:
    float compute_map_score(const std::vector<std::vector<float>>& similarities,
                           const std::vector<std::vector<bool>>& relevance);
    float compute_separability(const std::vector<Tensor>& representations,
                              const std::vector<int>& labels);
};

} // namespace ai
} // namespace asekioml
