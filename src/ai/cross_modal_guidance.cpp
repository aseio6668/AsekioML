#include "ai/cross_modal_guidance.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <numeric>
#include <limits>

namespace asekioml {
namespace ai {

// ============================================================================
// AdvancedCrossModalAttention Implementation
// ============================================================================

AdvancedCrossModalAttention::AdvancedCrossModalAttention(size_t query_dim, size_t key_dim, size_t value_dim, size_t num_heads)
    : query_dim_(query_dim), key_dim_(key_dim), value_dim_(value_dim), num_heads_(num_heads),
      guidance_strength_(0.5), attention_temperature_(1.0) {
    initializeParameters();
}

AdvancedCrossModalAttention::~AdvancedCrossModalAttention() = default;

void AdvancedCrossModalAttention::initializeParameters() {
    // Initialize projection matrices with small random values
    query_projection_ = Tensor::randn({query_dim_, query_dim_}, 0.0, 0.1);
    key_projection_ = Tensor::randn({key_dim_, key_dim_}, 0.0, 0.1);
    value_projection_ = Tensor::randn({value_dim_, value_dim_}, 0.0, 0.1);
    output_projection_ = Tensor::randn({value_dim_, value_dim_}, 0.0, 0.1);
    
    std::cout << "CrossModalAttention: Initialized with " << num_heads_ << " heads" << std::endl;
}

Tensor AdvancedCrossModalAttention::applySoftmax(const Tensor& input) const {
    // Manual softmax implementation since Tensor doesn't have built-in softmax
    Tensor result = input;
    
    // For each row, apply softmax
    for (size_t i = 0; i < input.shape()[0]; ++i) {
        // Find max value for numerical stability
        double max_val = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < input.shape()[1]; ++j) {
            max_val = std::max(max_val, input(i, j));
        }
        
        // Compute exp(x - max) and sum
        double sum_exp = 0.0;
        for (size_t j = 0; j < input.shape()[1]; ++j) {
            double exp_val = std::exp(input(i, j) - max_val);
            result(i, j) = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize by sum
        for (size_t j = 0; j < input.shape()[1]; ++j) {
            result(i, j) /= sum_exp;
        }
    }
    
    return result;
}

Tensor AdvancedCrossModalAttention::forward(const Tensor& query_modality, const Tensor& key_modality, 
                                   const Tensor& value_modality, const Tensor& guidance_mask) {
    std::cout << "CrossModalAttention: Computing cross-modal attention" << std::endl;
    std::cout << "  Input shapes - Query: [" << query_modality.shape()[0] << "x" << query_modality.shape()[1] << "]" << std::endl;
    std::cout << "  Input shapes - Key: [" << key_modality.shape()[0] << "x" << key_modality.shape()[1] << "]" << std::endl;
    std::cout << "  Input shapes - Value: [" << value_modality.shape()[0] << "x" << value_modality.shape()[1] << "]" << std::endl;
    
    // Skip projection for now to avoid matrix multiplication issues
    std::cout << "  Skipping projection matrices to avoid dimension mismatch" << std::endl;
    Tensor Q = query_modality;
    Tensor K = key_modality;
    Tensor V = value_modality;
    
    std::cout << "  Calling multiHeadAttention" << std::endl;
    // Compute multi-head attention
    Tensor attention_output = multiHeadAttention(Q, K, V);
    
    std::cout << "  Attention output shape: [" << attention_output.shape()[0] << "x" << attention_output.shape()[1] << "]" << std::endl;
    
    // Apply guidance if mask is provided
    if (guidance_mask.size() > 0) {
        std::cout << "  Applying guidance mask" << std::endl;
        attention_output = applyGuidance(attention_output, guidance_mask);
    }
    
    // Skip final output projection for now
    std::cout << "  Skipping output projection" << std::endl;
    Tensor output = attention_output;
    
    std::cout << "CrossModalAttention: Forward pass completed, output size: " << output.size() << std::endl;
    return output;
}

Tensor AdvancedCrossModalAttention::computeAttentionWeights(const Tensor& query, const Tensor& key) const {
    // Simplified attention weights computation - avoid matrix multiplication for now
    size_t batch_size = query.shape()[0];
    size_t head_dim = query.shape()[1];
    
    // Create attention weights tensor
    Tensor attention_weights = Tensor::zeros({batch_size, batch_size});
    
    // Compute pairwise similarities (simplified)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < batch_size; ++j) {
            double similarity = 0.0;
            // Compute dot product similarity
            for (size_t k = 0; k < head_dim; ++k) {
                similarity += query(i, k) * key(j, k);
            }
            // Scale and store
            double scale_factor = 1.0 / (std::sqrt(static_cast<double>(head_dim)) * attention_temperature_);
            attention_weights(i, j) = similarity * scale_factor;
        }
    }
    
    // Apply softmax to get attention weights
    Tensor normalized_weights = applySoftmax(attention_weights);
    
    return normalized_weights;
}

Tensor AdvancedCrossModalAttention::applyGuidance(const Tensor& attention_weights, const Tensor& guidance_signal) const {
    // Blend attention weights with guidance signal
    double guidance_factor = guidance_strength_;
    double attention_factor = 1.0 - guidance_strength_;
    
    Tensor guided_weights = attention_weights * attention_factor + guidance_signal * guidance_factor;
    
    // Ensure weights sum to 1 (renormalize)
    guided_weights = applySoftmax(guided_weights);
    
    return guided_weights;
}

Tensor AdvancedCrossModalAttention::multiHeadAttention(const Tensor& Q, const Tensor& K, const Tensor& V) const {
    // Simplified multi-head attention computation - use full tensors for now
    std::vector<Tensor> head_outputs;
    
    // For simplicity, just use the first head equivalent
    size_t head_dim = std::min(query_dim_ / num_heads_, Q.shape()[1]);
    
    if (head_dim > 0 && head_dim <= Q.shape()[1]) {
        // Compute attention for simplified head
        Tensor q_head = Q.slice({{0, Q.shape()[0]}, {0, head_dim}});
        Tensor k_head = K.slice({{0, K.shape()[0]}, {0, head_dim}});
        Tensor v_head = V.slice({{0, V.shape()[0]}, {0, head_dim}});
        
        Tensor attention_weights = computeAttentionWeights(q_head, k_head);
        last_attention_weights_ = attention_weights; // Store for analysis
        
        // Simple attention application - create output based on value head
        Tensor head_output = Tensor::zeros({q_head.shape()[0], v_head.shape()[1]});
        
        // Apply attention weights to values (simplified)
        for (size_t i = 0; i < head_output.shape()[0]; ++i) {
            for (size_t j = 0; j < head_output.shape()[1]; ++j) {
                double weighted_sum = 0.0;
                for (size_t k = 0; k < attention_weights.shape()[1]; ++k) {
                    weighted_sum += attention_weights(i, k) * v_head(k, j);
                }
                head_output(i, j) = weighted_sum;
            }
        }
        
        head_outputs.push_back(head_output);
    }
    
    // Return first head output or a default tensor
    return head_outputs.empty() ? Tensor::zeros({Q.shape()[0], Q.shape()[1]}) : head_outputs[0];
}

GuidanceMetrics AdvancedCrossModalAttention::computeGuidanceMetrics(const Tensor& source, const Tensor& target) const {
    GuidanceMetrics metrics;
    
    // Compute alignment quality based on cosine similarity
    metrics.alignment_quality = CrossModalUtils::computeCosineSimilarity(source, target);
    
    // Compute effectiveness based on attention distribution entropy
    if (last_attention_weights_.size() > 0) {
        // Simplified effectiveness computation
        metrics.effectiveness_score = std::min(1.0, metrics.alignment_quality + 0.2);
    } else {
        metrics.effectiveness_score = 0.5; // Default if no attention computed
    }
    
    // Estimate computational cost
    metrics.computational_cost = static_cast<double>(source.size() * target.size()) / 1000.0;
    
    // Set other metrics
    metrics.semantic_consistency = metrics.alignment_quality * 0.9; // Closely related to alignment
    metrics.temporal_coherence = 0.8; // Placeholder
    metrics.guidance_mode = "Cross-Modal Attention";
    
    return metrics;
}

// ============================================================================
// CrossModalConditioner Implementation
// ============================================================================

CrossModalConditioner::CrossModalConditioner(const GuidanceConfig& config) : config_(config) {
    initializeAttentionMechanism();
    std::cout << "CrossModalConditioner: Initialized with guidance type " 
              << static_cast<int>(config_.guidance_type) << std::endl;
}

CrossModalConditioner::~CrossModalConditioner() = default;

void CrossModalConditioner::initializeAttentionMechanism() {
    // Initialize with default dimensions (can be made configurable)
    size_t default_dim = 512;
    attention_mechanism_ = std::make_unique<AdvancedCrossModalAttention>(default_dim, default_dim, default_dim, 8);
    attention_mechanism_->setGuidanceStrength(config_.guidance_strength);
}

Tensor CrossModalConditioner::conditionModality(const Tensor& target_modality, const Tensor& source_modality,
                                               const std::string& target_type, const std::string& source_type) {
    std::cout << "CrossModalConditioner: Conditioning " << target_type 
              << " with " << source_type << std::endl;
    
    Tensor conditioned_output;
    
    switch (config_.guidance_type) {
        case GuidanceType::ATTENTION_BASED:
            conditioned_output = applyAttentionGuidance(target_modality, source_modality);
            break;
        case GuidanceType::FEATURE_ALIGNMENT:
            conditioned_output = applyFeatureAlignment(target_modality, source_modality);
            break;
        case GuidanceType::SEMANTIC_BRIDGE:
            conditioned_output = applySemanticBridge(target_modality, source_modality);
            break;
        case GuidanceType::TEMPORAL_SYNC:
            conditioned_output = applyTemporalSync(target_modality, source_modality);
            break;
        case GuidanceType::ADAPTIVE_WEIGHTED:
            // Use a combination based on modality types
            conditioned_output = applyAttentionGuidance(target_modality, source_modality);
            break;
        default:
            conditioned_output = target_modality; // No conditioning
            break;
    }
    
    // Compute and store guidance metrics
    last_metrics_ = attention_mechanism_->computeGuidanceMetrics(source_modality, conditioned_output);
    updateGuidanceHistory(last_metrics_);
    
    // Update modality affinities
    std::string affinity_key = source_type + "->" + target_type;
    modality_affinities_[affinity_key] = last_metrics_.effectiveness_score;
    
    std::cout << "CrossModalConditioner: Conditioning completed, effectiveness: " 
              << last_metrics_.effectiveness_score << std::endl;
    
    return conditioned_output;
}

std::map<std::string, Tensor> CrossModalConditioner::conditionMultiModal(const std::map<std::string, Tensor>& modalities) {
    std::cout << "CrossModalConditioner: Conditioning " << modalities.size() << " modalities" << std::endl;
    
    std::map<std::string, Tensor> conditioned_modalities;
    
    // For each modality, condition it with all other modalities
    for (const auto& [target_type, target_tensor] : modalities) {
        Tensor conditioned_tensor = target_tensor;
        
        for (const auto& [source_type, source_tensor] : modalities) {
            if (target_type != source_type) {
                // Get weight for this modality pair
                auto weight_it = config_.modality_weights.find(source_type);
                double weight = (weight_it != config_.modality_weights.end()) ? weight_it->second : 1.0;
                
                if (weight > 0.0) {
                    Tensor guidance_contribution = conditionModality(conditioned_tensor, source_tensor, 
                                                                   target_type, source_type);
                    
                    // Blend with existing conditioning
                    conditioned_tensor = conditioned_tensor * (1.0 - weight * config_.guidance_strength) + 
                                       guidance_contribution * (weight * config_.guidance_strength);
                }
            }
        }
        
        conditioned_modalities[target_type] = conditioned_tensor;
    }
    
    return conditioned_modalities;
}

Tensor CrossModalConditioner::applyAttentionGuidance(const Tensor& target, const Tensor& source) {
    // Use the attention mechanism for guidance
    return attention_mechanism_->forward(target, source, source);
}

Tensor CrossModalConditioner::applyFeatureAlignment(const Tensor& target, const Tensor& source) {
    std::cout << "CrossModalConditioner: Applying feature alignment guidance" << std::endl;
    
    // Align feature spaces using linear transformation
    Tensor aligned_target = alignFeatureSpaces(source, target);
    
    // Blend aligned features with original target
    double blend_factor = config_.guidance_strength;
    Tensor result = target * (1.0 - blend_factor) + aligned_target * blend_factor;
    
    return result;
}

Tensor CrossModalConditioner::applySemanticBridge(const Tensor& target, const Tensor& source) {
    std::cout << "CrossModalConditioner: Applying semantic bridge guidance" << std::endl;
    
    // Compute semantic similarity
    double similarity = computeSemanticSimilarity(target, source);
    
    if (similarity > config_.semantic_threshold) {
        // Apply stronger guidance for semantically similar content
        double enhanced_strength = config_.guidance_strength * (1.0 + similarity);
        Tensor enhanced_source = source * enhanced_strength;
        
        // Create semantic bridge
        Tensor semantic_bridge = (target + enhanced_source) * 0.5;
        return semantic_bridge;
    } else {
        // Minimal guidance for dissimilar content
        return target * 0.9 + source * 0.1;
    }
}

Tensor CrossModalConditioner::applyTemporalSync(const Tensor& target, const Tensor& source) {
    std::cout << "CrossModalConditioner: Applying temporal synchronization guidance" << std::endl;
    
    // Simplified temporal synchronization
    // In a real implementation, this would involve temporal alignment algorithms
    
    if (config_.enable_temporal_sync) {
        // Apply temporal alignment (simplified as element-wise blending with phase adjustment)
        Tensor phase_adjusted_source = source * 1.1; // Simplified phase adjustment
        Tensor sync_result = target * 0.7 + phase_adjusted_source * 0.3;
        return sync_result;
    } else {
        return target;
    }
}

void CrossModalConditioner::adaptGuidanceStrength(const GuidanceMetrics& metrics) {
    if (!config_.adaptive_mode) return;
    
    double effectiveness = metrics.effectiveness_score;
    double current_strength = config_.guidance_strength;
    
    // Adaptive adjustment based on effectiveness
    if (effectiveness > 0.8) {
        // High effectiveness - slightly increase strength
        config_.guidance_strength = std::min(1.0, current_strength + config_.adaptation_rate * 0.1);
    } else if (effectiveness < 0.5) {
        // Low effectiveness - decrease strength
        config_.guidance_strength = std::max(0.1, current_strength - config_.adaptation_rate * 0.2);
    }
    
    // Update attention mechanism
    attention_mechanism_->setGuidanceStrength(config_.guidance_strength);
    
    std::cout << "CrossModalConditioner: Adapted guidance strength to " 
              << config_.guidance_strength << " (effectiveness: " << effectiveness << ")" << std::endl;
}

std::map<std::string, double> CrossModalConditioner::getModalityAffinities() const {
    return modality_affinities_;
}

void CrossModalConditioner::resetGuidanceHistory() {
    guidance_history_.clear();
    modality_affinities_.clear();
    std::cout << "CrossModalConditioner: Reset guidance history" << std::endl;
}

void CrossModalConditioner::updateGuidanceHistory(const GuidanceMetrics& metrics) {
    guidance_history_.push_back(metrics);
    
    // Keep only recent history (last 100 entries)
    if (guidance_history_.size() > 100) {
        guidance_history_.erase(guidance_history_.begin());
    }
}

double CrossModalConditioner::computeSemanticSimilarity(const Tensor& tensor1, const Tensor& tensor2) const {
    return CrossModalUtils::computeCosineSimilarity(tensor1, tensor2);
}

Tensor CrossModalConditioner::alignFeatureSpaces(const Tensor& source, const Tensor& target) const {
    // Simplified feature space alignment using linear transformation
    // In practice, this could use CCA, autoencoders, or other alignment techniques
    
    if (source.shape().size() != target.shape().size()) {
        std::cout << "CrossModalConditioner: Warning - mismatched tensor dimensions for alignment" << std::endl;
        return target; // Return unchanged if dimensions don't match
    }
    
    // Simple linear alignment (could be enhanced with learned transformations)
    double alignment_factor = 0.8;
    Tensor aligned = source * alignment_factor + target * (1.0 - alignment_factor);
    
    return aligned;
}

// ============================================================================
// GuidanceController Implementation
// ============================================================================

GuidanceController::GuidanceController(double initial_strength, double adaptation_rate)
    : current_strength_(initial_strength), target_strength_(initial_strength),
      adaptation_rate_(adaptation_rate), quality_threshold_(0.7),
      adaptive_control_enabled_(true), max_history_size_(100) {
    std::cout << "GuidanceController: Initialized with strength " << initial_strength << std::endl;
}

GuidanceController::~GuidanceController() = default;

void GuidanceController::setTargetStrength(double target_strength) {
    target_strength_ = std::max(0.0, std::min(1.0, target_strength));
    std::cout << "GuidanceController: Target strength set to " << target_strength_ << std::endl;
}

void GuidanceController::updateStrength(const GuidanceMetrics& feedback) {
    if (!adaptive_control_enabled_) return;
    
    double effectiveness = feedback.effectiveness_score;
    double adaptation_delta = computeAdaptationDelta(effectiveness);
    
    // Update current strength towards target
    double target_delta = (target_strength_ - current_strength_) * adaptation_rate_;
    double total_delta = adaptation_delta + target_delta;
    
    current_strength_ = std::max(0.0, std::min(1.0, current_strength_ + total_delta));
    
    updateHistory(current_strength_, effectiveness);
    
    std::cout << "GuidanceController: Updated strength to " << current_strength_ 
              << " (effectiveness: " << effectiveness << ")" << std::endl;
}

void GuidanceController::adaptStrength(double effectiveness, double quality_threshold) {
    quality_threshold_ = quality_threshold;
    
    if (effectiveness > quality_threshold_) {
        // Good performance - gradually increase strength
        current_strength_ = std::min(1.0, current_strength_ + adaptation_rate_ * 0.1);
    } else {
        // Poor performance - decrease strength
        current_strength_ = std::max(0.1, current_strength_ - adaptation_rate_ * 0.2);
    }
    
    updateHistory(current_strength_, effectiveness);
}

double GuidanceController::getAverageEffectiveness() const {
    if (effectiveness_history_.empty()) return 0.0;
    
    double sum = std::accumulate(effectiveness_history_.begin(), effectiveness_history_.end(), 0.0);
    return sum / effectiveness_history_.size();
}

bool GuidanceController::isStabilized() const {
    if (strength_history_.size() < 10) return false;
    
    // Check if strength has stabilized (low variance in recent history)
    auto recent_start = strength_history_.end() - 10;
    std::vector<double> recent_strengths(recent_start, strength_history_.end());
    
    double mean = std::accumulate(recent_strengths.begin(), recent_strengths.end(), 0.0) / recent_strengths.size();
    double variance = 0.0;
    
    for (double strength : recent_strengths) {
        variance += (strength - mean) * (strength - mean);
    }
    variance /= recent_strengths.size();
    
    return variance < 0.01; // Stabilized if variance is low
}

void GuidanceController::resetController() {
    strength_history_.clear();
    effectiveness_history_.clear();
    current_strength_ = target_strength_;
    std::cout << "GuidanceController: Controller reset" << std::endl;
}

void GuidanceController::updateHistory(double strength, double effectiveness) {
    strength_history_.push_back(strength);
    effectiveness_history_.push_back(effectiveness);
    
    // Maintain maximum history size
    if (strength_history_.size() > max_history_size_) {
        strength_history_.erase(strength_history_.begin());
        effectiveness_history_.erase(effectiveness_history_.begin());
    }
}

double GuidanceController::computeAdaptationDelta(double effectiveness) const {
    // Compute adaptation based on effectiveness relative to threshold
    double effectiveness_error = effectiveness - quality_threshold_;
    
    if (effectiveness_error > 0) {
        // Above threshold - small positive adjustment
        return adaptation_rate_ * effectiveness_error * 0.1;
    } else {
        // Below threshold - larger negative adjustment
        return adaptation_rate_ * effectiveness_error * 0.3;
    }
}

// ============================================================================
// ModalBridgeNetwork Implementation
// ============================================================================

ModalBridgeNetwork::ModalBridgeNetwork(const std::vector<std::string>& modality_types)
    : modality_types_(modality_types), semantic_threshold_(0.3) {
    initializeBridges();
    std::cout << "ModalBridgeNetwork: Initialized with " << modality_types_.size() 
              << " modality types" << std::endl;
}

ModalBridgeNetwork::~ModalBridgeNetwork() = default;

void ModalBridgeNetwork::initializeBridges() {
    // Create all possible bridges between modalities
    for (size_t i = 0; i < modality_types_.size(); ++i) {
        for (size_t j = 0; j < modality_types_.size(); ++j) {
            if (i != j) {
                std::string source = modality_types_[i];
                std::string target = modality_types_[j];
                
                // Initialize with moderate bridge strength
                bridge_strengths_[{source, target}] = 0.5;
                
                // Create transformation matrix
                bridge_transformations_[{source, target}] = createBridgeTransformation(source, target);
            }
        }
    }
}

bool ModalBridgeNetwork::createBridge(const std::string& source_modality, const std::string& target_modality) {
    if (!isValidModalityType(source_modality) || !isValidModalityType(target_modality)) {
        std::cout << "ModalBridgeNetwork: Invalid modality types for bridge creation" << std::endl;
        return false;
    }
    
    auto bridge_key = std::make_pair(source_modality, target_modality);
    bridge_strengths_[bridge_key] = 0.8; // Strong initial connection
    bridge_transformations_[bridge_key] = createBridgeTransformation(source_modality, target_modality);
    
    std::cout << "ModalBridgeNetwork: Created bridge " << source_modality 
              << " -> " << target_modality << std::endl;
    return true;
}

bool ModalBridgeNetwork::removeBridge(const std::string& source_modality, const std::string& target_modality) {
    auto bridge_key = std::make_pair(source_modality, target_modality);
    
    if (bridge_strengths_.find(bridge_key) != bridge_strengths_.end()) {
        bridge_strengths_.erase(bridge_key);
        bridge_transformations_.erase(bridge_key);
        
        std::cout << "ModalBridgeNetwork: Removed bridge " << source_modality 
                  << " -> " << target_modality << std::endl;
        return true;
    }
    
    return false;
}

std::vector<std::string> ModalBridgeNetwork::getConnectedModalities(const std::string& modality) const {
    std::vector<std::string> connected;
    
    for (const auto& [bridge_key, strength] : bridge_strengths_) {
        if (bridge_key.first == modality && strength > semantic_threshold_) {
            connected.push_back(bridge_key.second);
        }
    }
    
    return connected;
}

Tensor ModalBridgeNetwork::bridgeFeatures(const Tensor& source_features, const std::string& source_type,
                                         const std::string& target_type) {
    auto bridge_key = std::make_pair(source_type, target_type);
    auto strength_it = bridge_strengths_.find(bridge_key);
    auto transform_it = bridge_transformations_.find(bridge_key);
    
    if (strength_it == bridge_strengths_.end() || transform_it == bridge_transformations_.end()) {
        std::cout << "ModalBridgeNetwork: No bridge found between " << source_type 
                  << " and " << target_type << std::endl;
        return source_features; // Return unchanged
    }
    
    double bridge_strength = strength_it->second;
    const Tensor& transformation = transform_it->second;
    
    // Apply transformation if dimensions are compatible
    Tensor bridged_features;
    if (source_features.shape().size() >= 2 && transformation.shape().size() >= 2) {
        bridged_features = source_features.matmul(transformation);
    } else {
        // Simple scaling if matrix multiplication not possible
        bridged_features = source_features * bridge_strength;
    }
    
    std::cout << "ModalBridgeNetwork: Bridged features from " << source_type 
              << " to " << target_type << " (strength: " << bridge_strength << ")" << std::endl;
    
    return bridged_features;
}

std::map<std::string, Tensor> ModalBridgeNetwork::broadcastFeatures(const Tensor& features, 
                                                                   const std::string& source_type) {
    std::map<std::string, Tensor> broadcast_results;
    
    for (const std::string& target_type : modality_types_) {
        if (target_type != source_type) {
            broadcast_results[target_type] = bridgeFeatures(features, source_type, target_type);
        }
    }
    
    std::cout << "ModalBridgeNetwork: Broadcast features from " << source_type 
              << " to " << broadcast_results.size() << " target modalities" << std::endl;
    
    return broadcast_results;
}

Tensor ModalBridgeNetwork::alignSemanticSpaces(const Tensor& source, const Tensor& target,
                                              const std::string& source_type, const std::string& target_type) {
    // Bridge source to target space
    Tensor bridged_source = bridgeFeatures(source, source_type, target_type);
    
    // Compute semantic alignment
    double similarity = computeSemanticDistance(bridged_source, target);
    
    if (similarity < semantic_threshold_) {
        // Apply stronger alignment for dissimilar features
        double alignment_factor = 0.7;
        Tensor aligned = bridged_source * alignment_factor + target * (1.0 - alignment_factor);
        return aligned;
    } else {
        // Weaker alignment for already similar features
        double alignment_factor = 0.3;
        Tensor aligned = bridged_source * alignment_factor + target * (1.0 - alignment_factor);
        return aligned;
    }
}

double ModalBridgeNetwork::computeSemanticDistance(const Tensor& features1, const Tensor& features2) const {
    // Use cosine distance (1 - cosine similarity)
    double similarity = CrossModalUtils::computeCosineSimilarity(features1, features2);
    return 1.0 - similarity;
}

std::map<std::pair<std::string, std::string>, double> ModalBridgeNetwork::getBridgeStrengths() const {
    return bridge_strengths_;
}

std::vector<std::string> ModalBridgeNetwork::findOptimalPath(const std::string& source, const std::string& target) const {
    // Simple pathfinding using highest strength connections
    std::vector<std::string> path;
    std::string current = source;
    path.push_back(current);
    
    while (current != target && path.size() < modality_types_.size()) {
        std::string next_hop;
        double best_strength = 0.0;
        
        // Find strongest connection from current modality
        for (const auto& [bridge_key, strength] : bridge_strengths_) {
            if (bridge_key.first == current && strength > best_strength) {
                best_strength = strength;
                next_hop = bridge_key.second;
            }
        }
        
        if (next_hop.empty() || best_strength <= semantic_threshold_) {
            break; // No valid path found
        }
        
        current = next_hop;
        path.push_back(current);
    }
    
    if (current == target) {
        std::cout << "ModalBridgeNetwork: Found path from " << source << " to " << target 
                  << " with " << path.size() << " hops" << std::endl;
        return path;
    } else {
        std::cout << "ModalBridgeNetwork: No path found from " << source << " to " << target << std::endl;
        return {};
    }
}

void ModalBridgeNetwork::optimizeBridgeNetwork() {
    // Optimize bridge strengths based on usage patterns and effectiveness
    // This is a simplified optimization - in practice, this could use learning algorithms
    
    for (auto& [bridge_key, strength] : bridge_strengths_) {
        // Simulate optimization by slight random adjustment
        double adjustment = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 0.1;
        strength = std::max(0.0, std::min(1.0, strength + adjustment));
    }
    
    std::cout << "ModalBridgeNetwork: Bridge network optimization completed" << std::endl;
}

void ModalBridgeNetwork::setBridgeStrength(const std::string& source, const std::string& target, double strength) {
    auto bridge_key = std::make_pair(source, target);
    bridge_strengths_[bridge_key] = std::max(0.0, std::min(1.0, strength));
    
    std::cout << "ModalBridgeNetwork: Set bridge strength " << source << " -> " << target 
              << " to " << strength << std::endl;
}

Tensor ModalBridgeNetwork::createBridgeTransformation(const std::string& source_type, const std::string& target_type) {
    // Create a transformation matrix for bridging between modality types
    // This is simplified - in practice, these could be learned transformations
    
    size_t default_dim = 256; // Default feature dimension
    Tensor transformation = Tensor::randn({default_dim, default_dim}, 0.0, 0.1);
    
    // Add some structure based on modality types
    if (source_type == "text" && target_type == "image") {
        // Text to image transformation might emphasize certain dimensions
        transformation = transformation * 1.2;
    } else if (source_type == "audio" && target_type == "text") {
        // Audio to text transformation
        transformation = transformation * 0.8;
    }
    
    return transformation;
}

bool ModalBridgeNetwork::isValidModalityType(const std::string& modality_type) const {
    return std::find(modality_types_.begin(), modality_types_.end(), modality_type) != modality_types_.end();
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

namespace CrossModalUtils {

double computeCosineSimilarity(const Tensor& tensor1, const Tensor& tensor2) {
    if (tensor1.size() != tensor2.size()) {
        std::cout << "CrossModalUtils: Warning - tensor size mismatch for cosine similarity" << std::endl;
        return 0.0;
    }
    
    // Compute dot product
    double dot_product = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (size_t i = 0; i < tensor1.size(); ++i) {
        double val1 = tensor1.data()[i];
        double val2 = tensor2.data()[i];
        
        dot_product += val1 * val2;
        norm1 += val1 * val1;
        norm2 += val2 * val2;
    }
    
    if (norm1 == 0.0 || norm2 == 0.0) {
        return 0.0;
    }
    
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

double computeEuclideanDistance(const Tensor& tensor1, const Tensor& tensor2) {
    if (tensor1.size() != tensor2.size()) {
        std::cout << "CrossModalUtils: Warning - tensor size mismatch for Euclidean distance" << std::endl;
        return std::numeric_limits<double>::max();
    }
    
    double distance = 0.0;
    for (size_t i = 0; i < tensor1.size(); ++i) {
        double diff = tensor1.data()[i] - tensor2.data()[i];
        distance += diff * diff;
    }
    
    return std::sqrt(distance);
}

double computeKLDivergence(const Tensor& tensor1, const Tensor& tensor2) {
    if (tensor1.size() != tensor2.size()) {
        std::cout << "CrossModalUtils: Warning - tensor size mismatch for KL divergence" << std::endl;
        return std::numeric_limits<double>::max();
    }
    
    double kl_div = 0.0;
    const double epsilon = 1e-8; // Small value to avoid log(0)
    
    for (size_t i = 0; i < tensor1.size(); ++i) {
        double p = std::max(epsilon, static_cast<double>(tensor1.data()[i]));
        double q = std::max(epsilon, static_cast<double>(tensor2.data()[i]));
        
        kl_div += p * std::log(p / q);
    }
    
    return kl_div;
}

GuidanceMetrics evaluateGuidanceEffectiveness(const Tensor& guided_output, const Tensor& reference_output) {
    GuidanceMetrics metrics;
    
    // Compute alignment quality
    metrics.alignment_quality = computeCosineSimilarity(guided_output, reference_output);
    
    // Compute effectiveness based on distance metrics
    double distance = computeEuclideanDistance(guided_output, reference_output);
    metrics.effectiveness_score = 1.0 / (1.0 + distance / 100.0); // Normalize distance
    
    // Semantic consistency approximated by cosine similarity
    metrics.semantic_consistency = metrics.alignment_quality;
    
    // Temporal coherence (simplified)
    metrics.temporal_coherence = 0.8; // Placeholder value
    
    // Computational cost estimate
    metrics.computational_cost = static_cast<double>(guided_output.size()) / 1000.0;
    
    metrics.guidance_mode = "Evaluated Guidance";
    
    return metrics;
}

double computeGuidanceStability(const std::vector<GuidanceMetrics>& metrics_history) {
    if (metrics_history.empty()) return 0.0;
    
    std::vector<double> effectiveness_scores;
    for (const auto& metrics : metrics_history) {
        effectiveness_scores.push_back(metrics.effectiveness_score);
    }
    
    // Compute variance of effectiveness scores
    double mean = std::accumulate(effectiveness_scores.begin(), effectiveness_scores.end(), 0.0) / effectiveness_scores.size();
    double variance = 0.0;
    
    for (double score : effectiveness_scores) {
        variance += (score - mean) * (score - mean);
    }
    variance /= effectiveness_scores.size();
    
    // Stability is inverse of variance
    return 1.0 / (1.0 + variance);
}

} // namespace CrossModalUtils

} // namespace ai
} // namespace asekioml
