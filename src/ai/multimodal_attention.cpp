#include "ai/multimodal_attention.hpp"
#include "ai/memory_manager.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace asekioml {
namespace ai {

// MultiModalTensor Implementation
MultiModalTensor::MultiModalTensor(const Tensor& data, Modality modality, const std::string& description)
    : data_(data), modality_(modality), description_(description) {
}

std::vector<size_t> MultiModalTensor::get_sequence_shape() const {
    const auto& shape = data_.shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Sequence tensor must have at least 3 dimensions [batch, seq_len, features]");
    }
    return {shape[0], shape[1], shape[2]}; // [batch, seq_len, features]
}

std::vector<size_t> MultiModalTensor::get_spatial_shape() const {
    const auto& shape = data_.shape();
    if (shape.size() < 4) {
        throw std::invalid_argument("Spatial tensor must have 4 dimensions [batch, height, width, channels]");
    }
    return {shape[0], shape[1], shape[2], shape[3]}; // [batch, height, width, channels]
}

std::vector<size_t> MultiModalTensor::get_temporal_shape() const {
    const auto& shape = data_.shape();
    if (shape.size() < 3) {
        throw std::invalid_argument("Temporal tensor must have at least 3 dimensions [batch, time, features]");
    }
    return {shape[0], shape[1], shape[2]}; // [batch, time, features]
}

// CrossModalAttention Implementation
CrossModalAttention::CrossModalAttention(size_t source_dim, size_t target_dim, size_t hidden_dim, size_t num_heads)
    : source_dim_(source_dim), target_dim_(target_dim), hidden_dim_(hidden_dim), num_heads_(num_heads) {
    
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument("Hidden dimension must be divisible by number of heads");
    }
    
    // Initialize core attention layer
    attention_layer_ = std::make_unique<MultiHeadAttentionLayer>(hidden_dim, num_heads);
    
    // Initialize projection matrices for different modality pairs
    initialize_projection_matrices();
}

Tensor CrossModalAttention::forward(const Tensor& source, const Tensor& target, 
                                   Modality source_modality, Modality target_modality) {
    // Project source and target to common hidden dimension
    Tensor projected_source = project_modality_features(source, source_modality, target_modality);
    Tensor projected_target = project_modality_features(target, target_modality, source_modality);
    
    // Apply cross-attention: target attends to source
    Tensor attended_output = attention_layer_->forward_tensor(projected_target, projected_source, projected_source);
    
    // Store attention weights for visualization
    last_attention_weights_ = attention_layer_->get_last_attention_weights();
    
    return attended_output;
}

std::pair<Tensor, Tensor> CrossModalAttention::bidirectional_attention(const Tensor& modal1, const Tensor& modal2,
                                                                       Modality modality1, Modality modality2) {
    // Forward: modal1 attends to modal2
    Tensor attended_modal1 = forward(modal2, modal1, modality2, modality1);
    
    // Backward: modal2 attends to modal1  
    Tensor attended_modal2 = forward(modal1, modal2, modality1, modality2);
    
    return std::make_pair(attended_modal1, attended_modal2);
}

Tensor CrossModalAttention::get_attention_weights() const {
    return last_attention_weights_;
}

std::string CrossModalAttention::get_modality_pair_key(Modality source, Modality target) const {
    auto modality_to_string = [](Modality m) -> const char* {
        switch (m) {
            case Modality::TEXT: return "text";
            case Modality::IMAGE: return "image";
            case Modality::AUDIO: return "audio";
            case Modality::VIDEO: return "video";
            default: return "unknown";
        }
    };
    
    return std::string(modality_to_string(source)) + "_to_" + std::string(modality_to_string(target));
}

Tensor CrossModalAttention::project_modality_features(const Tensor& features, Modality source, Modality target) {
    std::string key = get_modality_pair_key(source, target);
    
    auto it = projection_matrices_.find(key);
    if (it == projection_matrices_.end()) {
        throw std::runtime_error("Projection matrix not found for modality pair: " + key);
    }
    
    // Linear projection: features @ projection_matrix
    return features.matmul(it->second);
}

void CrossModalAttention::initialize_projection_matrices() {
    // Initialize projection matrices for all modality pairs
    std::vector<Modality> modalities = {Modality::TEXT, Modality::IMAGE, Modality::AUDIO, Modality::VIDEO};
    
    for (auto source : modalities) {
        for (auto target : modalities) {
            std::string key = get_modality_pair_key(source, target);
            
            // Determine input dimension based on source modality
            size_t input_dim = (source == Modality::TEXT || source == Modality::AUDIO) ? source_dim_ : target_dim_;
            
            // Xavier initialization for projection matrix
            projection_matrices_[key] = Tensor({input_dim, hidden_dim_});
            float scale = std::sqrt(2.0f / (input_dim + hidden_dim_));
            
            // Initialize with small random values
            auto& data = projection_matrices_[key].data();
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
            }
        }
    }
}

// MultiModalFusion Implementation
MultiModalFusion::MultiModalFusion(FusionStrategy strategy, 
                                  const std::unordered_map<Modality, size_t>& modality_dims,
                                  size_t output_dim)
    : strategy_(strategy), modality_dims_(modality_dims), output_dim_(output_dim) {
    
    initialize_fusion_components();
}

Tensor MultiModalFusion::fuse(const std::unordered_map<Modality, Tensor>& modality_features) {
    switch (strategy_) {
        case FusionStrategy::EARLY_FUSION:
            return early_fusion(modality_features);
        case FusionStrategy::LATE_FUSION:
            return late_fusion(modality_features);
        case FusionStrategy::INTERMEDIATE_FUSION:
            return intermediate_fusion(modality_features);
        case FusionStrategy::ATTENTION_FUSION:
            return attention_fusion(modality_features);
        default:
            throw std::invalid_argument("Unknown fusion strategy");
    }
}

std::pair<Tensor, Tensor> MultiModalFusion::fuse_with_attention(
    const std::unordered_map<Modality, Tensor>& modality_features,
    const std::unordered_map<Modality, Tensor>& attention_weights) {
    
    Tensor fused_features = fuse(modality_features);
    
    // Compute attention scores if not provided
    Tensor attention_scores;
    if (attention_weights.empty()) {
        // Simple uniform attention
        attention_scores = Tensor({modality_features.size()});
        float uniform_weight = 1.0f / static_cast<float>(modality_features.size());
        std::fill(attention_scores.data().begin(), attention_scores.data().end(), uniform_weight);
    } else {
        // Use provided attention weights
        std::vector<float> scores;
        for (const auto& pair : modality_features) {
            auto it = attention_weights.find(pair.first);
            if (it != attention_weights.end()) {
                scores.push_back(it->second.data()[0]); // Assume scalar attention weight
            } else {
                scores.push_back(1.0f / static_cast<float>(modality_features.size()));
            }
        }
        attention_scores = Tensor({scores.size()});
        std::copy(scores.begin(), scores.end(), attention_scores.data().begin());
    }
    
    return std::make_pair(fused_features, attention_scores);
}

Tensor MultiModalFusion::early_fusion(const std::unordered_map<Modality, Tensor>& features) {
    // Concatenate all modality features along the feature dimension
    std::vector<Tensor> feature_list;
    for (const auto& pair : features) {
        feature_list.push_back(pair.second);
    }
    
    if (feature_list.empty()) {
        throw std::invalid_argument("No features to fuse");
    }
    
    // Simple concatenation implementation
    const auto& first_shape = feature_list[0].shape();
    size_t batch_size = first_shape[0];
    size_t seq_len = first_shape.size() > 1 ? first_shape[1] : 1;
    
    size_t total_features = 0;
    for (const auto& tensor : feature_list) {
        total_features += tensor.shape().back(); // Last dimension is feature dimension
    }
    
    Tensor concatenated({batch_size, seq_len, total_features});
    
    // Copy features (simplified implementation)
    size_t offset = 0;
    for (const auto& tensor : feature_list) {
        size_t feature_dim = tensor.shape().back();
        // Copy logic would go here - simplified for now
        offset += feature_dim;
    }
    
    // Apply final projection to output dimension
    if (fusion_weights_.shape().empty()) {
        fusion_weights_ = Tensor({total_features, output_dim_});
        // Initialize fusion weights
        float scale = std::sqrt(2.0f / (total_features + output_dim_));
        for (auto& val : fusion_weights_.data()) {
            val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }
    
    return concatenated.matmul(fusion_weights_);
}

Tensor MultiModalFusion::late_fusion(const std::unordered_map<Modality, Tensor>& features) {
    // Process each modality separately, then combine outputs
    std::vector<Tensor> processed_features;
    
    for (const auto& pair : features) {
        Modality modality = pair.first;
        const Tensor& feature = pair.second;
        
        // Apply modality-specific projection
        auto it = projection_weights_.find(modality);
        if (it != projection_weights_.end()) {
            processed_features.push_back(feature.matmul(it->second));
        } else {
            processed_features.push_back(feature);
        }
    }
    
    // Simple averaging for late fusion
    if (processed_features.empty()) {
        throw std::invalid_argument("No features to fuse");
    }
    
    Tensor result = processed_features[0];
    for (size_t i = 1; i < processed_features.size(); ++i) {
        const auto& feature = processed_features[i];
        const auto& result_data = result.data();
        const auto& feature_data = feature.data();
        
        auto& output_data = result.data();
        for (size_t j = 0; j < result_data.size() && j < feature_data.size(); ++j) {
            output_data[j] = result_data[j] + feature_data[j];
        }
    }
    
    float scale = 1.0f / processed_features.size();
    for (auto& val : result.data()) {
        val *= scale;
    }
    
    return result;
}

Tensor MultiModalFusion::intermediate_fusion(const std::unordered_map<Modality, Tensor>& features) {
    // Combine early and late fusion strategies
    // First apply early fusion to related modalities, then late fusion to combine groups
    
    std::vector<Tensor> intermediate_results;
    
    // Group related modalities for early fusion
    std::vector<Tensor> visual_features;
    std::vector<Tensor> text_features;
    std::vector<Tensor> audio_features;
    
    for (const auto& pair : features) {
        switch (pair.first) {
            case Modality::IMAGE:
            case Modality::VIDEO:
                visual_features.push_back(pair.second);
                break;
            case Modality::TEXT:
                text_features.push_back(pair.second);
                break;
            case Modality::AUDIO:
                audio_features.push_back(pair.second);
                break;
        }
    }
    
    // Process each group separately
    if (!visual_features.empty()) {
        // Apply early fusion to visual features
        std::unordered_map<Modality, Tensor> visual_map;
        visual_map[Modality::IMAGE] = visual_features[0];
        intermediate_results.push_back(early_fusion(visual_map));
    }
    
    if (!text_features.empty()) {
        std::unordered_map<Modality, Tensor> text_map;
        text_map[Modality::TEXT] = text_features[0];
        intermediate_results.push_back(early_fusion(text_map));
    }
    
    if (!audio_features.empty()) {
        std::unordered_map<Modality, Tensor> audio_map;
        audio_map[Modality::AUDIO] = audio_features[0];
        intermediate_results.push_back(early_fusion(audio_map));
    }
    
    // Apply late fusion to intermediate results
    if (intermediate_results.empty()) {
        throw std::invalid_argument("No intermediate features to fuse");
    }
    
    Tensor result = intermediate_results[0];
    for (size_t i = 1; i < intermediate_results.size(); ++i) {
        const auto& feature = intermediate_results[i];
        const auto& result_data = result.data();
        const auto& feature_data = feature.data();
        
        auto& output_data = result.data();
        for (size_t j = 0; j < result_data.size() && j < feature_data.size(); ++j) {
            output_data[j] = result_data[j] + feature_data[j];
        }
    }
    
    float scale = 1.0f / intermediate_results.size();
    for (auto& val : result.data()) {
        val *= scale;
    }
    
    return result;
}

Tensor MultiModalFusion::attention_fusion(const std::unordered_map<Modality, Tensor>& features) {
    // Use cross-modal attention to selectively combine features
    if (!fusion_attention_) {
        throw std::runtime_error("Fusion attention not initialized");
    }
    
    std::vector<Tensor> attended_features;
    std::vector<Modality> modality_order;
    
    // Apply cross-modal attention between all pairs
    for (const auto& source_pair : features) {
        Tensor combined_attention = source_pair.second;
        
        for (const auto& target_pair : features) {
            if (source_pair.first != target_pair.first) {
                Tensor attended = fusion_attention_->forward(
                    target_pair.second, source_pair.second,
                    target_pair.first, source_pair.first
                );
                
                // Combine with existing attention
                const auto& combined_data = combined_attention.data();
                const auto& attended_data = attended.data();
                
                auto& output_data = combined_attention.data();
                for (size_t i = 0; i < combined_data.size() && i < attended_data.size(); ++i) {
                    output_data[i] = combined_data[i] + attended_data[i];
                }
            }
        }
        
        attended_features.push_back(combined_attention);
        modality_order.push_back(source_pair.first);
    }
    
    // Combine attended features
    if (attended_features.empty()) {
        throw std::invalid_argument("No attended features to combine");
    }
    
    Tensor result = attended_features[0];
    for (size_t i = 1; i < attended_features.size(); ++i) {
        const auto& feature = attended_features[i];
        const auto& result_data = result.data();
        const auto& feature_data = feature.data();
        
        auto& output_data = result.data();
        for (size_t j = 0; j < result_data.size() && j < feature_data.size(); ++j) {
            output_data[j] = result_data[j] + feature_data[j];
        }
    }
    
    float scale = 1.0f / attended_features.size();
    for (auto& val : result.data()) {
        val *= scale;
    }
    
    return result;
}

void MultiModalFusion::initialize_fusion_components() {
    // Initialize fusion attention for attention-based fusion
    if (strategy_ == FusionStrategy::ATTENTION_FUSION) {
        size_t max_dim = 0;
        for (const auto& pair : modality_dims_) {
            max_dim = std::max(max_dim, pair.second);
        }
        fusion_attention_ = std::make_unique<CrossModalAttention>(max_dim, max_dim, output_dim_);
    }
    
    // Initialize projection weights for late fusion
    if (strategy_ == FusionStrategy::LATE_FUSION) {
        for (const auto& pair : modality_dims_) {
            Modality modality = pair.first;
            size_t input_dim = pair.second;
            
            projection_weights_[modality] = Tensor({input_dim, output_dim_});
            
            // Xavier initialization
            float scale = std::sqrt(2.0f / (input_dim + output_dim_));
            auto& data = projection_weights_[modality].data();
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
            }
        }
    }
}

// MultiModalTransformer Implementation
MultiModalTransformer::MultiModalTransformer(const std::unordered_map<Modality, size_t>& modality_configs,
                                           size_t hidden_dim, size_t num_layers, size_t num_heads)
    : modality_configs_(modality_configs), hidden_dim_(hidden_dim), num_layers_(num_layers), num_heads_(num_heads) {
    initialize_layers();
}

std::unordered_map<Modality, Tensor> MultiModalTransformer::forward(
    const std::unordered_map<Modality, Tensor>& inputs) {
    
    std::unordered_map<Modality, Tensor> outputs;
    
    // Project inputs to common hidden dimension
    std::unordered_map<Modality, Tensor> projected_inputs;
    for (const auto& pair : inputs) {
        Modality modality = pair.first;
        const Tensor& input = pair.second;
        
        auto it = input_projections_.find(modality);
        if (it != input_projections_.end()) {
            projected_inputs[modality] = input.matmul(it->second);
        } else {
            projected_inputs[modality] = input;
        }
    }
    
    // Apply transformer layers
    std::unordered_map<Modality, Tensor> current_states = projected_inputs;
    
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        std::unordered_map<Modality, Tensor> layer_outputs;
        
        // Self-attention within each modality
        for (const auto& pair : current_states) {
            Modality modality = pair.first;
            const Tensor& state = pair.second;
            
            auto it = self_attention_layers_.find(modality);
            if (it != self_attention_layers_.end() && layer < it->second.size()) {
                Tensor attended = it->second[layer]->forward_tensor(state, state, state);
                
                // Apply layer normalization if available
                auto norm_it = layer_norms_.find(modality);
                if (norm_it != layer_norms_.end() && layer < norm_it->second.size()) {
                    attended = apply_layer_norm(attended, norm_it->second[layer]);
                }
                
                layer_outputs[modality] = attended;
            } else {
                layer_outputs[modality] = state;
            }
        }
        
        // Cross-modal attention if cross-attention layers are available
        if (layer < cross_attention_layers_.size()) {
            std::unordered_map<Modality, Tensor> cross_attended;
            
            for (const auto& target_pair : layer_outputs) {
                Modality target_modality = target_pair.first;
                const Tensor& target_state = target_pair.second;
                
                Tensor combined_cross_attention = target_state;
                size_t count = 1;
                
                for (const auto& source_pair : layer_outputs) {
                    if (source_pair.first != target_modality) {
                        Tensor cross_output = cross_attention_layers_[layer]->forward(
                            source_pair.second, target_state, 
                            source_pair.first, target_modality
                        );
                        
                        // Simple addition for cross-modal fusion
                        const auto& combined_data = combined_cross_attention.data();
                        const auto& cross_data = cross_output.data();
                        auto& output_data = combined_cross_attention.data();
                        
                        for (size_t i = 0; i < combined_data.size() && i < cross_data.size(); ++i) {
                            output_data[i] = combined_data[i] + cross_data[i];
                        }
                        count++;
                    }
                }
                
                // Average the cross-attention results
                if (count > 1) {
                    float scale = 1.0f / count;
                    for (auto& val : combined_cross_attention.data()) {
                        val *= scale;
                    }
                }
                
                cross_attended[target_modality] = combined_cross_attention;
            }
            
            layer_outputs = cross_attended;
        }
        
        current_states = layer_outputs;
    }
    
    return current_states;
}

Tensor MultiModalTransformer::generate_conditioned(Modality target_modality,
                                                  const std::unordered_map<Modality, Tensor>& conditioning_modalities,
                                                  size_t max_length) {
    // Simple implementation: use forward pass and extract target modality output
    auto outputs = forward(conditioning_modalities);
    
    auto it = outputs.find(target_modality);
    if (it != outputs.end()) {
        return it->second;
    }
    
    // If target modality not found, create a simple output tensor
    size_t batch_size = 1;
    size_t feature_dim = hidden_dim_;
    
    if (!conditioning_modalities.empty()) {
        const auto& first_input = conditioning_modalities.begin()->second;
        if (!first_input.shape().empty()) {
            batch_size = first_input.shape()[0];
        }
    }
    
    return Tensor({batch_size, max_length, feature_dim});
}

void MultiModalTransformer::initialize_layers() {
    // Initialize input projections for each modality
    for (const auto& pair : modality_configs_) {
        Modality modality = pair.first;
        size_t input_dim = pair.second;
        
        // Create projection matrix from input_dim to hidden_dim
        input_projections_[modality] = Tensor({input_dim, hidden_dim_});
        
        // Xavier initialization
        float scale = std::sqrt(2.0f / (input_dim + hidden_dim_));
        auto& data = input_projections_[modality].data();
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        
        // Initialize self-attention layers for this modality
        std::vector<std::unique_ptr<MultiHeadAttentionLayer>> self_attention_layers;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            self_attention_layers.push_back(
                std::make_unique<MultiHeadAttentionLayer>(hidden_dim_, num_heads_)
            );
        }
        self_attention_layers_[modality] = std::move(self_attention_layers);
        
        // Initialize layer normalization weights
        std::vector<Tensor> layer_norm_weights;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            layer_norm_weights.push_back(Tensor({hidden_dim_}));
            // Initialize to ones for layer norm
            std::fill(layer_norm_weights.back().data().begin(), layer_norm_weights.back().data().end(), 1.0f);
        }
        layer_norms_[modality] = std::move(layer_norm_weights);
    }
    
    // Initialize cross-modal attention layers
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        cross_attention_layers_.push_back(
            std::make_unique<CrossModalAttention>(hidden_dim_, hidden_dim_, hidden_dim_, num_heads_)
        );
    }
}

Tensor MultiModalTransformer::apply_layer_norm(const Tensor& input, const Tensor& norm_weights) {
    // Simple layer normalization implementation
    Tensor output = input;
    const auto& input_data = input.data();
    auto& output_data = output.data();
    const auto& norm_data = norm_weights.data();
    
    // Simple element-wise multiplication with norm weights
    for (size_t i = 0; i < input_data.size() && i < norm_data.size(); ++i) {
        output_data[i] = input_data[i] * norm_data[i % norm_data.size()];
    }
    
    return output;
}

Tensor MultiModalTransformer::apply_feed_forward(const Tensor& input, const Tensor& weights) {
    // Simple feed-forward network (linear transformation)
    return input.matmul(weights);
}

// MultiModalUtils Implementation
namespace MultiModalUtils {

MultiModalTensor to_multimodal(const Tensor& tensor, Modality modality, const std::string& description) {
    return MultiModalTensor(tensor, modality, description);
}

std::unordered_map<Modality, Tensor> align_sequences(const std::unordered_map<Modality, Tensor>& sequences) {
    // Find maximum sequence length
    size_t max_length = 0;
    for (const auto& pair : sequences) {
        const auto& shape = pair.second.shape();
        if (shape.size() > 1) {
            max_length = std::max(max_length, shape[1]);
        }
    }
    
    std::unordered_map<Modality, Tensor> aligned_sequences;
    for (const auto& pair : sequences) {
        // For now, just copy the tensor as-is
        // In a real implementation, this would pad or truncate sequences
        aligned_sequences[pair.first] = pair.second;
    }
    
    return aligned_sequences;
}

Tensor compute_cross_modal_similarity(const Tensor& features1, const Tensor& features2) {
    // Compute cosine similarity matrix
    return features1.matmul(features2.transpose());
}

Tensor interpolate_modalities(const Tensor& modal1_features, const Tensor& modal2_features, double interpolation_factor) {
    // Linear interpolation between modalities
    Tensor result = modal1_features;
    const auto& data1 = modal1_features.data();
    const auto& data2 = modal2_features.data();
    
    auto& result_data = result.data();
    for (size_t i = 0; i < data1.size() && i < data2.size(); ++i) {
        result_data[i] = static_cast<float>((1.0 - interpolation_factor) * data1[i] + interpolation_factor * data2[i]);
    }
    
    return result;
}

} // namespace MultiModalUtils

} // namespace ai
} // namespace asekioml
