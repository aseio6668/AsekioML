#pragma once

#include "tensor.hpp"
#include "ai/orchestral_ai_workflow.hpp"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <chrono>

namespace asekioml {
namespace ai {

// Forward declarations
class GuidanceController;
class ModalBridgeNetwork;
class AdaptivePipelineManager;
class MultiModalCoordinator;

/**
 * @brief Types of cross-modal guidance mechanisms
 */
enum class GuidanceType {
    ATTENTION_BASED,    // Attention-driven guidance
    FEATURE_ALIGNMENT,  // Feature space alignment
    SEMANTIC_BRIDGE,    // Semantic concept bridging
    TEMPORAL_SYNC,      // Temporal synchronization
    ADAPTIVE_WEIGHTED   // Adaptive weight-based guidance
};

/**
 * @brief Guidance effectiveness metrics
 */
struct GuidanceMetrics {
    double effectiveness_score;     // Overall guidance effectiveness (0-1)
    double alignment_quality;       // Cross-modal alignment quality (0-1)
    double semantic_consistency;    // Semantic consistency score (0-1)
    double temporal_coherence;      // Temporal coherence measure (0-1)
    double computational_cost;      // Computational overhead in ms
    std::string guidance_mode;      // Current guidance mode description
    
    GuidanceMetrics() : effectiveness_score(0.0), alignment_quality(0.0),
                       semantic_consistency(0.0), temporal_coherence(0.0),
                       computational_cost(0.0) {}
};

/**
 * @brief Configuration for cross-modal guidance
 */
struct GuidanceConfig {
    GuidanceType guidance_type;
    double guidance_strength;       // Strength of guidance (0-1)
    double adaptation_rate;         // Rate of adaptive adjustment (0-1)
    double semantic_threshold;      // Minimum semantic similarity threshold
    bool enable_temporal_sync;      // Enable temporal synchronization
    bool adaptive_mode;             // Enable adaptive guidance adjustment
    std::map<std::string, double> modality_weights;  // Per-modality guidance weights
    
    GuidanceConfig() : guidance_type(GuidanceType::ATTENTION_BASED),
                      guidance_strength(0.5), adaptation_rate(0.1),
                      semantic_threshold(0.3), enable_temporal_sync(true),
                      adaptive_mode(true) {}
};

/**
 * @brief Advanced cross-modal attention mechanism for guidance (Week 14)
 */
class AdvancedCrossModalAttention {
public:
    AdvancedCrossModalAttention(size_t query_dim, size_t key_dim, size_t value_dim, size_t num_heads = 8);
    ~AdvancedCrossModalAttention();
    
    // Attention computation
    Tensor forward(const Tensor& query_modality, const Tensor& key_modality, 
                  const Tensor& value_modality, const Tensor& guidance_mask = Tensor());
    Tensor computeAttentionWeights(const Tensor& query, const Tensor& key) const;
    Tensor applyGuidance(const Tensor& attention_weights, const Tensor& guidance_signal) const;
    
    // Configuration
    void setGuidanceStrength(double strength) { guidance_strength_ = strength; }
    void setTemperature(double temperature) { attention_temperature_ = temperature; }
    double getGuidanceStrength() const { return guidance_strength_; }
    
    // Analysis
    GuidanceMetrics computeGuidanceMetrics(const Tensor& source, const Tensor& target) const;
    Tensor getLastAttentionWeights() const { return last_attention_weights_; }
    
private:
    size_t query_dim_, key_dim_, value_dim_, num_heads_;
    double guidance_strength_;
    double attention_temperature_;
    mutable Tensor last_attention_weights_;  // Mutable to allow assignment in const methods
    
    // Learned parameters (simplified representation)
    Tensor query_projection_;
    Tensor key_projection_;
    Tensor value_projection_;
    Tensor output_projection_;
    
    void initializeParameters();
    Tensor multiHeadAttention(const Tensor& Q, const Tensor& K, const Tensor& V) const;
    Tensor applySoftmax(const Tensor& input) const;  // Helper for softmax operation
};

/**
 * @brief Main cross-modal conditioner for advanced guidance mechanisms
 */
class CrossModalConditioner {
public:
    CrossModalConditioner(const GuidanceConfig& config = GuidanceConfig());
    ~CrossModalConditioner();
    
    // Core conditioning methods
    Tensor conditionModality(const Tensor& target_modality, const Tensor& source_modality,
                            const std::string& target_type, const std::string& source_type);
    std::map<std::string, Tensor> conditionMultiModal(const std::map<std::string, Tensor>& modalities);
    
    // Guidance mechanisms
    Tensor applyAttentionGuidance(const Tensor& target, const Tensor& source);
    Tensor applyFeatureAlignment(const Tensor& target, const Tensor& source);
    Tensor applySemanticBridge(const Tensor& target, const Tensor& source);
    Tensor applyTemporalSync(const Tensor& target, const Tensor& source);
    
    // Configuration and adaptation
    void setGuidanceConfig(const GuidanceConfig& config) { config_ = config; }
    void adaptGuidanceStrength(const GuidanceMetrics& metrics);
    const GuidanceConfig& getGuidanceConfig() const { return config_; }
    
    // Metrics and analysis
    GuidanceMetrics getLastGuidanceMetrics() const { return last_metrics_; }
    std::map<std::string, double> getModalityAffinities() const;
    void resetGuidanceHistory();
    
private:    GuidanceConfig config_;
    GuidanceMetrics last_metrics_;
    std::unique_ptr<AdvancedCrossModalAttention> attention_mechanism_;
    std::vector<GuidanceMetrics> guidance_history_;
    std::map<std::string, double> modality_affinities_;
    
    void updateGuidanceHistory(const GuidanceMetrics& metrics);
    double computeSemanticSimilarity(const Tensor& tensor1, const Tensor& tensor2) const;
    Tensor alignFeatureSpaces(const Tensor& source, const Tensor& target) const;
    void initializeAttentionMechanism();
};

/**
 * @brief Controller for dynamic guidance strength adjustment
 */
class GuidanceController {
public:
    GuidanceController(double initial_strength = 0.5, double adaptation_rate = 0.1);
    ~GuidanceController();
    
    // Guidance strength management
    double getCurrentStrength() const { return current_strength_; }
    void setTargetStrength(double target_strength);
    void updateStrength(const GuidanceMetrics& feedback);
    void adaptStrength(double effectiveness, double quality_threshold = 0.7);
    
    // Adaptive control
    void enableAdaptiveControl(bool enable) { adaptive_control_enabled_ = enable; }
    void setAdaptationRate(double rate) { adaptation_rate_ = rate; }
    void setQualityThreshold(double threshold) { quality_threshold_ = threshold; }
    
    // Analysis and monitoring
    std::vector<double> getStrengthHistory() const { return strength_history_; }
    double getAverageEffectiveness() const;
    bool isStabilized() const;
    void resetController();
    
private:
    double current_strength_;
    double target_strength_;
    double adaptation_rate_;
    double quality_threshold_;
    bool adaptive_control_enabled_;
    
    std::vector<double> strength_history_;
    std::vector<double> effectiveness_history_;
    size_t max_history_size_;
    
    void updateHistory(double strength, double effectiveness);
    double computeAdaptationDelta(double effectiveness) const;
};

/**
 * @brief Network for sophisticated inter-modal communication
 */
class ModalBridgeNetwork {
public:
    ModalBridgeNetwork(const std::vector<std::string>& modality_types);
    ~ModalBridgeNetwork();
    
    // Bridge construction and management
    bool createBridge(const std::string& source_modality, const std::string& target_modality);
    bool removeBridge(const std::string& source_modality, const std::string& target_modality);
    std::vector<std::string> getConnectedModalities(const std::string& modality) const;
    
    // Feature bridging
    Tensor bridgeFeatures(const Tensor& source_features, const std::string& source_type,
                         const std::string& target_type);
    std::map<std::string, Tensor> broadcastFeatures(const Tensor& features, 
                                                   const std::string& source_type);
    
    // Semantic alignment
    Tensor alignSemanticSpaces(const Tensor& source, const Tensor& target,
                              const std::string& source_type, const std::string& target_type);
    double computeSemanticDistance(const Tensor& features1, const Tensor& features2) const;
    
    // Network analysis
    std::map<std::pair<std::string, std::string>, double> getBridgeStrengths() const;
    std::vector<std::string> findOptimalPath(const std::string& source, const std::string& target) const;
    void optimizeBridgeNetwork();
    
    // Configuration
    void setBridgeStrength(const std::string& source, const std::string& target, double strength);
    void setSemanticThreshold(double threshold) { semantic_threshold_ = threshold; }
    
private:
    std::vector<std::string> modality_types_;
    std::map<std::pair<std::string, std::string>, double> bridge_strengths_;
    std::map<std::pair<std::string, std::string>, Tensor> bridge_transformations_;
    double semantic_threshold_;
    
    void initializeBridges();
    Tensor createBridgeTransformation(const std::string& source_type, const std::string& target_type);
    bool isValidModalityType(const std::string& modality_type) const;
};

/**
 * @brief Manager for adaptive pipeline modification based on guidance effectiveness
 */
class AdaptivePipelineManager {
public:
    AdaptivePipelineManager(std::shared_ptr<WorkflowManager> workflow_manager);
    ~AdaptivePipelineManager();
    
    // Pipeline adaptation
    bool adaptPipeline(const std::string& workflow_name, const GuidanceMetrics& metrics);
    bool optimizePipelineForGuidance(const std::string& workflow_name);
    void revertPipelineChanges(const std::string& workflow_name);
    
    // Adaptation strategies
    void addAdaptationStage(const std::string& workflow_name, const TaskConfig& stage_config);
    void modifyStageParameters(const std::string& workflow_name, const std::string& stage_name,
                              const std::map<std::string, std::string>& new_parameters);
    void adjustStagePriorities(const std::string& workflow_name, const GuidanceMetrics& metrics);
    
    // Real-time monitoring
    void enableRealTimeAdaptation(bool enable) { real_time_adaptation_ = enable; }
    void setAdaptationThreshold(double threshold) { adaptation_threshold_ = threshold; }
    std::map<std::string, std::vector<std::string>> getPipelineModifications() const;
    
    // Performance tracking
    std::map<std::string, GuidanceMetrics> getWorkflowGuidanceMetrics() const;
    void recordGuidancePerformance(const std::string& workflow_name, const GuidanceMetrics& metrics);
    
private:
    std::shared_ptr<WorkflowManager> workflow_manager_;
    bool real_time_adaptation_;
    double adaptation_threshold_;
    
    std::map<std::string, std::vector<std::string>> pipeline_modifications_;
    std::map<std::string, std::vector<GuidanceMetrics>> workflow_guidance_history_;
    std::map<std::string, std::shared_ptr<PipelineBuilder>> original_pipelines_;
    
    void savePipelineState(const std::string& workflow_name);
    bool shouldAdaptPipeline(const GuidanceMetrics& metrics) const;
    TaskConfig createAdaptationStage(const GuidanceMetrics& metrics) const;
};

/**
 * @brief High-level coordinator for multi-modal guidance systems
 */
class MultiModalCoordinator {
public:
    MultiModalCoordinator();
    ~MultiModalCoordinator();
    
    // System coordination
    bool initialize(const std::vector<std::string>& modality_types);
    bool coordinateGuidance(const std::map<std::string, Tensor>& modalities,
                           const GuidanceConfig& config = GuidanceConfig());
    void shutdown();
    
    // Component access
    CrossModalConditioner& getConditioner() { return *conditioner_; }
    GuidanceController& getGuidanceController() { return *guidance_controller_; }
    ModalBridgeNetwork& getBridgeNetwork() { return *bridge_network_; }
    AdaptivePipelineManager& getPipelineManager() { return *pipeline_manager_; }
    
    // Coordination workflows
    std::map<std::string, Tensor> executeGuidedWorkflow(const std::string& workflow_name,
                                                       const std::map<std::string, Tensor>& inputs);
    bool registerGuidedWorkflow(const std::string& workflow_name,
                               std::shared_ptr<PipelineBuilder> pipeline,
                               const GuidanceConfig& guidance_config);
    
    // System monitoring
    std::map<std::string, GuidanceMetrics> getSystemGuidanceMetrics() const;
    std::vector<std::string> getActiveGuidanceWorkflows() const;
    void optimizeGuidanceSystem();
    
    // Configuration
    void setGlobalGuidanceConfig(const GuidanceConfig& config);
    void setWorkflowGuidanceConfig(const std::string& workflow_name, const GuidanceConfig& config);
    
private:
    std::unique_ptr<CrossModalConditioner> conditioner_;
    std::unique_ptr<GuidanceController> guidance_controller_;
    std::unique_ptr<ModalBridgeNetwork> bridge_network_;
    std::unique_ptr<AdaptivePipelineManager> pipeline_manager_;
    std::shared_ptr<WorkflowManager> workflow_manager_;
    
    std::map<std::string, GuidanceConfig> workflow_guidance_configs_;
    std::map<std::string, GuidanceMetrics> system_guidance_metrics_;
    std::vector<std::string> supported_modalities_;
    bool is_initialized_;
    
    void setupComponents();
    void connectComponents();
    bool validateModalities(const std::map<std::string, Tensor>& modalities) const;
};

/**
 * @brief Utility functions for cross-modal guidance analysis
 */
namespace CrossModalUtils {
    
    // Similarity and distance metrics
    double computeCosineSimilarity(const Tensor& tensor1, const Tensor& tensor2);
    double computeEuclideanDistance(const Tensor& tensor1, const Tensor& tensor2);
    double computeKLDivergence(const Tensor& tensor1, const Tensor& tensor2);
    
    // Feature space analysis
    Tensor computePrincipalComponents(const Tensor& features, size_t num_components);
    Tensor projectToCommonSpace(const Tensor& features, const Tensor& projection_matrix);
    std::pair<Tensor, Tensor> computeCanonicalCorrelation(const Tensor& modality1, const Tensor& modality2);
    
    // Temporal alignment utilities
    std::vector<std::pair<size_t, size_t>> computeOptimalAlignment(const Tensor& sequence1, const Tensor& sequence2);
    Tensor interpolateTemporalFeatures(const Tensor& features, const std::vector<double>& time_points);
    double computeTemporalCoherence(const Tensor& sequence1, const Tensor& sequence2);
    
    // Guidance effectiveness evaluation
    GuidanceMetrics evaluateGuidanceEffectiveness(const Tensor& guided_output, const Tensor& reference_output);
    double computeGuidanceStability(const std::vector<GuidanceMetrics>& metrics_history);
    std::map<std::string, double> analyzeModalContributions(const std::map<std::string, Tensor>& modalities,
                                                           const Tensor& guided_output);
}

} // namespace ai
} // namespace asekioml
