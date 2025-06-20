#pragma once

#include "tensor.hpp"
#include "ai/orchestral_ai_director.hpp"
#include "ai/dynamic_model_dispatcher.hpp"
#include "ai/video_audio_text_fusion.hpp"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <chrono>
#include <queue>
#include <mutex>
#include <atomic>

namespace asekioml {
namespace ai {

/**
 * @brief Adaptive Quality Engine - Real-time quality optimization and model selection
 * 
 * This class provides dynamic quality threshold adjustment, intelligent quality-speed 
 * trade-off optimization, and adaptive model selection based on quality requirements.
 */
class AdaptiveQualityEngine {
public:
    /**
     * @brief Quality optimization strategy
     */
    enum class OptimizationStrategy {
        QUALITY_FIRST,      // Prioritize quality over speed
        SPEED_FIRST,        // Prioritize speed over quality
        BALANCED,           // Balance quality and speed
        ADAPTIVE,           // Dynamically adapt based on system state
        CUSTOM              // User-defined optimization function
    };

    /**
     * @brief Quality assessment metrics
     */
    struct QualityMetrics {
        double overall_quality;
        double visual_quality;
        double audio_quality;
        double text_coherence;
        double temporal_consistency;
        double cross_modal_alignment;
        double processing_efficiency;
        double user_satisfaction_score;
        std::chrono::steady_clock::time_point timestamp;
        
        QualityMetrics() : overall_quality(0.0), visual_quality(0.0),
                          audio_quality(0.0), text_coherence(0.0),
                          temporal_consistency(0.0), cross_modal_alignment(0.0),
                          processing_efficiency(0.0), user_satisfaction_score(0.0) {}
    };

    /**
     * @brief Quality prediction model
     */
    struct QualityPrediction {
        double predicted_quality;
        double confidence_score;
        double estimated_processing_time_ms;
        std::string recommended_model;
        std::map<std::string, double> optimization_suggestions;
        
        QualityPrediction() : predicted_quality(0.0), confidence_score(0.0),
                             estimated_processing_time_ms(0.0) {}
    };

    /**
     * @brief Adaptive thresholds configuration
     */
    struct AdaptiveThresholds {
        double min_quality_threshold;
        double max_quality_threshold;
        double target_quality;
        double quality_tolerance;
        double speed_weight;
        double quality_weight;
        std::chrono::milliseconds adaptation_interval;
        double learning_rate;
        
        AdaptiveThresholds() : min_quality_threshold(0.6),
                              max_quality_threshold(0.95),
                              target_quality(0.85),
                              quality_tolerance(0.05),
                              speed_weight(0.3),
                              quality_weight(0.7),
                              adaptation_interval(std::chrono::milliseconds(1000)),
                              learning_rate(0.1) {}
    };

    /**
     * @brief Quality optimization context
     */
    struct OptimizationContext {
        std::string content_type;
        std::map<std::string, double> requirements;
        double system_load;
        double available_resources;
        std::vector<QualityMetrics> recent_metrics;
        std::map<std::string, double> model_performance_history;
        
        OptimizationContext() : system_load(0.5), available_resources(1.0) {}
    };

public:
    /**
     * @brief Constructor
     */
    AdaptiveQualityEngine(const AdaptiveThresholds& thresholds = AdaptiveThresholds{});
    
    /**
     * @brief Destructor
     */
    ~AdaptiveQualityEngine();

    // Core quality optimization
    
    /**
     * @brief Initialize the quality engine
     */
    bool initialize();
    
    /**
     * @brief Shutdown the quality engine
     */
    void shutdown();
    
    /**
     * @brief Set optimization strategy
     */
    void setOptimizationStrategy(OptimizationStrategy strategy);
    
    /**
     * @brief Assess content quality
     */
    QualityMetrics assessQuality(const MultiModalContent& content);
    
    /**
     * @brief Predict quality for given configuration
     */
    QualityPrediction predictQuality(const MultiModalContent& input, 
                                   const std::string& model_id,
                                   const OptimizationContext& context);
    
    /**
     * @brief Optimize quality thresholds based on recent performance
     */
    void optimizeThresholds(const std::vector<QualityMetrics>& recent_metrics);
    
    /**
     * @brief Get optimal model selection for quality requirements
     */
    std::string selectOptimalModel(const std::vector<std::string>& available_models,
                                  const OptimizationContext& context);

    // Quality monitoring and feedback
    
    /**
     * @brief Record quality metrics for learning
     */
    void recordQualityMetrics(const std::string& model_id, 
                             const QualityMetrics& metrics,
                             double processing_time_ms);
    
    /**
     * @brief Get current quality trends
     */
    std::map<std::string, double> getQualityTrends() const;
    
    /**
     * @brief Get optimization recommendations
     */
    std::map<std::string, std::string> getOptimizationRecommendations(
        const OptimizationContext& context) const;

    // Adaptive learning
    
    /**
     * @brief Update quality prediction model
     */
    void updatePredictionModel(const std::vector<QualityMetrics>& training_data);
    
    /**
     * @brief Get quality-speed trade-off curve
     */
    std::vector<std::pair<double, double>> getQualitySpeedCurve(
        const std::string& content_type) const;
    
    /**
     * @brief Calculate optimal quality target for current conditions
     */
    double calculateOptimalQualityTarget(const OptimizationContext& context) const;

    // Configuration and status
    
    /**
     * @brief Update adaptive thresholds
     */
    void updateThresholds(const AdaptiveThresholds& thresholds);
    
    /**
     * @brief Get current thresholds
     */
    AdaptiveThresholds getCurrentThresholds() const;
    
    /**
     * @brief Get engine statistics
     */
    std::map<std::string, double> getEngineStatistics() const;
    
    /**
     * @brief Generate quality analysis report
     */
    std::string generateQualityReport() const;

private:
    // Configuration
    AdaptiveThresholds thresholds_;
    OptimizationStrategy strategy_;
    
    // Quality tracking
    std::map<std::string, std::vector<QualityMetrics>> quality_history_;
    std::map<std::string, double> model_quality_scores_;
    std::map<std::string, double> model_speed_scores_;
    mutable std::mutex metrics_mutex_;
    
    // Adaptive learning
    std::atomic<bool> is_running_{false};
    std::thread adaptation_thread_;
    
    // Statistics
    std::map<std::string, size_t> assessment_counts_;
    std::map<std::string, double> total_quality_scores_;
    std::map<std::string, double> total_processing_times_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    void adaptationLoop();
    double calculateQualityScore(const MultiModalContent& content) const;
    double calculateVisualQuality(const Tensor& visual_features) const;
    double calculateAudioQuality(const Tensor& audio_features) const;
    double calculateTextCoherence(const Tensor& text_features) const;
    double calculateTemporalConsistency(const MultiModalContent& content) const;
    double calculateCrossModalAlignment(const MultiModalContent& content) const;
    double calculateProcessingEfficiency(double processing_time, double content_complexity) const;
    
    std::string getBestModelForQuality(const std::vector<std::string>& models,
                                      double target_quality) const;
    std::string getBestModelForSpeed(const std::vector<std::string>& models) const;
    std::string getBestModelBalanced(const std::vector<std::string>& models,
                                   const OptimizationContext& context) const;
    
    void updateModelScores(const std::string& model_id, 
                          const QualityMetrics& metrics, 
                          double processing_time);
    
    double computeQualityTrend(const std::vector<QualityMetrics>& metrics) const;
    void generateOptimizationSuggestions(const OptimizationContext& context,
                                        std::map<std::string, std::string>& suggestions) const;
};

} // namespace ai
} // namespace asekioml
