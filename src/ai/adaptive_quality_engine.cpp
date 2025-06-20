#include "ai/adaptive_quality_engine.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <iostream>
#include <iomanip>

namespace asekioml {
namespace ai {

AdaptiveQualityEngine::AdaptiveQualityEngine(const AdaptiveThresholds& thresholds) 
    : thresholds_(thresholds), strategy_(OptimizationStrategy::BALANCED) {
    // Initialize default settings
}

AdaptiveQualityEngine::~AdaptiveQualityEngine() {
    shutdown();
}

bool AdaptiveQualityEngine::initialize() {
    is_running_.store(true);
    adaptation_thread_ = std::thread(&AdaptiveQualityEngine::adaptationLoop, this);
    return true;
}

void AdaptiveQualityEngine::shutdown() {
    is_running_.store(false);
    if (adaptation_thread_.joinable()) {
        adaptation_thread_.join();
    }
    
    std::cout << "AdaptiveQualityEngine: Shutdown complete\n";
}

void AdaptiveQualityEngine::setOptimizationStrategy(OptimizationStrategy strategy) {
    strategy_ = strategy;
}

AdaptiveQualityEngine::QualityMetrics AdaptiveQualityEngine::assessQuality(const MultiModalContent& content) {
    QualityMetrics metrics;
    
    // Calculate individual quality components
    metrics.overall_quality = calculateQualityScore(content);
    metrics.visual_quality = calculateVisualQuality(content.video_features);
    metrics.audio_quality = calculateAudioQuality(content.audio_features);
    metrics.text_coherence = calculateTextCoherence(content.text_features);
    metrics.temporal_consistency = calculateTemporalConsistency(content);
    metrics.cross_modal_alignment = calculateCrossModalAlignment(content);
    metrics.processing_efficiency = calculateProcessingEfficiency(100.0, 1.0); // Default values
    
    // Set timestamp
    metrics.timestamp = std::chrono::steady_clock::now();
    
    return metrics;
}

AdaptiveQualityEngine::QualityPrediction AdaptiveQualityEngine::predictQuality(
    const MultiModalContent& input, 
    const std::string& model_id,
    const OptimizationContext& context) {
    
    QualityPrediction prediction;
    
    // Look up model performance history
    auto quality_it = model_quality_scores_.find(model_id);
    auto speed_it = model_speed_scores_.find(model_id);
    
    if (quality_it != model_quality_scores_.end()) {
        prediction.predicted_quality = quality_it->second;
        prediction.confidence_score = 0.8; // High confidence for known models
    } else {
        prediction.predicted_quality = 0.7; // Default estimate
        prediction.confidence_score = 0.3; // Low confidence for unknown models
    }
    
    if (speed_it != model_speed_scores_.end()) {
        prediction.estimated_processing_time_ms = speed_it->second;
    } else {
        prediction.estimated_processing_time_ms = 100.0; // Default estimate
    }
    
    prediction.recommended_model = model_id;
    
    return prediction;
}

void AdaptiveQualityEngine::optimizeThresholds(const std::vector<QualityMetrics>& recent_metrics) {
    if (recent_metrics.empty()) return;
    
    // Calculate average quality from recent metrics
    double avg_quality = 0.0;
    for (const auto& metrics : recent_metrics) {
        avg_quality += metrics.overall_quality;
    }
    avg_quality /= recent_metrics.size();
    
    // Adapt thresholds based on recent performance
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (avg_quality < thresholds_.target_quality - thresholds_.quality_tolerance) {
        // Quality is too low, adjust thresholds
        thresholds_.target_quality = std::max(thresholds_.min_quality_threshold,
                                            thresholds_.target_quality - thresholds_.learning_rate);
    } else if (avg_quality > thresholds_.target_quality + thresholds_.quality_tolerance) {
        // Quality is good, can aim higher
        thresholds_.target_quality = std::min(thresholds_.max_quality_threshold,
                                            thresholds_.target_quality + thresholds_.learning_rate);
    }
}

std::string AdaptiveQualityEngine::selectOptimalModel(
    const std::vector<std::string>& available_models,
    const OptimizationContext& context) {
    
    if (available_models.empty()) {
        return "";
    }
    
    switch (strategy_) {
        case OptimizationStrategy::QUALITY_FIRST:
            return getBestModelForQuality(available_models, thresholds_.target_quality);
        
        case OptimizationStrategy::SPEED_FIRST:
            return getBestModelForSpeed(available_models);
        
        case OptimizationStrategy::BALANCED:
        case OptimizationStrategy::ADAPTIVE:
            return getBestModelBalanced(available_models, context);
        
        case OptimizationStrategy::CUSTOM:
        default:
            return available_models[0]; // Fallback to first model
    }
}

void AdaptiveQualityEngine::recordQualityMetrics(const std::string& model_id, 
                                                const QualityMetrics& metrics,
                                                double processing_time_ms) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Store in quality history
    quality_history_[model_id].push_back(metrics);
    
    // Keep only recent metrics (last 100)
    if (quality_history_[model_id].size() > 100) {
        quality_history_[model_id].erase(quality_history_[model_id].begin());
    }
    
    // Update model scores
    updateModelScores(model_id, metrics, processing_time_ms);
}

std::map<std::string, double> AdaptiveQualityEngine::getQualityTrends() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    std::map<std::string, double> trends;
    
    for (const auto& [model_id, history] : quality_history_) {
        if (!history.empty()) {
            trends[model_id] = computeQualityTrend(history);
        }
    }
    
    return trends;
}

std::map<std::string, std::string> AdaptiveQualityEngine::getOptimizationRecommendations(
    const OptimizationContext& context) const {
    
    std::map<std::string, std::string> recommendations;
    generateOptimizationSuggestions(context, recommendations);
    return recommendations;
}

void AdaptiveQualityEngine::updatePredictionModel(const std::vector<QualityMetrics>& training_data) {
    // Simple learning: update model averages based on training data
    // In a real implementation, this would use more sophisticated ML algorithms
    
    for (const auto& metrics : training_data) {
        // Update overall statistics
        // This is a simplified approach
    }
}

std::vector<std::pair<double, double>> AdaptiveQualityEngine::getQualitySpeedCurve(
    const std::string& content_type) const {
    
    std::vector<std::pair<double, double>> curve;
    
    // Generate a sample quality-speed curve
    // In reality, this would be based on actual model performance data
    for (double quality = 0.5; quality <= 1.0; quality += 0.1) {
        double speed = 100.0 / (quality * quality); // Inverse relationship
        curve.emplace_back(quality, speed);
    }
    
    return curve;
}

double AdaptiveQualityEngine::calculateOptimalQualityTarget(const OptimizationContext& context) const {
    double base_target = thresholds_.target_quality;
    
    // Adjust based on system load
    if (context.system_load > 0.8) {
        base_target *= 0.9; // Reduce quality target under high load
    } else if (context.system_load < 0.3) {
        base_target *= 1.1; // Increase quality target under low load
    }
    
    return std::clamp(base_target, thresholds_.min_quality_threshold, thresholds_.max_quality_threshold);
}

void AdaptiveQualityEngine::updateThresholds(const AdaptiveThresholds& thresholds) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    thresholds_ = thresholds;
}

AdaptiveQualityEngine::AdaptiveThresholds AdaptiveQualityEngine::getCurrentThresholds() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return thresholds_;
}

std::map<std::string, double> AdaptiveQualityEngine::getEngineStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    std::map<std::string, double> stats;
    
    stats["total_assessments"] = static_cast<double>(assessment_counts_.size());
    stats["average_quality"] = 0.0;
    stats["average_processing_time"] = 0.0;
    
    if (!total_quality_scores_.empty()) {
        double total_quality = 0.0;
        for (const auto& [model, score] : total_quality_scores_) {
            total_quality += score;
        }
        stats["average_quality"] = total_quality / total_quality_scores_.size();
    }
    
    if (!total_processing_times_.empty()) {
        double total_time = 0.0;
        for (const auto& [model, time] : total_processing_times_) {
            total_time += time;
        }
        stats["average_processing_time"] = total_time / total_processing_times_.size();
    }
    
    return stats;
}

std::string AdaptiveQualityEngine::generateQualityReport() const {
    std::ostringstream report;
    report << "=== Adaptive Quality Engine Report ===" << std::endl;
    
    auto thresholds = getCurrentThresholds();
    auto stats = getEngineStatistics();
    
    report << "Strategy: ";
    switch (strategy_) {
        case OptimizationStrategy::QUALITY_FIRST: report << "Quality First"; break;
        case OptimizationStrategy::SPEED_FIRST: report << "Speed First"; break;
        case OptimizationStrategy::BALANCED: report << "Balanced"; break;
        case OptimizationStrategy::ADAPTIVE: report << "Adaptive"; break;
        case OptimizationStrategy::CUSTOM: report << "Custom"; break;
    }
    report << std::endl;
    
    report << "Target Quality: " << std::fixed << std::setprecision(3) 
           << thresholds.target_quality << std::endl;
    report << "Quality Range: [" << thresholds.min_quality_threshold 
           << ", " << thresholds.max_quality_threshold << "]" << std::endl;
    
    if (stats.find("average_quality") != stats.end()) {
        report << "Average Quality: " << std::fixed << std::setprecision(3) 
               << stats.at("average_quality") << std::endl;
    }
    
    if (stats.find("average_processing_time") != stats.end()) {
        report << "Average Processing Time: " << std::fixed << std::setprecision(1) 
               << stats.at("average_processing_time") << " ms" << std::endl;
    }
    
    return report.str();
}

// Private methods implementation

void AdaptiveQualityEngine::adaptationLoop() {
    while (is_running_.load()) {
        std::this_thread::sleep_for(thresholds_.adaptation_interval);
        
        if (!is_running_.load()) break;
        
        // Perform periodic adaptation
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Collect recent metrics for adaptation
        std::vector<QualityMetrics> all_recent_metrics;
        for (const auto& [model_id, history] : quality_history_) {
            if (!history.empty()) {
                // Take last 10 metrics
                size_t start = history.size() > 10 ? history.size() - 10 : 0;
                for (size_t i = start; i < history.size(); ++i) {
                    all_recent_metrics.push_back(history[i]);
                }
            }
        }
        
        if (!all_recent_metrics.empty()) {
            optimizeThresholds(all_recent_metrics);
        }
    }
}

double AdaptiveQualityEngine::calculateQualityScore(const MultiModalContent& content) const {
    // Weighted combination of different quality aspects
    double visual_weight = 0.4;
    double audio_weight = 0.3;
    double text_weight = 0.3;
    
    double visual_score = calculateVisualQuality(content.video_features);
    double audio_score = calculateAudioQuality(content.audio_features);
    double text_score = calculateTextCoherence(content.text_features);
    
    return visual_weight * visual_score + audio_weight * audio_score + text_weight * text_score;
}

double AdaptiveQualityEngine::calculateVisualQuality(const Tensor& visual_features) const {
    // Simplified visual quality assessment
    if (visual_features.data().empty()) return 0.0;
    
    // In a real implementation, this would analyze image quality metrics
    // like sharpness, noise, color accuracy, etc.
    return 0.8 + 0.2 * (static_cast<double>(rand()) / RAND_MAX);
}

double AdaptiveQualityEngine::calculateAudioQuality(const Tensor& audio_features) const {
    // Simplified audio quality assessment
    if (audio_features.data().empty()) return 0.0;
    
    // In a real implementation, this would analyze audio quality metrics
    // like clarity, noise levels, frequency response, etc.
    return 0.75 + 0.25 * (static_cast<double>(rand()) / RAND_MAX);
}

double AdaptiveQualityEngine::calculateTextCoherence(const Tensor& text_features) const {
    // Simplified text coherence assessment
    if (text_features.data().empty()) return 0.0;
    
    // In a real implementation, this would analyze semantic coherence,
    // grammar, relevance, etc.
    return 0.85 + 0.15 * (static_cast<double>(rand()) / RAND_MAX);
}

double AdaptiveQualityEngine::calculateTemporalConsistency(const MultiModalContent& content) const {
    // Simplified temporal consistency calculation
    return 0.8; // Placeholder
}

double AdaptiveQualityEngine::calculateCrossModalAlignment(const MultiModalContent& content) const {
    // Simplified cross-modal alignment calculation
    return 0.75; // Placeholder
}

double AdaptiveQualityEngine::calculateProcessingEfficiency(double processing_time, double content_complexity) const {
    if (processing_time <= 0.0 || content_complexity <= 0.0) return 0.0;
    
    // Higher efficiency for lower processing time relative to complexity
    return std::max(0.0, 1.0 - (processing_time / (content_complexity * 100.0)));
}

std::string AdaptiveQualityEngine::getBestModelForQuality(const std::vector<std::string>& models,
                                                         double target_quality) const {
    std::string best_model = models[0];
    double best_quality = 0.0;
    
    for (const auto& model : models) {
        auto it = model_quality_scores_.find(model);
        if (it != model_quality_scores_.end() && it->second > best_quality) {
            best_quality = it->second;
            best_model = model;
        }
    }
    
    return best_model;
}

std::string AdaptiveQualityEngine::getBestModelForSpeed(const std::vector<std::string>& models) const {
    std::string best_model = models[0];
    double best_speed = std::numeric_limits<double>::max();
    
    for (const auto& model : models) {
        auto it = model_speed_scores_.find(model);
        if (it != model_speed_scores_.end() && it->second < best_speed) {
            best_speed = it->second;
            best_model = model;
        }
    }
    
    return best_model;
}

std::string AdaptiveQualityEngine::getBestModelBalanced(const std::vector<std::string>& models,
                                                       const OptimizationContext& context) const {
    std::string best_model = models[0];
    double best_score = -1.0;
    
    for (const auto& model : models) {
        auto quality_it = model_quality_scores_.find(model);
        auto speed_it = model_speed_scores_.find(model);
        
        if (quality_it != model_quality_scores_.end() && speed_it != model_speed_scores_.end()) {
            double quality_score = quality_it->second;
            double speed_score = 1.0 / (1.0 + speed_it->second); // Invert speed (lower is better)
            
            double balanced_score = thresholds_.quality_weight * quality_score + 
                                  thresholds_.speed_weight * speed_score;
            
            if (balanced_score > best_score) {
                best_score = balanced_score;
                best_model = model;
            }
        }
    }
    
    return best_model;
}

void AdaptiveQualityEngine::updateModelScores(const std::string& model_id, 
                                             const QualityMetrics& metrics, 
                                             double processing_time) {
    // Update quality scores (exponential moving average)
    double alpha = 0.1; // Learning rate
    
    if (model_quality_scores_.find(model_id) != model_quality_scores_.end()) {
        model_quality_scores_[model_id] = (1.0 - alpha) * model_quality_scores_[model_id] + 
                                         alpha * metrics.overall_quality;
    } else {
        model_quality_scores_[model_id] = metrics.overall_quality;
    }
    
    if (model_speed_scores_.find(model_id) != model_speed_scores_.end()) {
        model_speed_scores_[model_id] = (1.0 - alpha) * model_speed_scores_[model_id] + 
                                       alpha * processing_time;
    } else {
        model_speed_scores_[model_id] = processing_time;
    }
}

double AdaptiveQualityEngine::computeQualityTrend(const std::vector<QualityMetrics>& metrics) const {
    if (metrics.size() < 2) return 0.0;
    
    // Simple trend calculation: compare recent vs older metrics
    size_t half = metrics.size() / 2;
    
    double older_avg = 0.0;
    double recent_avg = 0.0;
    
    for (size_t i = 0; i < half; ++i) {
        older_avg += metrics[i].overall_quality;
    }
    older_avg /= half;
    
    for (size_t i = half; i < metrics.size(); ++i) {
        recent_avg += metrics[i].overall_quality;
    }
    recent_avg /= (metrics.size() - half);
    
    return recent_avg - older_avg; // Positive = improving, Negative = degrading
}

void AdaptiveQualityEngine::generateOptimizationSuggestions(const OptimizationContext& context,
                                                          std::map<std::string, std::string>& suggestions) const {
    if (context.system_load > 0.8) {
        suggestions["system"] = "Consider reducing quality targets due to high system load";
    }
    
    if (context.available_resources < 0.5) {
        suggestions["resources"] = "Limited resources detected, recommend speed-optimized models";
    }
    
    auto trends = getQualityTrends();
    for (const auto& [model, trend] : trends) {
        if (trend < -0.1) {
            suggestions["model_" + model] = "Quality trend declining, consider model retraining";
        }
    }
}

} // namespace ai
} // namespace asekioml
