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
#include <thread>
#include <iostream>

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
        std::chrono::milliseconds processing_time;

        QualityMetrics() : overall_quality(0.0), visual_quality(0.0),
                          audio_quality(0.0), text_coherence(0.0),
                          temporal_consistency(0.0), cross_modal_alignment(0.0),
                          processing_efficiency(0.0), user_satisfaction_score(0.0),
                          processing_time(0) {}
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
     * @brief Default Constructor - Initializes with default thresholds
     */
    AdaptiveQualityEngine()
        : AdaptiveQualityEngine(AdaptiveThresholds()) {
        std::cout << "AdaptiveQualityEngine: Initialized with default thresholds" << std::endl;
    }

    /**
     * @brief Constructor with custom thresholds
     */
    AdaptiveQualityEngine(const AdaptiveThresholds& thresholds) 
        : thresholds_(thresholds),
          strategy_(OptimizationStrategy::BALANCED),
          quality_history_(),
          model_quality_scores_(),
          model_speed_scores_(),
          is_running_(false) {
        std::cout << "AdaptiveQualityEngine: Initialized with thresholds: "
                  << "min=" << thresholds.min_quality_threshold 
                  << ", max=" << thresholds.max_quality_threshold 
                  << ", target=" << thresholds.target_quality << std::endl;
    }

    /**
     * @brief Destructor
     */
    ~AdaptiveQualityEngine() {
        shutdown();
        if (adaptation_thread_.joinable()) {
            adaptation_thread_.join();
        }
    }

    // Core quality optimization
    
    /**
     * @brief Initialize the quality engine
     */
    bool initialize() {
        if (!is_running_.load()) {
            is_running_.store(true);
            adaptation_thread_ = std::thread(&AdaptiveQualityEngine::adaptationLoop, this);
            std::cout << "AdaptiveQualityEngine: Initialized and started adaptation loop" << std::endl;
            return true;
        }
        return false;
    }
    
    /**
     * @brief Shutdown the quality engine
     */
    void shutdown() {
        if (is_running_.load()) {
            is_running_.store(false);
            if (adaptation_thread_.joinable()) {
                adaptation_thread_.join();
            }
            std::cout << "AdaptiveQualityEngine: Shutdown complete" << std::endl;
        }
    }

    // Get current optimization strategy
    OptimizationStrategy getStrategy() const { return strategy_; }

    // Set optimization strategy
    void setStrategy(OptimizationStrategy strategy) { strategy_ = strategy; }

    // Check if engine is running
    bool isRunning() const { return is_running_.load(); }

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

    // Private methods
    void adaptationLoop() {
        while (is_running_.load()) {
            // Perform quality adaptation
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                // Process quality history and adjust thresholds
            }
            
            // Sleep for the adaptation interval
            std::this_thread::sleep_for(thresholds_.adaptation_interval);
        }
    }
};

} // namespace ai
} // namespace asekioml
