#pragma once

#include "tensor.hpp"
#include "ai/orchestral_ai_director.hpp"
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
 * @brief Dynamic Model Dispatcher - Intelligent routing and load balancing across specialized models
 * 
 * This class provides intelligent task routing, dynamic load balancing, and adaptive model
 * selection for the orchestral AI system.
 */
class DynamicModelDispatcher {
public:
    /**
     * @brief Task routing strategy
     */
    enum class RoutingStrategy {
        LOAD_BALANCED,      // Balance load across available models
        QUALITY_OPTIMIZED,  // Route to highest quality model
        LATENCY_OPTIMIZED,  // Route to fastest model
        ADAPTIVE,           // Adaptively choose strategy based on conditions
        ROUND_ROBIN,        // Simple round-robin distribution
        CONTENT_AWARE       // Route based on content analysis
    };

    /**
     * @brief Model performance metrics for routing decisions
     */
    struct ModelPerformance {
        std::string model_id;
        double avg_processing_time_ms;
        double quality_score;
        double current_load;          // 0.0 to 1.0
        size_t active_tasks;
        size_t completed_tasks;
        double success_rate;
        std::chrono::steady_clock::time_point last_update;
        
        // Resource utilization
        double cpu_usage;
        double memory_usage;
        double gpu_usage;
        
        ModelPerformance() : avg_processing_time_ms(0.0), quality_score(1.0),
                           current_load(0.0), active_tasks(0), completed_tasks(0),
                           success_rate(1.0), cpu_usage(0.0), memory_usage(0.0),
                           gpu_usage(0.0) {}
    };

    /**
     * @brief Task dispatch result
     */
    struct DispatchResult {
        std::string selected_model_id;
        std::string reason;
        double confidence_score;
        double estimated_processing_time_ms;
        double estimated_quality;
        bool success;
        
        DispatchResult() : confidence_score(0.0), estimated_processing_time_ms(0.0),
                          estimated_quality(0.0), success(false) {}
    };

    /**
     * @brief Content analysis for routing decisions
     */
    struct ContentAnalysis {
        std::vector<std::string> detected_modalities;
        std::map<std::string, double> complexity_scores;  // Per modality
        std::map<std::string, double> quality_requirements;
        double overall_complexity;
        double processing_priority;
        
        ContentAnalysis() : overall_complexity(0.5), processing_priority(0.5) {}
    };

    /**
     * @brief Load balancing configuration
     */
    struct LoadBalancingConfig {
        double max_model_load;              // Maximum load per model (0.0-1.0)
        size_t max_concurrent_tasks;        // Maximum concurrent tasks per model
        double load_balance_threshold;      // Threshold for triggering load balancing
        bool enable_predictive_scaling;     // Enable predictive model scaling
        double scaling_factor;              // Factor for scaling decisions
        std::chrono::milliseconds update_interval; // Performance update interval
        
        LoadBalancingConfig() : max_model_load(0.8), max_concurrent_tasks(4),
                               load_balance_threshold(0.7), enable_predictive_scaling(true),
                               scaling_factor(1.2), update_interval(std::chrono::milliseconds(1000)) {}
    };

public:
    /**
     * @brief Constructor
     */
    DynamicModelDispatcher(const LoadBalancingConfig& config = LoadBalancingConfig{});
    
    /**
     * @brief Destructor
     */
    ~DynamicModelDispatcher();

    // Core dispatching methods
    
    /**
     * @brief Initialize the dispatcher
     */
    bool initialize();
    
    /**
     * @brief Shutdown the dispatcher
     */
    void shutdown();
    
    /**
     * @brief Dispatch a task to the optimal model
     */
    DispatchResult dispatchTask(const OrchestralAIDirector::OrchestralTask& task);
    
    /**
     * @brief Analyze content to determine routing strategy
     */
    ContentAnalysis analyzeContent(const MultiModalContent& content);
    
    /**
     * @brief Select optimal model for a task
     */
    std::string selectOptimalModel(const OrchestralAIDirector::OrchestralTask& task, 
                                  const ContentAnalysis& analysis);

    // Model management
    
    /**
     * @brief Register a model for dispatching
     */
    bool registerModel(const OrchestralAIDirector::ModelInfo& model_info);
    
    /**
     * @brief Unregister a model
     */
    bool unregisterModel(const std::string& model_id);
    
    /**
     * @brief Update model performance metrics
     */
    void updateModelPerformance(const std::string& model_id, 
                               double processing_time_ms, 
                               double quality_score, 
                               bool task_success);
    
    /**
     * @brief Get model performance metrics
     */
    ModelPerformance getModelPerformance(const std::string& model_id) const;
    
    /**
     * @brief Get all registered models
     */
    std::vector<std::string> getRegisteredModels() const;

    // Load balancing
    
    /**
     * @brief Check if model can accept new task
     */
    bool canAcceptTask(const std::string& model_id) const;
    
    /**
     * @brief Get current system load
     */
    double getSystemLoad() const;
    
    /**
     * @brief Trigger load balancing if needed
     */
    void balanceLoad();
    
    /**
     * @brief Scale model resources based on demand
     */
    void scaleResources();

    // Routing configuration
    
    /**
     * @brief Set routing strategy
     */
    void setRoutingStrategy(RoutingStrategy strategy);
    
    /**
     * @brief Get current routing strategy
     */
    RoutingStrategy getRoutingStrategy() const;
    
    /**
     * @brief Update load balancing configuration
     */
    void updateConfig(const LoadBalancingConfig& config);
    
    /**
     * @brief Get current configuration
     */
    LoadBalancingConfig getConfig() const;

    // Analytics and monitoring
    
    /**
     * @brief Get dispatch statistics
     */
    std::map<std::string, size_t> getDispatchStatistics() const;
    
    /**
     * @brief Get performance analytics
     */
    std::map<std::string, double> getPerformanceAnalytics() const;
    
    /**
     * @brief Get load balancing metrics
     */
    std::map<std::string, double> getLoadBalancingMetrics() const;
    
    /**
     * @brief Generate performance report
     */
    std::string generatePerformanceReport() const;

private:
    // Configuration
    LoadBalancingConfig config_;
    RoutingStrategy routing_strategy_;
    
    // Model management
    std::map<std::string, OrchestralAIDirector::ModelInfo> registered_models_;
    std::map<std::string, ModelPerformance> model_performance_;
    mutable std::mutex models_mutex_;
    
    // Load balancing
    std::atomic<bool> is_running_{false};
    std::thread load_balancer_thread_;
    std::thread performance_monitor_thread_;
    
    // Statistics
    std::map<std::string, size_t> dispatch_counts_;
    std::map<std::string, double> total_processing_times_;
    std::map<std::string, double> total_quality_scores_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    void loadBalancerLoop();
    void performanceMonitorLoop();
    double calculateModelScore(const std::string& model_id, 
                              const OrchestralAIDirector::OrchestralTask& task,
                              const ContentAnalysis& analysis) const;
    std::vector<std::string> getCompatibleModels(const OrchestralAIDirector::OrchestralTask& task) const;
    double calculateContentComplexity(const MultiModalContent& content) const;
    std::string selectByStrategy(const std::vector<std::string>& candidates,
                                const OrchestralAIDirector::OrchestralTask& task,
                                const ContentAnalysis& analysis) const;
    void updateDispatchStatistics(const std::string& model_id, double processing_time, double quality);
};

} // namespace ai
} // namespace asekioml
