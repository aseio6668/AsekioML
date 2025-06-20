#pragma once

#include "tensor.hpp"
#include "video_audio_text_fusion.hpp"
#include "multimodal_attention.hpp"
#include "cross_modal_guidance.hpp"
#include "audio_visual_sync.hpp"
#include "orchestral_ai_workflow.hpp"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace clmodel {
namespace ai {

// Forward declarations
class DynamicModelDispatcher;
class RealTimeContentPipeline;
class AdaptiveQualityEngine;
class ProductionStreamingManager;

/**
 * @brief Orchestral AI Director - Master coordination system for multi-model workflows
 * 
 * This class serves as the central coordination hub for the orchestral AI system,
 * managing multiple specialized AI models and coordinating complex multi-modal workflows.
 */
class OrchestralAIDirector {
public:
    /**
     * @brief Configuration for the orchestral AI system
     */
    struct OrchestralConfig {
        // Resource management
        size_t max_concurrent_models = 8;
        size_t max_memory_mb = 4096;
        size_t thread_pool_size = std::thread::hardware_concurrency();
        
        // Performance settings
        double target_latency_ms = 100.0;
        double quality_threshold = 0.8;
        bool enable_adaptive_quality = true;
        bool enable_real_time_optimization = true;
        
        // Model coordination
        bool enable_model_caching = true;
        bool enable_dynamic_loading = true;
        size_t model_cache_size = 16;
        
        // Streaming settings
        size_t buffer_size = 1024;
        double streaming_chunk_duration_s = 0.5;
        bool enable_low_latency_mode = false;
        
        // Quality and monitoring
        bool enable_performance_monitoring = true;
        bool enable_quality_prediction = true;
        double performance_sample_interval_s = 1.0;
    };

    /**
     * @brief Information about a registered AI model
     */
    struct ModelInfo {
        std::string model_id;
        std::string model_type;  // "video", "audio", "text", "fusion", etc.
        std::string version;
        
        // Capabilities
        std::vector<std::string> supported_modalities;
        std::vector<std::string> supported_tasks;
        
        // Performance characteristics
        double avg_processing_time_ms = 0.0;
        double quality_score = 1.0;
        size_t memory_usage_mb = 0;
        
        // State
        bool is_loaded = false;
        bool is_available = true;
        std::chrono::steady_clock::time_point last_used;
        
        // Custom parameters
        std::map<std::string, double> parameters;
    };

    /**
     * @brief Task request for the orchestral system
     */
    struct OrchestralTask {
        std::string task_id;
        std::string task_type;  // "generate", "enhance", "analyze", etc.
        
        // Input data
        MultiModalContent input_content;
        std::map<std::string, std::string> text_prompts;
        
        // Requirements
        std::vector<std::string> required_modalities;
        double quality_requirement = 0.8;
        double max_latency_ms = 1000.0;
        
        // Processing options
        bool enable_streaming = false;
        bool enable_real_time = false;
        FusionConfig fusion_config;
        
        // Callback for results
        std::function<void(const MultiModalContent&, const ContentQualityMetrics&)> result_callback;
        std::function<void(const std::string&)> error_callback;
        
        // Timing
        std::chrono::steady_clock::time_point submitted_time;
        std::chrono::steady_clock::time_point started_time;
    };

    /**
     * @brief System performance metrics
     */
    struct SystemMetrics {
        // Throughput
        double tasks_per_second = 0.0;
        double avg_task_latency_ms = 0.0;
        double avg_queue_wait_time_ms = 0.0;
        
        // Resource utilization
        double cpu_utilization = 0.0;
        double memory_utilization = 0.0;
        double gpu_utilization = 0.0;
        
        // Quality metrics
        double avg_output_quality = 0.0;
        double quality_consistency = 0.0;
        
        // Model performance
        size_t active_models = 0;
        size_t cached_models = 0;
        double model_switch_frequency = 0.0;
        
        // Error rates
        double error_rate = 0.0;
        double timeout_rate = 0.0;
        
        std::chrono::steady_clock::time_point last_updated;
    };

public:
    /**
     * @brief Constructor
     */
    OrchestralAIDirector(const OrchestralConfig& config = OrchestralConfig{});
    
    /**
     * @brief Destructor
     */
    ~OrchestralAIDirector();

    // Core orchestration methods
    
    /**
     * @brief Initialize the orchestral system
     */
    bool initialize();
    
    /**
     * @brief Shutdown the orchestral system
     */
    void shutdown();
    
    /**
     * @brief Submit a task for orchestral processing
     */
    std::string submitTask(const OrchestralTask& task);
    
    /**
     * @brief Cancel a submitted task
     */
    bool cancelTask(const std::string& task_id);
    
    /**
     * @brief Get task status and progress
     */
    bool getTaskStatus(const std::string& task_id, std::string& status, double& progress);

    // Model management
    
    /**
     * @brief Register a new AI model with the system
     */
    bool registerModel(const ModelInfo& model_info);
    
    /**
     * @brief Unregister a model from the system
     */
    bool unregisterModel(const std::string& model_id);
    
    /**
     * @brief Get information about registered models
     */
    std::vector<ModelInfo> getRegisteredModels() const;
    
    /**
     * @brief Update model performance metrics
     */
    void updateModelMetrics(const std::string& model_id, double processing_time_ms, double quality_score);

    // Workflow coordination
    
    /**
     * @brief Create a custom workflow template
     */
    bool createWorkflowTemplate(const std::string& template_name, const std::vector<std::string>& stages);
    
    /**
     * @brief Execute a workflow template
     */
    std::string executeWorkflow(const std::string& template_name, const MultiModalContent& input);
    
    /**
     * @brief Get available workflow templates
     */
    std::vector<std::string> getWorkflowTemplates() const;

    // System monitoring and control
    
    /**
     * @brief Get current system performance metrics
     */
    SystemMetrics getSystemMetrics() const;
    
    /**
     * @brief Get system health status
     */
    bool isSystemHealthy() const;
    
    /**
     * @brief Update system configuration
     */
    void updateConfiguration(const OrchestralConfig& new_config);
    
    /**
     * @brief Get current configuration
     */
    OrchestralConfig getConfiguration() const;

    // Advanced coordination features
    
    /**
     * @brief Enable or disable adaptive quality control
     */
    void setAdaptiveQualityEnabled(bool enabled);
    
    /**
     * @brief Set quality target for adaptive system
     */
    void setQualityTarget(double quality_target);
    
    /**
     * @brief Enable or disable real-time optimization
     */
    void setRealTimeOptimizationEnabled(bool enabled);
    
    /**
     * @brief Force model cache refresh
     */
    void refreshModelCache();

    // Component access (for integration)
    
    /**
     * @brief Get the model dispatcher component
     */
    std::shared_ptr<DynamicModelDispatcher> getModelDispatcher() const;
    
    /**
     * @brief Get the content pipeline component
     */
    std::shared_ptr<RealTimeContentPipeline> getContentPipeline() const;
    
    /**
     * @brief Get the quality engine component
     */
    std::shared_ptr<AdaptiveQualityEngine> getQualityEngine() const;
    
    /**
     * @brief Get the streaming manager component
     */
    std::shared_ptr<ProductionStreamingManager> getStreamingManager() const;

private:
    // Internal implementation
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Configuration
    OrchestralConfig config_;
    
    // Core components
    std::shared_ptr<DynamicModelDispatcher> model_dispatcher_;
    std::shared_ptr<RealTimeContentPipeline> content_pipeline_;
    std::shared_ptr<AdaptiveQualityEngine> quality_engine_;
    std::shared_ptr<ProductionStreamingManager> streaming_manager_;
    
    // Task management
    std::atomic<bool> is_running_{false};
    std::atomic<size_t> next_task_id_{0};
    
    // Thread management
    std::vector<std::thread> worker_threads_;
    std::thread monitoring_thread_;
    
    // Task queue and processing
    std::queue<OrchestralTask> task_queue_;
    std::map<std::string, OrchestralTask> active_tasks_;
    std::mutex task_mutex_;
    std::condition_variable task_condition_;
    
    // Model registry
    std::map<std::string, ModelInfo> registered_models_;
    std::mutex model_mutex_;
    
    // Performance monitoring
    SystemMetrics current_metrics_;
    std::mutex metrics_mutex_;
    
    // Internal methods
    void workerThreadFunction();
    void monitoringThreadFunction();
    void processTask(OrchestralTask& task);
    void updateSystemMetrics();
    std::string generateTaskId();
    std::vector<std::string> selectOptimalModels(const OrchestralTask& task);
    void optimizeSystemPerformance();
    void handleTaskError(const std::string& task_id, const std::string& error_message);
    void handleTaskCompletion(const std::string& task_id, const MultiModalContent& result, const ContentQualityMetrics& quality);
};

} // namespace ai
} // namespace clmodel
