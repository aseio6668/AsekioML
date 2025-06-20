#pragma once

#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <functional>

namespace clmodel {
namespace ai {

// Forward declarations
class RealTimeContentPipeline;
class AdaptiveQualityEngine;

/**
 * @brief Stream processing configuration for production environments
 * 
 * Controls all aspects of enterprise streaming infrastructure including
 * throughput, latency, buffering, and resource management parameters.
 */
struct StreamingConfig {
    // Throughput Configuration
    size_t max_concurrent_streams = 100;        ///< Maximum simultaneous processing streams
    size_t buffer_size_mb = 512;               ///< Per-stream buffer size in MB
    size_t max_queue_size = 10000;             ///< Maximum queued requests per stream
    
    // Latency Configuration
    std::chrono::milliseconds target_latency{50};       ///< Target end-to-end latency
    std::chrono::milliseconds max_latency{200};         ///< Maximum acceptable latency
    std::chrono::milliseconds timeout_ms{5000};         ///< Request timeout threshold
    
    // Quality Configuration
    float min_quality_threshold = 0.7f;        ///< Minimum acceptable quality score
    float target_quality = 0.85f;              ///< Target quality for optimization
    bool adaptive_quality_enabled = true;      ///< Enable adaptive quality management
    
    // Resource Management
    size_t max_memory_usage_mb = 8192;         ///< Maximum memory usage limit
    size_t max_cpu_cores = 16;                 ///< Maximum CPU cores to utilize
    float cpu_usage_threshold = 0.8f;          ///< CPU usage warning threshold
    
    // Monitoring Configuration
    std::chrono::seconds metrics_interval{1};           ///< Metrics collection interval
    std::chrono::seconds health_check_interval{5};      ///< Health check frequency
    bool detailed_logging = false;             ///< Enable detailed operation logging
    
    // Auto-scaling Configuration
    bool auto_scaling_enabled = true;          ///< Enable automatic resource scaling
    float scale_up_threshold = 0.8f;           ///< Resource usage threshold for scaling up
    float scale_down_threshold = 0.3f;         ///< Resource usage threshold for scaling down
    size_t min_workers = 4;                    ///< Minimum worker thread count
    size_t max_workers = 32;                   ///< Maximum worker thread count
};

/**
 * @brief Stream processing statistics and performance metrics
 * 
 * Comprehensive metrics collection for monitoring production streaming
 * performance, throughput analysis, and system health assessment.
 */
struct StreamMetrics {
    // Throughput Metrics
    std::atomic<uint64_t> total_requests{0};            ///< Total processed requests
    std::atomic<uint64_t> successful_requests{0};       ///< Successfully completed requests
    std::atomic<uint64_t> failed_requests{0};           ///< Failed request count
    std::atomic<uint64_t> timeout_requests{0};          ///< Timed out requests
    
    // Performance Metrics
    std::atomic<double> average_latency_ms{0.0};        ///< Average processing latency
    std::atomic<double> average_quality{0.0};           ///< Average content quality
    std::atomic<size_t> active_streams{0};              ///< Currently active streams
    std::atomic<size_t> queued_requests{0};             ///< Requests waiting in queue
    
    // Resource Metrics
    std::atomic<double> cpu_usage{0.0};                 ///< Current CPU utilization
    std::atomic<size_t> memory_usage_mb{0};             ///< Current memory usage
    std::atomic<size_t> active_workers{0};              ///< Currently active worker threads
    std::atomic<double> throughput_rps{0.0};            ///< Requests per second
    
    // System Health
    std::atomic<bool> is_healthy{true};                 ///< Overall system health status
    std::chrono::steady_clock::time_point last_update;  ///< Last metrics update time
    std::string last_error;                             ///< Last encountered error message
    
    // Performance Distribution
    std::vector<double> latency_histogram;              ///< Latency distribution buckets
    std::vector<double> quality_histogram;              ///< Quality distribution buckets
    
    /**
     * @brief Calculate success rate percentage
     * @return Success rate as percentage (0.0 to 100.0)
     */
    double getSuccessRate() const {
        uint64_t total = total_requests.load();
        if (total == 0) return 100.0;
        return (static_cast<double>(successful_requests.load()) / total) * 100.0;
    }
    
    /**
     * @brief Calculate error rate percentage
     * @return Error rate as percentage (0.0 to 100.0)
     */
    double getErrorRate() const {
        uint64_t total = total_requests.load();
        if (total == 0) return 0.0;
        return (static_cast<double>(failed_requests.load()) / total) * 100.0;
    }
    
    /**
     * @brief Calculate resource utilization score
     * @return Utilization score (0.0 to 1.0)
     */
    double getResourceUtilization() const {
        return (cpu_usage.load() + static_cast<double>(memory_usage_mb.load()) / 8192.0) / 2.0;
    }
};

/**
 * @brief Stream processing request with content and metadata
 * 
 * Encapsulates all information needed for production streaming including
 * content data, processing parameters, callbacks, and quality requirements.
 */
struct StreamRequest {
    std::string request_id;                     ///< Unique request identifier
    std::string content_type;                   ///< Content type (video, audio, text, multimodal)
    std::vector<uint8_t> content_data;          ///< Raw content data buffer
    std::unordered_map<std::string, float> parameters; ///< Processing parameters
    
    // Quality Requirements
    float quality_target = 0.85f;              ///< Target quality for this request
    std::chrono::milliseconds max_latency{100}; ///< Maximum acceptable processing time
    
    // Callbacks
    std::function<void(const std::string&, const std::vector<uint8_t>&, float)> success_callback;
    std::function<void(const std::string&, const std::string&)> error_callback;
    std::function<void(const std::string&, float)> progress_callback;
    
    // Timing
    std::chrono::steady_clock::time_point submit_time; ///< Request submission timestamp
    std::chrono::steady_clock::time_point start_time;  ///< Processing start timestamp
    
    // Priority and Routing
    int priority = 0;                           ///< Request priority (higher = more priority)
    std::string preferred_model;                ///< Preferred model for processing
    bool requires_gpu = false;                  ///< Whether GPU acceleration is required
};

/**
 * @brief Stream processing result with output and metadata
 * 
 * Contains the complete result of stream processing including generated
 * content, quality metrics, performance data, and processing metadata.
 */
struct StreamResult {
    std::string request_id;                     ///< Corresponding request identifier
    bool success = false;                       ///< Processing success status
    std::vector<uint8_t> output_data;           ///< Generated content output
    float quality_score = 0.0f;                ///< Achieved quality score
    
    // Performance Metrics
    std::chrono::milliseconds processing_time{0}; ///< Total processing duration
    std::chrono::milliseconds queue_time{0};     ///< Time spent in queue
    std::string model_used;                      ///< Model that processed the request
    
    // Error Information
    std::string error_message;                   ///< Error description if failed
    std::string error_code;                      ///< Standardized error code
    
    // Metadata
    std::unordered_map<std::string, float> output_metadata; ///< Additional output information
    std::chrono::steady_clock::time_point completion_time;  ///< Processing completion time
};

/**
 * @brief Enterprise-grade streaming manager for production AI workloads
 * 
 * ProductionStreamingManager provides high-throughput, low-latency streaming
 * infrastructure for coordinating real-time AI content generation at scale.
 * Designed for enterprise deployment with comprehensive monitoring, auto-scaling,
 * fault tolerance, and production-ready reliability features.
 * 
 * Key Features:
 * - High-throughput stream processing with configurable concurrency
 * - Intelligent load balancing and resource management
 * - Real-time quality monitoring and adaptive optimization
 * - Production-grade error handling and fault tolerance
 * - Comprehensive metrics collection and monitoring
 * - Auto-scaling based on load and performance
 * - Enterprise security and compliance features
 * 
 * Integration:
 * - Works seamlessly with RealTimeContentPipeline for content generation
 * - Integrates with AdaptiveQualityEngine for quality optimization
 * - Provides enterprise streaming infrastructure for OrchestralAIDirector
 * 
 * Performance:
 * - Sub-100ms latency for most content types
 * - Handles 100+ concurrent streams with proper resource allocation
 * - Adaptive quality management maintaining target quality thresholds
 * - Automatic scaling from 4 to 32 worker threads based on load
 */
class ProductionStreamingManager {
public:
    /**
     * @brief Constructor with streaming configuration
     * @param config Streaming configuration parameters
     */
    explicit ProductionStreamingManager(const StreamingConfig& config = StreamingConfig{});
    
    /**
     * @brief Destructor ensuring clean shutdown
     */
    ~ProductionStreamingManager();
    
    // Core Stream Management
    
    /**
     * @brief Initialize streaming infrastructure and start worker threads
     * @return True if initialization successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Submit content for streaming processing
     * @param request Stream processing request with content and parameters
     * @return Request ID for tracking, empty string if submission failed
     */
    std::string submitStream(const StreamRequest& request);
    
    /**
     * @brief Submit batch of streams for parallel processing
     * @param requests Vector of stream processing requests
     * @return Vector of request IDs, empty entries for failed submissions
     */
    std::vector<std::string> submitBatchStreams(const std::vector<StreamRequest>& requests);
    
    /**
     * @brief Cancel active stream processing
     * @param request_id ID of stream to cancel
     * @return True if cancellation successful, false if not found
     */
    bool cancelStream(const std::string& request_id);
    
    /**
     * @brief Get current status of stream processing
     * @param request_id ID of stream to check
     * @return Processing status string, "NOT_FOUND" if invalid ID
     */
    std::string getStreamStatus(const std::string& request_id) const;
    
    // Quality and Performance Management
    
    /**
     * @brief Set global quality target for all streams
     * @param quality_target Target quality score (0.0 to 1.0)
     */
    void setQualityTarget(float quality_target);
    
    /**
     * @brief Enable or disable adaptive quality management
     * @param enabled Whether to enable adaptive quality
     */
    void setAdaptiveQuality(bool enabled);
    
    /**
     * @brief Update maximum acceptable latency
     * @param max_latency Maximum latency threshold
     */
    void setMaxLatency(std::chrono::milliseconds max_latency);
    
    /**
     * @brief Trigger manual resource scaling
     * @param target_workers Target number of worker threads
     */
    void scaleResources(size_t target_workers);
    
    // Integration with Content Pipeline
    
    /**
     * @brief Set content pipeline for processing integration
     * @param pipeline Shared pointer to content pipeline
     */
    void setContentPipeline(std::shared_ptr<RealTimeContentPipeline> pipeline);
    
    /**
     * @brief Set quality engine for optimization integration
     * @param engine Shared pointer to quality engine
     */
    void setQualityEngine(std::shared_ptr<AdaptiveQualityEngine> engine);
    
    /**
     * @brief Register custom stream processor for specific content types
     * @param content_type Content type identifier
     * @param processor Processing function
     */
    void registerStreamProcessor(
        const std::string& content_type,
        std::function<StreamResult(const StreamRequest&)> processor
    );
    
    // Monitoring and Analytics
      /**
     * @brief Get current streaming performance metrics
     * @param output Reference to metrics structure to fill
     */
    void getMetrics(StreamMetrics& output) const;
    
    /**
     * @brief Get detailed performance report
     * @return Formatted performance report string
     */
    std::string getPerformanceReport() const;
    
    /**
     * @brief Get system health status
     * @return True if system is healthy, false if issues detected
     */
    bool isHealthy() const;
    
    /**
     * @brief Get currently active stream count
     * @return Number of streams being processed
     */
    size_t getActiveStreamCount() const;
    
    /**
     * @brief Get queued request count
     * @return Number of requests waiting for processing
     */
    size_t getQueuedRequestCount() const;
    
    /**
     * @brief Get current resource utilization
     * @return Resource utilization as percentage (0.0 to 100.0)
     */
    double getResourceUtilization() const;
    
    // Lifecycle Management
    
    /**
     * @brief Gracefully shutdown streaming manager
     * @param timeout_ms Maximum time to wait for shutdown
     * @return True if shutdown completed within timeout
     */
    bool shutdown(std::chrono::milliseconds timeout_ms = std::chrono::milliseconds{5000});
    
    /**
     * @brief Force immediate shutdown (emergency use)
     */
    void forceShutdown();
    
    /**
     * @brief Check if streaming manager is running
     * @return True if actively processing streams
     */
    bool isRunning() const;
    
    // Configuration Management
    
    /**
     * @brief Update streaming configuration at runtime
     * @param config New configuration parameters
     * @return True if update successful, false if invalid configuration
     */
    bool updateConfiguration(const StreamingConfig& config);
    
    /**
     * @brief Get current streaming configuration
     * @return Current configuration copy
     */
    StreamingConfig getConfiguration() const;

private:
    // Internal Types
    struct WorkerThread {
        std::thread thread;
        std::atomic<bool> active{false};
        std::atomic<uint64_t> processed_count{0};
        std::chrono::steady_clock::time_point last_activity;
    };
    
    struct ActiveStream {
        StreamRequest request;
        std::chrono::steady_clock::time_point start_time;
        std::atomic<bool> cancelled{false};
        size_t worker_id;
    };
    
    // Configuration and State
    StreamingConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Thread Management
    std::vector<std::unique_ptr<WorkerThread>> workers_;
    std::atomic<size_t> active_worker_count_{0};
    
    // Request Processing
    std::queue<StreamRequest> request_queue_;
    std::unordered_map<std::string, std::unique_ptr<ActiveStream>> active_streams_;
    std::unordered_map<std::string, std::function<StreamResult(const StreamRequest&)>> processors_;
    
    // Synchronization
    mutable std::mutex queue_mutex_;
    mutable std::mutex streams_mutex_;
    mutable std::mutex metrics_mutex_;
    mutable std::mutex config_mutex_;
    std::condition_variable queue_condition_;
    std::condition_variable shutdown_condition_;
    
    // Metrics and Monitoring
    mutable StreamMetrics metrics_;
    std::thread metrics_thread_;
    std::thread health_monitor_thread_;
    
    // Component Integration
    std::shared_ptr<RealTimeContentPipeline> content_pipeline_;
    std::shared_ptr<AdaptiveQualityEngine> quality_engine_;
    
    // Internal Methods
    void workerThreadFunction(size_t worker_id);
    void metricsCollectionFunction();
    void healthMonitorFunction();
    
    StreamResult processStreamRequest(const StreamRequest& request);
    StreamResult processVideoStream(const StreamRequest& request);
    StreamResult processAudioStream(const StreamRequest& request);
    StreamResult processTextStream(const StreamRequest& request);
    StreamResult processMultimodalStream(const StreamRequest& request);
    
    void updateMetrics();
    void checkSystemHealth();
    void autoScaleWorkers();
    
    bool shouldScaleUp() const;
    bool shouldScaleDown() const;
    void addWorker();
    void removeWorker();
    
    std::string generateRequestId() const;
    void logError(const std::string& error) const;
    void logInfo(const std::string& info) const;
};

} // namespace ai
} // namespace clmodel
