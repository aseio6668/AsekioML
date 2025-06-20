#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <chrono>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace clmodel {
namespace ai {

// Forward declarations
class ModelRegistry;
class PipelineBuilder;
class ResourceScheduler;
class QualityOrchestrator;

/**
 * @brief Represents the different modalities supported by the orchestral AI system
 */
enum class OrchestralModality {
    TEXT,
    AUDIO,
    IMAGE,
    VIDEO,
    MULTIMODAL
};

/**
 * @brief Represents different AI model types that can be orchestrated
 */
enum class ModelType {
    TEXT_ENCODER,
    AUDIO_ENCODER,
    IMAGE_ENCODER,
    VIDEO_PROCESSOR,
    CROSS_MODAL_ATTENTION,
    FUSION_NETWORK,
    GENERATION_MODEL,
    CLASSIFICATION_MODEL
};

/**
 * @brief Processing priority levels for resource scheduling
 */
enum class ProcessingPriority {
    LOW = 1,
    NORMAL = 2,
    HIGH = 3,
    CRITICAL = 4
};

/**
 * @brief Performance metrics for quality orchestration
 */
struct PerformanceMetrics {
    double processing_time_ms;
    double memory_usage_mb;
    double quality_score;
    double confidence_score;
    std::string error_message;
    
    PerformanceMetrics() : processing_time_ms(0.0), memory_usage_mb(0.0), 
                          quality_score(0.0), confidence_score(0.0) {}
};

/**
 * @brief Configuration for a processing task
 */
struct TaskConfig {
    std::string task_id;    OrchestralModality input_modality;
    OrchestralModality output_modality;
    ModelType preferred_model;
    ProcessingPriority priority;
    std::map<std::string, std::string> parameters;
    
    TaskConfig(const std::string& id = "") : task_id(id),        input_modality(OrchestralModality::MULTIMODAL), 
        output_modality(OrchestralModality::MULTIMODAL),
        preferred_model(ModelType::FUSION_NETWORK),
        priority(ProcessingPriority::NORMAL) {}
};

/**
 * @brief Base class for all AI models in the orchestral system
 */
class BaseModel {
public:
    BaseModel(const std::string& name, ModelType type) 
        : model_name_(name), model_type_(type), is_loaded_(false) {}
    virtual ~BaseModel() = default;
    
    virtual bool load() = 0;
    virtual bool unload() = 0;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual PerformanceMetrics benchmark(const Tensor& input);
    
    const std::string& getName() const { return model_name_; }
    ModelType getType() const { return model_type_; }
    bool isLoaded() const { return is_loaded_; }
    
protected:
    std::string model_name_;
    ModelType model_type_;
    bool is_loaded_;
};

/**
 * @brief Registry for managing AI models with dynamic loading and selection
 */
class ModelRegistry {
public:
    ModelRegistry();
    ~ModelRegistry();
    
    // Model registration and management
    bool registerModel(std::shared_ptr<BaseModel> model);
    bool unregisterModel(const std::string& model_name);
    std::shared_ptr<BaseModel> getModel(const std::string& model_name);
    std::shared_ptr<BaseModel> selectBestModel(ModelType type, const Tensor& input);
    
    // Model discovery and information
    std::vector<std::string> getAvailableModels() const;
    std::vector<std::string> getModelsByType(ModelType type) const;
    PerformanceMetrics getModelPerformance(const std::string& model_name) const;
    
    // Resource management
    void preloadModels(const std::vector<std::string>& model_names);
    void unloadUnusedModels();
    size_t getLoadedModelCount() const;
    
private:
    std::map<std::string, std::shared_ptr<BaseModel>> models_;
    std::map<std::string, PerformanceMetrics> performance_cache_;
    mutable std::mutex registry_mutex_;
    
    std::shared_ptr<BaseModel> selectByPerformance(const std::vector<std::shared_ptr<BaseModel>>& candidates, const Tensor& input);
};

/**
 * @brief Processing task for the workflow pipeline
 */
class ProcessingTask {
public:
    ProcessingTask(const TaskConfig& config, const Tensor& input);
    ~ProcessingTask();
    
    // Task execution
    bool execute(ModelRegistry& registry);
    bool isComplete() const { return is_complete_; }
    bool hasError() const { return has_error_; }
    
    // Results and metrics
    const Tensor& getResult() const { return result_; }
    const PerformanceMetrics& getMetrics() const { return metrics_; }
    const TaskConfig& getConfig() const { return config_; }
    
    // Task management
    void cancel();
    bool isCancelled() const { return is_cancelled_; }
    
private:
    TaskConfig config_;
    Tensor input_data_;
    Tensor result_;
    PerformanceMetrics metrics_;
    bool is_complete_;
    bool has_error_;
    bool is_cancelled_;
    std::chrono::steady_clock::time_point start_time_;
};

/**
 * @brief Builder for creating configurable multi-modal processing pipelines
 */
class PipelineBuilder {
public:
    PipelineBuilder();
    ~PipelineBuilder();
    
    // Pipeline construction
    PipelineBuilder& addStage(const std::string& stage_name, const TaskConfig& config);
    PipelineBuilder& addDependency(const std::string& from_stage, const std::string& to_stage);
    PipelineBuilder& setParallelExecution(bool enable);
    PipelineBuilder& setQualityThreshold(double threshold);
    
    // Pipeline execution
    bool build();
    bool execute(const std::map<std::string, Tensor>& inputs, ModelRegistry& registry);
    std::map<std::string, Tensor> getResults() const;
    std::map<std::string, PerformanceMetrics> getStageMetrics() const;
    
    // Pipeline management
    void reset();
    bool isBuilt() const { return is_built_; }
    size_t getStageCount() const { return stages_.size(); }
    
private:
    struct PipelineStage {
        std::string name;
        TaskConfig config;
        std::vector<std::string> dependencies;
        std::shared_ptr<ProcessingTask> task;
        bool is_ready;
        bool is_complete;
    };
    
    std::vector<PipelineStage> stages_;
    std::map<std::string, size_t> stage_index_;
    bool parallel_execution_;
    double quality_threshold_;
    bool is_built_;
    
    bool validateDependencies() const;
    std::vector<size_t> getExecutionOrder() const;
    bool executeStage(size_t stage_index, ModelRegistry& registry);
};

/**
 * @brief Resource scheduler for efficient allocation of compute resources
 */
class ResourceScheduler {
public:
    ResourceScheduler(size_t max_concurrent_tasks = 4);
    ~ResourceScheduler();
    
    // Task scheduling
    bool scheduleTask(std::shared_ptr<ProcessingTask> task);
    void setPriority(const std::string& task_id, ProcessingPriority priority);
    bool cancelTask(const std::string& task_id);
    
    // Resource management
    void setMaxConcurrentTasks(size_t max_tasks);
    size_t getActiveTaskCount() const;
    size_t getQueuedTaskCount() const;
    double getCPUUsage() const;
    double getMemoryUsage() const;
    
    // Scheduler control
    void start();
    void stop();
    void pauseScheduling();
    void resumeScheduling();
    bool isRunning() const { return is_running_; }
    
private:
    struct ScheduledTask {
        std::shared_ptr<ProcessingTask> task;
        ProcessingPriority priority;
        std::chrono::steady_clock::time_point submit_time;
        
        bool operator<(const ScheduledTask& other) const {
            if (priority != other.priority) {
                return priority < other.priority; // Higher priority = larger value
            }
            return submit_time > other.submit_time; // Earlier submit time = higher priority
        }
    };
    
    std::priority_queue<ScheduledTask> task_queue_;
    std::vector<std::thread> worker_threads_;
    std::map<std::string, std::shared_ptr<ProcessingTask>> active_tasks_;
    
    size_t max_concurrent_tasks_;
    bool is_running_;
    bool is_paused_;
    mutable std::mutex scheduler_mutex_;
    std::condition_variable worker_condition_;
    std::condition_variable completion_condition_;
    
    void workerLoop();
    std::shared_ptr<ProcessingTask> getNextTask();
    void updateResourceMetrics();
};

/**
 * @brief Quality orchestrator for adaptive performance monitoring and optimization
 */
class QualityOrchestrator {
public:
    QualityOrchestrator();
    ~QualityOrchestrator();
    
    // Quality monitoring
    void recordMetrics(const std::string& task_id, const PerformanceMetrics& metrics);
    double getAverageQuality(const std::string& model_name) const;
    double getAverageProcessingTime(const std::string& model_name) const;
    bool isPerformanceAcceptable(const PerformanceMetrics& metrics) const;
    
    // Adaptive optimization
    void setQualityThresholds(double min_quality, double max_processing_time);
    std::string recommendModel(ModelType type, const std::vector<std::string>& available_models) const;
    void adaptResourceAllocation(ResourceScheduler& scheduler) const;
    
    // Quality reporting
    std::map<std::string, PerformanceMetrics> getModelSummary() const;
    std::vector<std::string> getUnderperformingModels() const;
    void generateQualityReport(const std::string& output_path) const;
    
private:
    struct QualityHistory {
        std::vector<PerformanceMetrics> metrics_history;
        double average_quality;
        double average_processing_time;
        size_t success_count;
        size_t failure_count;
    };
    
    std::map<std::string, QualityHistory> model_history_;
    double min_quality_threshold_;
    double max_processing_time_ms_;
    mutable std::mutex quality_mutex_;
    
    void updateModelHistory(const std::string& model_name, const PerformanceMetrics& metrics);
    double calculateTrend(const std::vector<double>& values) const;
};

/**
 * @brief Main workflow manager for orchestrating multi-modal AI processing
 */
class WorkflowManager {
public:
    WorkflowManager();
    ~WorkflowManager();
    
    // Workflow orchestration
    bool executeWorkflow(const std::string& workflow_name, 
                        const std::map<std::string, Tensor>& inputs);
    bool registerWorkflow(const std::string& workflow_name, 
                         std::shared_ptr<PipelineBuilder> pipeline);
    
    // Component access
    ModelRegistry& getModelRegistry() { return model_registry_; }
    ResourceScheduler& getResourceScheduler() { return resource_scheduler_; }
    QualityOrchestrator& getQualityOrchestrator() { return quality_orchestrator_; }
    
    // Workflow management
    std::vector<std::string> getAvailableWorkflows() const;
    std::map<std::string, Tensor> getWorkflowResults(const std::string& workflow_name) const;
    PerformanceMetrics getWorkflowMetrics(const std::string& workflow_name) const;
    
    // System control
    void initialize();
    void shutdown();
    bool isInitialized() const { return is_initialized_; }
    
private:
    ModelRegistry model_registry_;
    ResourceScheduler resource_scheduler_;
    QualityOrchestrator quality_orchestrator_;
    
    std::map<std::string, std::shared_ptr<PipelineBuilder>> workflows_;
    std::map<std::string, std::map<std::string, Tensor>> workflow_results_;
    std::map<std::string, PerformanceMetrics> workflow_metrics_;
    
    bool is_initialized_;
    mutable std::mutex workflow_mutex_;
    
    void setupDefaultModels();
    void setupDefaultWorkflows();
};

} // namespace ai
} // namespace clmodel
