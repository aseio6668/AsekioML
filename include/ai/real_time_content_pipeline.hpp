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

namespace asekioml {
namespace ai {

/**
 * @brief Real-Time Content Pipeline - End-to-end content creation with temporal synchronization
 * 
 * This class provides unified content creation workflows spanning multiple modalities
 * with real-time processing, temporal consistency, and cross-modal alignment.
 */
class RealTimeContentPipeline {
public:
    /**
     * @brief Pipeline processing mode
     */
    enum class ProcessingMode {
        SEQUENTIAL,        // Process stages sequentially
        PARALLEL,          // Process compatible stages in parallel
        STREAMING,         // Real-time streaming processing
        BATCH,             // Batch processing for efficiency
        HYBRID             // Adaptive mode selection
    };

    /**
     * @brief Content generation stage
     */
    struct GenerationStage {
        std::string stage_id;
        std::string stage_name;
        std::vector<std::string> input_modalities;
        std::vector<std::string> output_modalities;
        std::vector<std::string> dependencies;  // Stage dependencies
        double estimated_processing_time_ms;
        double quality_requirement;
        bool supports_streaming;
        bool supports_parallel;
        
        GenerationStage() : estimated_processing_time_ms(100.0), 
                           quality_requirement(0.8),
                           supports_streaming(false), 
                           supports_parallel(false) {}
    };

    /**
     * @brief Pipeline configuration
     */
    struct PipelineConfig {
        ProcessingMode mode;
        double target_latency_ms;
        double quality_threshold;
        size_t max_parallel_stages;
        size_t buffer_size;
        bool enable_temporal_consistency;
        bool enable_cross_modal_alignment;
        bool enable_adaptive_quality;
        bool enable_real_time_optimization;
        std::chrono::milliseconds streaming_chunk_duration;
        double temporal_window_size_s;
        
        PipelineConfig() : mode(ProcessingMode::HYBRID),
                          target_latency_ms(200.0),
                          quality_threshold(0.85),
                          max_parallel_stages(4),
                          buffer_size(1024),
                          enable_temporal_consistency(true),
                          enable_cross_modal_alignment(true),
                          enable_adaptive_quality(true),
                          enable_real_time_optimization(true),
                          streaming_chunk_duration(std::chrono::milliseconds(250)),
                          temporal_window_size_s(2.0) {}
    };

    /**
     * @brief Content generation request
     */
    struct ContentRequest {
        std::string request_id;
        std::string content_type;  // "video", "audio", "text", "multimodal"
        MultiModalContent input_content;
        std::map<std::string, std::string> prompts;
        std::map<std::string, double> quality_requirements;
        std::vector<std::string> required_outputs;
        double max_latency_ms;
        bool enable_streaming;
        std::function<void(const MultiModalContent&, double)> progress_callback;
        std::function<void(const MultiModalContent&, const ContentQualityMetrics&)> completion_callback;
        
        ContentRequest() : max_latency_ms(1000.0), enable_streaming(false) {}
    };

    /**
     * @brief Pipeline execution result
     */
    struct PipelineResult {
        std::string request_id;
        MultiModalContent generated_content;
        ContentQualityMetrics quality_metrics;
        double total_processing_time_ms;
        double actual_latency_ms;
        std::vector<std::string> completed_stages;
        std::map<std::string, double> stage_timings;
        bool success;
        std::string error_message;
        
        PipelineResult() : total_processing_time_ms(0.0),
                          actual_latency_ms(0.0),
                          success(false) {}
    };

    /**
     * @brief Temporal synchronization context
     */
    struct TemporalContext {
        std::vector<double> timestamps;
        std::map<std::string, Tensor> temporal_features;
        std::map<std::string, std::vector<double>> temporal_alignments;
        double base_frame_rate;
        double target_frame_rate;
        double temporal_consistency_score;
        
        TemporalContext() : base_frame_rate(30.0), 
                           target_frame_rate(30.0),
                           temporal_consistency_score(1.0) {}
    };

    /**
     * @brief Cross-modal alignment context
     */
    struct CrossModalContext {
        std::map<std::string, Tensor> modal_embeddings;
        std::map<std::pair<std::string, std::string>, double> alignment_scores;
        std::map<std::string, double> modal_weights;
        Tensor unified_representation;
        double overall_alignment_score;
        
        CrossModalContext() : overall_alignment_score(1.0) {}
    };

public:
    /**
     * @brief Constructor
     */
    RealTimeContentPipeline(const PipelineConfig& config = PipelineConfig{});
    
    /**
     * @brief Destructor
     */
    ~RealTimeContentPipeline();

    // Core pipeline operations
    
    /**
     * @brief Initialize the pipeline
     */
    bool initialize();
    
    /**
     * @brief Shutdown the pipeline
     */
    void shutdown();
    
    /**
     * @brief Set orchestral director reference
     */
    void setOrchestralDirector(std::shared_ptr<OrchestralAIDirector> director);
    
    /**
     * @brief Set model dispatcher reference
     */
    void setModelDispatcher(std::shared_ptr<DynamicModelDispatcher> dispatcher);
    
    /**
     * @brief Generate content from request
     */
    PipelineResult generateContent(const ContentRequest& request);
    
    /**
     * @brief Generate content asynchronously
     */
    std::string generateContentAsync(const ContentRequest& request);
    
    /**
     * @brief Generate streaming content
     */
    std::string startStreamingGeneration(const ContentRequest& request);
    
    /**
     * @brief Process streaming chunk
     */
    MultiModalContent processStreamingChunk(const std::string& stream_id, 
                                           const MultiModalContent& chunk);

    // Pipeline configuration
    
    /**
     * @brief Add generation stage
     */
    bool addGenerationStage(const GenerationStage& stage);
    
    /**
     * @brief Remove generation stage
     */
    bool removeGenerationStage(const std::string& stage_id);
    
    /**
     * @brief Configure pipeline stages
     */
    void configurePipelineStages(const std::vector<GenerationStage>& stages);
    
    /**
     * @brief Set processing mode
     */
    void setProcessingMode(ProcessingMode mode);
    
    /**
     * @brief Update pipeline configuration
     */
    void updateConfig(const PipelineConfig& config);

    // Temporal consistency
    
    /**
     * @brief Ensure temporal consistency
     */
    MultiModalContent ensureTemporalConsistency(const MultiModalContent& content,
                                               const TemporalContext& context);
    
    /**
     * @brief Create temporal context
     */
    TemporalContext createTemporalContext(const MultiModalContent& content);
    
    /**
     * @brief Align temporal features
     */
    MultiModalContent alignTemporalFeatures(const MultiModalContent& content,
                                           const TemporalContext& context);

    // Cross-modal alignment
    
    /**
     * @brief Align cross-modal content
     */
    MultiModalContent alignCrossModalContent(const MultiModalContent& content,
                                            const CrossModalContext& context);
    
    /**
     * @brief Create cross-modal context
     */
    CrossModalContext createCrossModalContext(const MultiModalContent& content);
    
    /**
     * @brief Compute cross-modal alignments
     */
    void computeCrossModalAlignments(CrossModalContext& context);

    // Quality optimization
    
    /**
     * @brief Optimize content quality
     */
    MultiModalContent optimizeContentQuality(const MultiModalContent& content,
                                            const std::map<std::string, double>& quality_requirements);
    
    /**
     * @brief Assess content quality
     */
    ContentQualityMetrics assessContentQuality(const MultiModalContent& content);
    
    /**
     * @brief Adapt quality based on performance
     */
    void adaptiveQualityControl(const std::string& request_id, double current_latency);

    // Status and monitoring
    
    /**
     * @brief Get pipeline status
     */
    std::map<std::string, std::string> getPipelineStatus() const;
    
    /**
     * @brief Get performance metrics
     */
    std::map<std::string, double> getPerformanceMetrics() const;
    
    /**
     * @brief Get generation stages
     */
    std::vector<GenerationStage> getGenerationStages() const;
    
    /**
     * @brief Get active requests
     */
    std::vector<std::string> getActiveRequests() const;
    
    /**
     * @brief Generate performance report
     */
    std::string generatePerformanceReport() const;

private:
    // Configuration
    PipelineConfig config_;
    std::vector<GenerationStage> stages_;
    
    // Component references
    std::shared_ptr<OrchestralAIDirector> orchestral_director_;
    std::shared_ptr<DynamicModelDispatcher> model_dispatcher_;
    
    // Processing infrastructure
    std::atomic<bool> is_running_{false};
    std::thread processing_thread_;
    std::thread streaming_thread_;
    
    // Request management
    std::queue<ContentRequest> request_queue_;
    std::map<std::string, PipelineResult> active_requests_;
    std::map<std::string, std::queue<MultiModalContent>> streaming_buffers_;
    mutable std::mutex request_mutex_;
    mutable std::mutex streaming_mutex_;
    
    // Performance tracking
    std::map<std::string, double> stage_performance_;
    std::map<std::string, size_t> stage_execution_counts_;
    std::map<std::string, double> quality_history_;
    mutable std::mutex performance_mutex_;
    
    // Internal methods
    void processingLoop();
    void streamingLoop();
    PipelineResult executeSequentialPipeline(const ContentRequest& request);
    PipelineResult executeParallelPipeline(const ContentRequest& request);
    PipelineResult executeStreamingPipeline(const ContentRequest& request);
    PipelineResult executeBatchPipeline(const ContentRequest& request);
    PipelineResult executeHybridPipeline(const ContentRequest& request);
    
    MultiModalContent executeStage(const GenerationStage& stage, 
                                  const MultiModalContent& input,
                                  const ContentRequest& request);
    
    std::vector<GenerationStage> resolveStageDependencies(const std::vector<GenerationStage>& stages);
    bool canExecuteStageInParallel(const GenerationStage& stage, 
                                  const std::vector<std::string>& completed_stages);
    
    void updateStagePerformance(const std::string& stage_id, double processing_time);
    void updateQualityHistory(const std::string& request_id, double quality);
    
    std::string generateRequestId();
    ProcessingMode selectOptimalMode(const ContentRequest& request);
    void optimizePipelineForLatency(const ContentRequest& request);
    void optimizePipelineForQuality(const ContentRequest& request);
};

} // namespace ai
} // namespace asekioml
