#pragma once

#include "tensor.hpp"
#include "ai/orchestral_ai_workflow.hpp"
#include "ai/cross_modal_guidance.hpp"
#include "ai/audio_visual_sync.hpp"
#include "ai/advanced_frame_interpolation.hpp"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <chrono>
#include <queue>

namespace asekioml {
namespace ai {

// Forward declarations
class VideoAudioTextProcessor;
class ContentGenerationEngine;
class QualityAssessmentSystem;
class StreamingFusionManager;

/**
 * @brief Types of fusion strategies for multi-modal content
 */
enum class VideoAudioTextFusionStrategy {
    EARLY_FUSION,           // Fusion at feature level
    LATE_FUSION,            // Fusion at decision level
    ATTENTION_FUSION,       // Attention-weighted fusion
    HIERARCHICAL_FUSION,    // Multi-level hierarchical fusion
    ADAPTIVE_FUSION,        // Adaptive strategy selection
    TEMPORAL_FUSION         // Temporal sequence fusion
};

/**
 * @brief Quality metrics for multi-modal content
 */
struct ContentQualityMetrics {
    double overall_quality;         // Overall content quality (0-1)
    double visual_quality;          // Visual component quality (0-1)
    double audio_quality;           // Audio component quality (0-1)
    double text_coherence;          // Text coherence score (0-1)
    double temporal_consistency;    // Temporal alignment quality (0-1)
    double semantic_alignment;      // Cross-modal semantic consistency (0-1)
    double computational_efficiency; // Processing efficiency score (0-1)
    std::string generation_mode;    // Mode used for generation
    
    ContentQualityMetrics() : overall_quality(0.0), visual_quality(0.0),
                             audio_quality(0.0), text_coherence(0.0),
                             temporal_consistency(0.0), semantic_alignment(0.0),
                             computational_efficiency(0.0) {}
};

/**
 * @brief Configuration for multi-modal fusion operations
 */
struct FusionConfig {
    VideoAudioTextFusionStrategy strategy;
    double fusion_weight_video;     // Weight for video modality (0-1)
    double fusion_weight_audio;     // Weight for audio modality (0-1)
    double fusion_weight_text;      // Weight for text modality (0-1)
    double temporal_window_size;    // Size of temporal context window (seconds)
    bool enable_real_time;          // Enable real-time processing
    bool enable_quality_feedback;   // Enable quality-based adaptation
    size_t max_sequence_length;     // Maximum sequence length for processing
    std::map<std::string, double> custom_weights; // Custom modality weights
    
    FusionConfig() : strategy(VideoAudioTextFusionStrategy::ATTENTION_FUSION),
                    fusion_weight_video(0.4), fusion_weight_audio(0.3), fusion_weight_text(0.3),
                    temporal_window_size(2.0), enable_real_time(true),
                    enable_quality_feedback(true), max_sequence_length(1000) {}
};

/**
 * @brief Multi-modal content representation
 */
struct MultiModalContent {
    Tensor video_features;          // Video feature representation
    Tensor audio_features;          // Audio feature representation
    Tensor text_features;           // Text feature representation
    std::vector<double> timestamps; // Temporal alignment information
    std::map<std::string, std::string> metadata; // Additional metadata
    ContentQualityMetrics quality;  // Content quality assessment
    
    bool has_video() const { return video_features.size() > 0; }
    bool has_audio() const { return audio_features.size() > 0; }
    bool has_text() const { return text_features.size() > 0; }
    size_t get_sequence_length() const { return timestamps.size(); }
};

/**
 * @brief Streaming data chunk for real-time processing
 */
struct StreamingChunk {
    MultiModalContent content;
    std::chrono::high_resolution_clock::time_point timestamp;
    size_t chunk_id;
    bool is_final_chunk;
    
    StreamingChunk() : chunk_id(0), is_final_chunk(false) {}
};

/**
 * @brief Main multi-modal fusion pipeline for video-audio-text integration
 */
class MultiModalFusionPipeline {
public:
    MultiModalFusionPipeline(const FusionConfig& config = FusionConfig());
    ~MultiModalFusionPipeline();
    
    // Core fusion operations
    MultiModalContent fusePipeline(const MultiModalContent& input);
    std::vector<MultiModalContent> fuseBatch(const std::vector<MultiModalContent>& batch);
    MultiModalContent fuseStreaming(const StreamingChunk& chunk);
    
    // Fusion strategy methods
    MultiModalContent applyEarlyFusion(const MultiModalContent& input);
    MultiModalContent applyLateFusion(const MultiModalContent& input);
    MultiModalContent applyAttentionFusion(const MultiModalContent& input);
    MultiModalContent applyHierarchicalFusion(const MultiModalContent& input);
    MultiModalContent applyTemporalFusion(const MultiModalContent& input);
    
    // Configuration and adaptation
    void setFusionConfig(const FusionConfig& config) { config_ = config; }
    void adaptToQuality(const ContentQualityMetrics& quality);
    const FusionConfig& getFusionConfig() const { return config_; }
    
    // Pipeline management
    bool initializePipeline();
    void optimizePipeline();
    void resetPipeline();
    std::map<std::string, double> getPipelineStatistics() const;
      // Component access
    VideoAudioTextProcessor& getProcessor() { return *processor_; }
    ContentGenerationEngine& getGenerator() { return *generator_; }
    QualityAssessmentSystem& getQualitySystem() { return *quality_system_; }
    StreamingFusionManager& getStreamingManager() { return *streaming_manager_; }
    
private:
    FusionConfig config_;
    std::unique_ptr<VideoAudioTextProcessor> processor_;
    std::unique_ptr<ContentGenerationEngine> generator_;
    std::unique_ptr<QualityAssessmentSystem> quality_system_;
    std::unique_ptr<StreamingFusionManager> streaming_manager_;
    std::unique_ptr<AdvancedCrossModalAttention> fusion_attention_;
    
    std::map<std::string, double> pipeline_statistics_;
    std::vector<ContentQualityMetrics> quality_history_;
    bool is_initialized_;
    
    void setupComponents();
    Tensor computeFusionWeights(const MultiModalContent& input) const;
    MultiModalContent alignTemporalFeatures(const MultiModalContent& input) const;
    void updateQualityHistory(const ContentQualityMetrics& quality);
};

/**
 * @brief Specialized processor for video-audio-text synchronization and processing
 */
class VideoAudioTextProcessor {
public:
    VideoAudioTextProcessor();
    ~VideoAudioTextProcessor();
    
    // Core processing methods
    MultiModalContent processContent(const MultiModalContent& input);
    MultiModalContent synchronizeModalities(const MultiModalContent& input);
    MultiModalContent extractFeatures(const MultiModalContent& raw_input);
    
    // Individual modality processing
    Tensor processVideoStream(const Tensor& video_data, const std::vector<double>& timestamps);
    Tensor processAudioStream(const Tensor& audio_data, const std::vector<double>& timestamps);
    Tensor processTextStream(const Tensor& text_data, const std::vector<double>& timestamps);
    
    // Temporal alignment
    std::vector<double> computeOptimalAlignment(const MultiModalContent& content);
    MultiModalContent applyTemporalAlignment(const MultiModalContent& content, 
                                           const std::vector<double>& alignment);
    double computeTemporalConsistency(const MultiModalContent& content) const;
    
    // Quality assessment
    ContentQualityMetrics assessContentQuality(const MultiModalContent& content) const;
    std::map<std::string, double> analyzeModalityContributions(const MultiModalContent& content) const;
    
    // Configuration
    void setTemporalWindow(double window_size) { temporal_window_size_ = window_size; }
    void enableRealTimeMode(bool enable) { real_time_mode_ = enable; }
    
private:
    double temporal_window_size_;
    bool real_time_mode_;
    std::unique_ptr<AudioVisualSyncPipeline> av_sync_;
    std::unique_ptr<AdvancedFrameInterpolator> frame_interpolator_;
    
    Tensor interpolateFeatures(const Tensor& features, const std::vector<double>& source_times,
                              const std::vector<double>& target_times) const;
    std::vector<double> detectTemporalKeypoints(const MultiModalContent& content) const;
};

/**
 * @brief Advanced content generation engine with cross-modal conditioning
 */
class ContentGenerationEngine {
public:
    ContentGenerationEngine();
    ~ContentGenerationEngine();
    
    // Content generation methods
    MultiModalContent generateContent(const std::string& prompt, const FusionConfig& config);
    MultiModalContent generateFromTemplate(const MultiModalContent& template_content,
                                          const std::string& modifications);
    MultiModalContent enhanceContent(const MultiModalContent& input, 
                                   const std::string& enhancement_type);
    
    // Specialized generation
    Tensor generateVideo(const std::string& description, const Tensor& conditioning_audio = Tensor(),
                        const Tensor& conditioning_text = Tensor());
    Tensor generateAudio(const std::string& description, const Tensor& conditioning_video = Tensor(),
                        const Tensor& conditioning_text = Tensor());
    Tensor generateText(const std::string& prompt, const Tensor& conditioning_video = Tensor(),
                       const Tensor& conditioning_audio = Tensor());
    
    // Cross-modal conditioning
    MultiModalContent applyCrossModalConditioning(const MultiModalContent& content,
                                                 const std::map<std::string, Tensor>& conditions);
    std::map<std::string, double> computeConditioningWeights(const MultiModalContent& content) const;
    
    // Style and control
    void setGenerationStyle(const std::string& style) { generation_style_ = style; }
    void setQualityTarget(double target) { quality_target_ = target; }
    std::vector<std::string> getAvailableStyles() const;
    
    // Generation monitoring
    ContentQualityMetrics getLastGenerationQuality() const { return last_generation_quality_; }
    std::map<std::string, double> getGenerationStatistics() const;
    
private:
    std::string generation_style_;
    double quality_target_;
    ContentQualityMetrics last_generation_quality_;
    std::unique_ptr<CrossModalConditioner> conditioner_;
    std::map<std::string, double> generation_statistics_;
    
    MultiModalContent applyStyleTransfer(const MultiModalContent& content, 
                                       const std::string& target_style) const;
    void optimizeGenerationParameters(const ContentQualityMetrics& quality);
    Tensor generateWithGuidance(const std::string& modality, const std::string& prompt,
                               const std::map<std::string, Tensor>& guidance) const;
};

/**
 * @brief Quality assessment system for multi-modal content evaluation
 */
class QualityAssessmentSystem {
public:
    QualityAssessmentSystem();
    ~QualityAssessmentSystem();
    
    // Quality assessment methods
    ContentQualityMetrics assessContent(const MultiModalContent& content);
    ContentQualityMetrics compareContent(const MultiModalContent& content1,
                                       const MultiModalContent& content2);
    std::map<std::string, double> analyzeQualityTrends(
        const std::vector<ContentQualityMetrics>& history) const;
    
    // Individual modality assessment
    double assessVideoQuality(const Tensor& video_features) const;
    double assessAudioQuality(const Tensor& audio_features) const;
    double assessTextCoherence(const Tensor& text_features) const;
    double assessTemporalConsistency(const MultiModalContent& content) const;
    double assessSemanticAlignment(const MultiModalContent& content) const;
    
    // Adaptive thresholds
    void setQualityThresholds(const std::map<std::string, double>& thresholds);
    void adaptThresholds(const std::vector<ContentQualityMetrics>& feedback);
    std::map<std::string, double> getQualityThresholds() const { return quality_thresholds_; }
    
    // Quality improvement suggestions
    std::vector<std::string> suggestImprovements(const ContentQualityMetrics& quality) const;
    std::map<std::string, double> computeImprovementPriorities(
        const ContentQualityMetrics& quality) const;
    
private:
    std::map<std::string, double> quality_thresholds_;
    std::vector<ContentQualityMetrics> assessment_history_;
    
    double computeOverallQuality(const ContentQualityMetrics& individual_scores) const;
    std::map<std::string, double> analyzeQualityCorrelations(
        const std::vector<ContentQualityMetrics>& history) const;
};

/**
 * @brief Streaming fusion manager for real-time multi-modal processing
 */
class StreamingFusionManager {
public:
    StreamingFusionManager(size_t buffer_size = 1000);
    ~StreamingFusionManager();
    
    // Streaming operations
    bool addStreamingChunk(const StreamingChunk& chunk);
    std::vector<MultiModalContent> processAvailableChunks();
    void flushBuffer();
    
    // Buffer management
    void setBufferSize(size_t size) { max_buffer_size_ = size; }
    size_t getBufferSize() const { return chunk_buffer_.size(); }
    bool isBufferFull() const { return chunk_buffer_.size() >= max_buffer_size_; }
    
    // Real-time processing
    void enableRealTimeMode(bool enable) { real_time_mode_ = enable; }
    void setLatencyTarget(double target_ms) { latency_target_ms_ = target_ms; }
    double getCurrentLatency() const;
    
    // Statistics
    std::map<std::string, double> getStreamingStatistics() const;
    void resetStatistics();
    
private:
    std::queue<StreamingChunk> chunk_buffer_;
    size_t max_buffer_size_;
    bool real_time_mode_;
    double latency_target_ms_;
    
    std::chrono::high_resolution_clock::time_point last_process_time_;
    std::map<std::string, double> streaming_statistics_;
    
    bool shouldProcessChunk(const StreamingChunk& chunk) const;
    std::vector<StreamingChunk> getProcessableChunks();
    void updateStreamingStatistics();
};

/**
 * @brief Utility functions for multi-modal fusion operations
 */
namespace FusionUtils {
    
    // Content creation utilities
    MultiModalContent createEmptyContent(size_t sequence_length = 0);
    MultiModalContent mergeContents(const std::vector<MultiModalContent>& contents);
    MultiModalContent extractTimeWindow(const MultiModalContent& content, 
                                       double start_time, double end_time);
    
    // Feature manipulation
    Tensor normalizeFeatures(const Tensor& features);
    Tensor alignFeatureDimensions(const Tensor& source, const Tensor& target);
    std::vector<double> computeFeatureSimilarity(const Tensor& features1, const Tensor& features2);
    
    // Temporal utilities
    std::vector<double> generateTimeGrid(double start_time, double end_time, size_t num_points);
    std::vector<double> interpolateTimestamps(const std::vector<double>& timestamps, 
                                            size_t target_length);
    double computeTemporalOverlap(const std::vector<double>& times1, 
                                 const std::vector<double>& times2);
    
    // Quality utilities
    ContentQualityMetrics combineQualityMetrics(const std::vector<ContentQualityMetrics>& metrics);
    double computeQualityScore(const ContentQualityMetrics& metrics, 
                              const std::map<std::string, double>& weights);
    std::string formatQualityReport(const ContentQualityMetrics& metrics);
    
    // Configuration utilities
    FusionConfig createOptimalConfig(const MultiModalContent& sample_content);
    std::map<std::string, double> analyzeContentCharacteristics(const MultiModalContent& content);    VideoAudioTextFusionStrategy selectOptimalStrategy(const MultiModalContent& content, 
                                       const std::map<std::string, double>& requirements);
}

} // namespace ai
} // namespace asekioml
