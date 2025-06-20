#pragma once

#include "tensor.hpp"
#include "video_tensor_ops.hpp"
#include "simple_video_diffusion.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace asekioml {
namespace ai {

// Forward declarations
class MultiHeadAttentionLayer;
class Conv2DLayer;

/**
 * Advanced Optical Flow Estimation
 * Week 11: Enhanced motion vector computation with sub-pixel accuracy
 */
class AdvancedOpticalFlow {
public:
    struct FlowField {
        Tensor flow_x;     // Horizontal displacement field
        Tensor flow_y;     // Vertical displacement field
        Tensor confidence; // Flow confidence map
        double quality_score;
        
        FlowField() : quality_score(0.0) {}
        FlowField(const Tensor& fx, const Tensor& fy, const Tensor& conf, double quality = 0.0)
            : flow_x(fx), flow_y(fy), confidence(conf), quality_score(quality) {}
    };
    
    struct FlowConfig {
        int pyramid_levels = 3;     // Multi-scale pyramid levels
        int window_size = 5;        // Optical flow window size
        double confidence_threshold = 0.5;
        bool use_subpixel_refinement = true;
        bool temporal_consistency = true;
        
        FlowConfig() = default;
    };

private:
    FlowConfig config_;
    std::vector<Tensor> image_pyramid_;
    
public:
    AdvancedOpticalFlow(const FlowConfig& config = FlowConfig());
    
    // Core optical flow computation
    FlowField compute_flow(const Tensor& frame1, const Tensor& frame2);
    
    // Multi-frame flow estimation for temporal consistency
    std::vector<FlowField> compute_temporal_flow(const Tensor& video_sequence);
    
    // Flow field processing
    FlowField refine_flow(const FlowField& initial_flow, const Tensor& frame1, const Tensor& frame2);
    Tensor warp_frame(const Tensor& frame, const FlowField& flow);
    
    // Flow analysis
    double compute_flow_quality(const FlowField& flow);
    Tensor compute_motion_boundaries(const FlowField& flow);
    
private:
    void build_pyramid(const Tensor& image, std::vector<Tensor>& pyramid);
    FlowField compute_flow_at_level(const Tensor& img1, const Tensor& img2, int level);
    FlowField upsample_flow(const FlowField& flow, int scale_factor);
};

/**
 * Neural Motion Field Interpolation
 * Week 11: Learned motion field prediction for frame interpolation
 */
class MotionFieldInterpolator {
public:
    struct MotionField {
        Tensor forward_flow;   // Motion from frame A to interpolated frame
        Tensor backward_flow;  // Motion from frame B to interpolated frame
        Tensor occlusion_mask; // Occlusion detection mask
        Tensor blend_weights;  // Blending weights for interpolation
        
        MotionField() = default;
        MotionField(const Tensor& ff, const Tensor& bf, const Tensor& mask, const Tensor& weights)
            : forward_flow(ff), backward_flow(bf), occlusion_mask(mask), blend_weights(weights) {}
    };
    
    struct InterpolationConfig {
        bool use_bilateral_motion = true;
        bool occlusion_detection = true;
        double temporal_smoothing = 0.1;
        int refinement_iterations = 3;
        
        InterpolationConfig() = default;
    };

private:
    InterpolationConfig config_;
    std::unique_ptr<MultiHeadAttentionLayer> motion_attention_;
    std::unique_ptr<Conv2DLayer> motion_encoder_;
    std::unique_ptr<Conv2DLayer> motion_decoder_;
    
public:
    MotionFieldInterpolator(const InterpolationConfig& config = InterpolationConfig());
    ~MotionFieldInterpolator();
    
    // Motion field prediction
    MotionField predict_motion_field(const Tensor& frame_a, const Tensor& frame_b, double t = 0.5);
    
    // Multi-frame motion prediction
    std::vector<MotionField> predict_motion_sequence(const Tensor& video_frames, 
                                                   const std::vector<double>& time_points);
    
    // Motion field refinement
    MotionField refine_motion_field(const MotionField& initial_field, 
                                  const Tensor& frame_a, const Tensor& frame_b);
    
    // Occlusion handling
    Tensor detect_occlusions(const AdvancedOpticalFlow::FlowField& forward_flow,
                           const AdvancedOpticalFlow::FlowField& backward_flow);
    
private:
    void initialize_networks();
    Tensor encode_motion_context(const Tensor& frame_a, const Tensor& frame_b);
    MotionField decode_motion_field(const Tensor& motion_encoding, double t);
};

/**
 * Advanced Frame Interpolator
 * Week 11: High-quality frame interpolation using learned motion fields
 */
class AdvancedFrameInterpolator {
public:
    struct InterpolationResult {
        Tensor interpolated_frame;
        MotionFieldInterpolator::MotionField motion_field;
        double quality_score;
        std::vector<Tensor> intermediate_frames; // For debugging/visualization
        
        InterpolationResult() : quality_score(0.0) {}
    };
    
    struct InterpolatorConfig {
        bool use_adaptive_blending = true;
        bool temporal_consistency_loss = true;
        int quality_refinement_steps = 2;
        double motion_compensation_strength = 1.0;
        
        InterpolatorConfig() = default;
    };

private:
    InterpolatorConfig config_;
    std::unique_ptr<AdvancedOpticalFlow> optical_flow_;
    std::unique_ptr<MotionFieldInterpolator> motion_interpolator_;
    std::unique_ptr<Conv2DLayer> quality_network_;
    
public:
    AdvancedFrameInterpolator(const InterpolatorConfig& config = InterpolatorConfig());
    ~AdvancedFrameInterpolator();
    
    // Single frame interpolation
    InterpolationResult interpolate_frame(const Tensor& frame_a, const Tensor& frame_b, double t = 0.5);
    
    // Multi-frame interpolation
    std::vector<InterpolationResult> interpolate_sequence(const Tensor& video_frames, 
                                                        const std::vector<double>& time_points);
    
    // Temporal upsampling
    Tensor upsample_video_temporal(const Tensor& input_video, int target_fps_multiplier);
    
    // Quality assessment
    double assess_interpolation_quality(const InterpolationResult& result,
                                      const Tensor& frame_a, const Tensor& frame_b);
    
private:
    void initialize_components();
    Tensor blend_frames_with_motion(const Tensor& frame_a, const Tensor& frame_b,
                                  const MotionFieldInterpolator::MotionField& motion_field, double t);
    Tensor apply_temporal_consistency(const std::vector<Tensor>& frame_sequence);
};

/**
 * Motion-Guided Frame Synthesis
 * Week 11: Generate new frames guided by motion patterns
 */
class MotionGuidedSynthesis {
public:
    struct SynthesisConfig {
        bool use_motion_prediction = true;
        bool style_preservation = true;
        double motion_strength_threshold = 0.1;
        int synthesis_refinement_steps = 3;
        
        SynthesisConfig() = default;
    };
    
    struct SynthesisResult {
        Tensor synthesized_frame;
        Tensor motion_guidance_map;
        double synthesis_confidence;
        
        SynthesisResult() : synthesis_confidence(0.0) {}
    };

private:
    SynthesisConfig config_;
    std::unique_ptr<SimpleVideoDiffusionModel> video_diffusion_;
    std::unique_ptr<AdvancedOpticalFlow> motion_analyzer_;
    std::unique_ptr<Conv2DLayer> motion_encoder_;
    
public:
    MotionGuidedSynthesis(const SynthesisConfig& config = SynthesisConfig());
    ~MotionGuidedSynthesis();
    
    // Motion-guided frame synthesis
    SynthesisResult synthesize_frame(const std::vector<Tensor>& context_frames, 
                                   const Tensor& motion_guidance);
    
    // Video sequence synthesis
    Tensor synthesize_video_sequence(const Tensor& seed_frames, int target_length);
    
    // Motion pattern synthesis
    Tensor synthesize_motion_pattern(const std::vector<AdvancedOpticalFlow::FlowField>& flow_history,
                                   int prediction_steps);
    
    // Style transfer with motion preservation
    Tensor transfer_motion_style(const Tensor& content_video, const Tensor& style_reference);
    
private:
    void initialize_synthesis_components();
    Tensor encode_motion_guidance(const Tensor& motion_data);
    Tensor apply_motion_conditioning(const Tensor& base_frame, const Tensor& motion_guidance);
};

/**
 * Temporal Smoothing and Enhancement
 * Week 11: Advanced temporal processing for video quality
 */
class TemporalSmoothingEngine {
public:
    struct SmoothingConfig {
        double temporal_window_size = 1.0;    // Time window for smoothing (seconds)
        bool adaptive_smoothing = true;       // Adapt to motion content
        bool preserve_details = true;         // Detail preservation during smoothing
        double motion_threshold = 0.05;       // Motion sensitivity threshold
        
        SmoothingConfig() = default;
    };

private:
    SmoothingConfig config_;
    std::unique_ptr<AdvancedOpticalFlow> motion_analyzer_;
    std::vector<Tensor> temporal_buffer_;
    
public:
    TemporalSmoothingEngine(const SmoothingConfig& config = SmoothingConfig());
    ~TemporalSmoothingEngine();
    
    // Temporal smoothing
    Tensor apply_temporal_smoothing(const Tensor& video_sequence);
    
    // Adaptive temporal filtering
    Tensor adaptive_temporal_filter(const Tensor& video_sequence, 
                                  const std::vector<AdvancedOpticalFlow::FlowField>& motion_fields);
    
    // Temporal noise reduction
    Tensor reduce_temporal_noise(const Tensor& noisy_video);
    
    // Motion-aware enhancement
    Tensor enhance_temporal_consistency(const Tensor& video_sequence);
    
    // Temporal upsampling with smoothing
    Tensor upsample_and_smooth(const Tensor& input_video, int upsampling_factor);
    
private:
    Tensor compute_temporal_weights(const std::vector<AdvancedOpticalFlow::FlowField>& motion_fields);
    Tensor apply_motion_compensated_smoothing(const Tensor& video_sequence, 
                                            const Tensor& motion_weights);
};

/**
 * Integrated Advanced Frame Processing Pipeline
 * Week 11: Complete pipeline combining all advanced frame interpolation capabilities
 */
class AdvancedFrameProcessingPipeline {
public:
    struct ProcessingConfig {
        bool use_optical_flow = true;
        bool use_motion_interpolation = true;
        bool use_frame_synthesis = true;
        bool use_temporal_smoothing = true;
        
        AdvancedOpticalFlow::FlowConfig flow_config;
        MotionFieldInterpolator::InterpolationConfig interpolation_config;
        AdvancedFrameInterpolator::InterpolatorConfig frame_config;
        MotionGuidedSynthesis::SynthesisConfig synthesis_config;
        TemporalSmoothingEngine::SmoothingConfig smoothing_config;
        
        ProcessingConfig() = default;
    };
    
    struct ProcessingResult {
        Tensor processed_video;
        std::vector<AdvancedOpticalFlow::FlowField> motion_fields;
        std::vector<AdvancedFrameInterpolator::InterpolationResult> interpolation_results;
        double overall_quality_score;
        
        ProcessingResult() : overall_quality_score(0.0) {}
    };

private:
    ProcessingConfig config_;
    std::unique_ptr<AdvancedOpticalFlow> optical_flow_;
    std::unique_ptr<MotionFieldInterpolator> motion_interpolator_;
    std::unique_ptr<AdvancedFrameInterpolator> frame_interpolator_;
    std::unique_ptr<MotionGuidedSynthesis> frame_synthesis_;
    std::unique_ptr<TemporalSmoothingEngine> temporal_smoother_;
    
public:
    AdvancedFrameProcessingPipeline(const ProcessingConfig& config = ProcessingConfig());
    ~AdvancedFrameProcessingPipeline();
    
    // Complete video processing pipeline
    ProcessingResult process_video(const Tensor& input_video);
    
    // Targeted processing operations
    Tensor interpolate_to_target_fps(const Tensor& input_video, double target_fps, double source_fps);
    Tensor enhance_video_quality(const Tensor& input_video);
    Tensor synthesize_missing_frames(const Tensor& incomplete_video, const std::vector<int>& missing_indices);
    
    // Performance and quality analysis
    std::vector<double> analyze_processing_quality(const ProcessingResult& result);
    void benchmark_processing_performance(const std::vector<Tensor>& test_videos);
    
private:
    void initialize_pipeline_components();
    ProcessingResult create_result(const Tensor& processed_video, 
                                 const std::vector<AdvancedOpticalFlow::FlowField>& flows,
                                 const std::vector<AdvancedFrameInterpolator::InterpolationResult>& interpolations);
};

} // namespace ai
} // namespace asekioml
