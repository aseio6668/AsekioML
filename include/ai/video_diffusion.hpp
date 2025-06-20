#pragma once

#include "tensor.hpp"
#include "ai/video_tensor_ops.hpp"
#include "ai/text_to_image.hpp"
#include <string>
#include <vector>
#include <memory>

namespace asekioml {
namespace ai {

/**
 * @brief Video Diffusion Model for temporal video generation
 * 
 * Extends the existing 2D diffusion model to handle temporal (video) data.
 * Supports both frame-by-frame and video-chunk processing strategies.
 */
class VideoDiffusionModel {
public:
    /**
     * @brief Video generation strategy
     */
    enum class GenerationStrategy {
        FRAME_BY_FRAME,    // Generate each frame independently 
        VIDEO_CHUNK,       // Generate video chunks with temporal consistency
        TEMPORAL_AWARE     // Fully temporal-aware generation with 3D U-Net
    };

    /**
     * @brief Motion-aware noise scheduling
     */
    enum class MotionSchedule {
        UNIFORM,           // Same noise schedule for all frames
        MOTION_ADAPTIVE,   // Adapt noise based on estimated motion
        TEMPORAL_DECAY     // Decay noise influence over time
    };

    // ===== Temporal Noise Scheduling =====
    
    /**
     * @brief Create motion-aware noise schedule for video
     * @param num_steps Number of diffusion steps
     * @param num_frames Number of video frames
     * @param motion_schedule Type of motion scheduling
     * @param beta_start Starting beta value
     * @param beta_end Ending beta value
     * @return 2D vector of beta values [step, frame]
     */
    static std::vector<std::vector<double>> create_temporal_schedule(
        int num_steps, 
        int num_frames,
        MotionSchedule motion_schedule = MotionSchedule::UNIFORM,
        double beta_start = 0.0001, 
        double beta_end = 0.02
    );

    /**
     * @brief Compute temporal alpha values from beta schedule
     * @param temporal_betas 2D vector of beta values [step, frame]
     * @return Pair of (temporal_alphas, temporal_alpha_cumprod)
     */
    static std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
        compute_temporal_alphas(const std::vector<std::vector<double>>& temporal_betas);

    // ===== Temporal Forward Process =====
    
    /**
     * @brief Add noise to video at specific timestep
     * @param x0 Original clean video tensor [B, T, C, H, W]
     * @param t Timestep
     * @param temporal_alpha_cumprod Cumulative alpha values [step, frame]
     * @param format Video tensor format
     * @return Noisy video tensor
     */
    static Tensor add_temporal_noise(
        const Tensor& x0, 
        int t, 
        const std::vector<std::vector<double>>& temporal_alpha_cumprod,
        VideoFormat format = VideoFormat::BTCHW
    );

    /**
     * @brief Sample temporal noise tensor
     * @param video_shape Shape of video tensor to generate
     * @param format Video tensor format
     * @return Random noise tensor with temporal correlation
     */
    static Tensor sample_temporal_noise(
        const std::vector<size_t>& video_shape,
        VideoFormat format = VideoFormat::BTCHW
    );

    // ===== Temporal Reverse Process =====

    /**
     * @brief Single temporal denoising step
     * @param x_t Noisy video at timestep t
     * @param predicted_noise Predicted noise from temporal model
     * @param t Current timestep
     * @param temporal_alphas Alpha values [step, frame]
     * @param temporal_alpha_cumprod Cumulative alpha values [step, frame]
     * @param temporal_betas Beta values [step, frame]
     * @param format Video tensor format
     * @return Denoised video at timestep t-1
     */
    static Tensor temporal_denoise_step(
        const Tensor& x_t,
        const Tensor& predicted_noise,
        int t,
        const std::vector<std::vector<double>>& temporal_alphas,
        const std::vector<std::vector<double>>& temporal_alpha_cumprod,
        const std::vector<std::vector<double>>& temporal_betas,
        VideoFormat format = VideoFormat::BTCHW
    );

    /**
     * @brief Full temporal denoising loop
     * @param x_t Initial noise tensor
     * @param text_embeddings Text conditioning
     * @param num_steps Number of denoising steps
     * @param strategy Generation strategy
     * @param guidance_scale Classifier-free guidance scale
     * @param format Video tensor format
     * @return Generated video tensor
     */
    static Tensor temporal_sample_loop(
        const Tensor& x_t,
        const Tensor& text_embeddings,
        int num_steps = 50,
        GenerationStrategy strategy = GenerationStrategy::TEMPORAL_AWARE,
        double guidance_scale = 7.5,
        VideoFormat format = VideoFormat::BTCHW
    );
};

/**
 * @brief Temporal U-Net for video diffusion
 * 
 * Extends the 2D U-Net to handle temporal dimensions with 3D convolutions
 * and temporal attention mechanisms.
 */
class TemporalUNet {
public:
    /**
     * @brief Constructor
     * @param in_channels Input channels
     * @param out_channels Output channels
     * @param time_embed_dim Time embedding dimension
     * @param text_embed_dim Text embedding dimension
     * @param num_frames Number of frames to process
     */
    TemporalUNet(size_t in_channels = 4, 
                 size_t out_channels = 4,
                 size_t time_embed_dim = 512,
                 size_t text_embed_dim = 768,
                 size_t num_frames = 16);

    /**
     * @brief Forward pass through temporal U-Net
     * @param x_t Noisy video input [B, T, C, H, W]
     * @param t Timestep
     * @param text_embeddings Text conditioning [B, seq_len, embed_dim]
     * @param format Video tensor format
     * @return Predicted noise [B, T, C, H, W]
     */
    Tensor forward(
        const Tensor& x_t, 
        int t, 
        const Tensor& text_embeddings,
        VideoFormat format = VideoFormat::BTCHW
    );

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t time_embed_dim_;
    size_t text_embed_dim_;
    size_t num_frames_;

    // ===== Core Components =====

    /**
     * @brief 3D Convolutional block with temporal dimension
     * @param input Input tensor [B, T, C, H, W]
     * @param out_channels Output channels
     * @param kernel_size Spatial kernel size (temporal kernel is 3)
     * @param stride Stride
     * @param padding Padding
     * @return Convolved tensor
     */
    Tensor conv3d_block(
        const Tensor& input,
        size_t out_channels,
        size_t kernel_size = 3,
        size_t stride = 1,
        size_t padding = 1
    );

    /**
     * @brief Temporal attention across frames
     * @param input Input tensor [B, T, C, H, W]
     * @param num_heads Number of attention heads
     * @return Attention-processed tensor
     */
    Tensor temporal_attention_block(
        const Tensor& input,
        size_t num_heads = 8
    );

    /**
     * @brief Temporal residual block
     * @param input Input tensor
     * @param channels Number of channels
     * @return Residual block output
     */
    Tensor temporal_residual_block(
        const Tensor& input,
        size_t channels
    );

    /**
     * @brief Time embedding for temporal awareness
     * @param t Timestep
     * @param embed_dim Embedding dimension
     * @return Time embedding vector
     */
    Tensor time_embedding(int t, size_t embed_dim);

    /**
     * @brief Positional encoding for temporal positions
     * @param num_frames Number of frames
     * @param embed_dim Embedding dimension
     * @return Positional encoding tensor [T, embed_dim]
     */
    Tensor temporal_positional_encoding(size_t num_frames, size_t embed_dim);
};

/**
 * @brief Complete text-to-video pipeline
 * 
 * Combines text processing and video diffusion model for end-to-end
 * text-to-video generation.
 */
class TextToVideoPipeline {
public:
    /**
     * @brief Constructor
     * @param video_size Spatial size of generated videos (assumed square)
     * @param num_frames Number of frames to generate
     * @param num_steps Number of diffusion steps
     * @param strategy Generation strategy
     */
    TextToVideoPipeline(
        size_t video_size = 256, 
        size_t num_frames = 16,
        int num_steps = 50,
        VideoDiffusionModel::GenerationStrategy strategy = VideoDiffusionModel::GenerationStrategy::TEMPORAL_AWARE
    );

    /**
     * @brief Generate video from text prompt
     * @param prompt Text description
     * @param guidance_scale Classifier-free guidance scale
     * @param seed Random seed for reproducibility
     * @param motion_schedule Motion-aware scheduling
     * @return Generated video tensor [T, C, H, W] or [B, T, C, H, W]
     */
    Tensor generate(
        const std::string& prompt, 
        double guidance_scale = 7.5,
        int seed = -1,
        VideoDiffusionModel::MotionSchedule motion_schedule = VideoDiffusionModel::MotionSchedule::MOTION_ADAPTIVE
    );

    /**
     * @brief Generate video with frame interpolation
     * @param prompt Text description
     * @param keyframes Vector of keyframe indices
     * @param guidance_scale Classifier-free guidance scale
     * @param seed Random seed
     * @return Generated video with interpolated frames
     */
    Tensor generate_with_interpolation(
        const std::string& prompt,
        const std::vector<int>& keyframes,
        double guidance_scale = 7.5,
        int seed = -1
    );

    /**
     * @brief Video-to-video generation (style transfer, etc.)
     * @param source_video Input video tensor
     * @param prompt Text description for transformation
     * @param strength Transformation strength (0.0 to 1.0)
     * @param guidance_scale Classifier-free guidance scale
     * @return Transformed video tensor
     */
    Tensor video_to_video(
        const Tensor& source_video,
        const std::string& prompt,
        double strength = 0.7,
        double guidance_scale = 7.5
    );

    /**
     * @brief Get temporal consistency metrics
     * @param video_tensor Generated video tensor
     * @return Consistency score (higher is better)
     */
    double compute_temporal_consistency(const Tensor& video_tensor);

private:
    size_t video_size_;
    size_t num_frames_;
    int num_steps_;
    VideoDiffusionModel::GenerationStrategy strategy_;
    
    std::unique_ptr<TemporalUNet> temporal_unet_;
    
    // Diffusion parameters
    std::vector<std::vector<double>> temporal_betas_;
    std::vector<std::vector<double>> temporal_alphas_;
    std::vector<std::vector<double>> temporal_alpha_cumprod_;

    /**
     * @brief Initialize temporal diffusion parameters
     */
    void initialize_temporal_parameters();

    /**
     * @brief Create initial noise for video generation
     * @param batch_size Batch size
     * @return Initial noise tensor
     */
    Tensor create_initial_noise(size_t batch_size = 1);
};

/**
 * @brief Video motion estimation utilities for diffusion
 */
class VideoMotionUtils {
public:
    /**
     * @brief Estimate motion between consecutive frames
     * @param video_tensor Input video tensor
     * @param format Video tensor format
     * @return Motion vectors tensor
     */
    static Tensor estimate_motion_vectors(
        const Tensor& video_tensor,
        VideoFormat format = VideoFormat::BTCHW
    );

    /**
     * @brief Compute motion-aware scheduling weights
     * @param motion_vectors Motion vectors from estimate_motion_vectors
     * @param base_schedule Base scheduling values
     * @return Motion-adapted schedule
     */
    static std::vector<std::vector<double>> compute_motion_schedule(
        const Tensor& motion_vectors,
        const std::vector<double>& base_schedule
    );

    /**
     * @brief Temporal smoothing for generated videos
     * @param video_tensor Input video tensor
     * @param smoothing_factor Smoothing strength (0.0 to 1.0)
     * @param format Video tensor format
     * @return Smoothed video tensor
     */
    static Tensor temporal_smoothing(
        const Tensor& video_tensor,
        double smoothing_factor = 0.3,
        VideoFormat format = VideoFormat::BTCHW
    );

    /**
     * @brief Frame warping for temporal consistency
     * @param frame Current frame tensor
     * @param prev_frame Previous frame tensor
     * @param motion_vectors Motion vectors
     * @return Warped frame tensor
     */
    static Tensor warp_frame(
        const Tensor& frame,
        const Tensor& prev_frame,
        const Tensor& motion_vectors
    );
};

} // namespace ai
} // namespace asekioml
