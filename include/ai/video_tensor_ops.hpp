#pragma once

#include "tensor.hpp"
#include "multimodal_attention.hpp"
#include <string>
#include <vector>
#include <memory>

namespace asekioml {
namespace ai {

/**
 * @brief Video-specific tensor operations and utilities
 * 
 * Extends the existing Tensor class with video-specific functionality
 * including temporal operations, frame extraction, and video preprocessing.
 */

/**
 * @brief Video tensor format specifications
 */
enum class VideoFormat {
    BTHWC,  // [Batch, Time, Height, Width, Channels] - Common for processing
    BTCHW,  // [Batch, Time, Channels, Height, Width] - Common for models
    TBHWC,  // [Time, Batch, Height, Width, Channels] - Temporal-first
    TBCHW   // [Time, Batch, Channels, Height, Width] - Temporal-first with channel priority
};

/**
 * @brief Video preprocessing and postprocessing utilities
 */
class VideoTensorUtils {
public:
    /**
     * @brief Create a video tensor from frame sequence
     * @param frames Vector of image tensors (each should be [H, W, C] or [C, H, W])
     * @param format Desired video tensor format
     * @return Video tensor in specified format
     */
    static Tensor create_video_tensor(const std::vector<Tensor>& frames, VideoFormat format = VideoFormat::BTHWC);
    
    /**
     * @brief Extract frames from video tensor
     * @param video_tensor Video tensor in any supported format
     * @param format Format of the input video tensor
     * @return Vector of frame tensors
     */
    static std::vector<Tensor> extract_frames(const Tensor& video_tensor, VideoFormat format = VideoFormat::BTHWC);
    
    /**
     * @brief Convert between video tensor formats
     * @param video_tensor Input video tensor
     * @param from_format Current format
     * @param to_format Desired format
     * @return Converted video tensor
     */
    static Tensor convert_format(const Tensor& video_tensor, VideoFormat from_format, VideoFormat to_format);
    
    /**
     * @brief Temporal resize (interpolate frames to change temporal resolution)
     * @param video_tensor Input video tensor
     * @param target_frames Target number of frames
     * @param format Video tensor format
     * @return Temporally resized video tensor
     */
    static Tensor temporal_resize(const Tensor& video_tensor, size_t target_frames, VideoFormat format = VideoFormat::BTHWC);
    
    /**
     * @brief Temporal crop (extract subsequence of frames)
     * @param video_tensor Input video tensor
     * @param start_frame Starting frame index
     * @param num_frames Number of frames to extract
     * @param format Video tensor format
     * @return Cropped video tensor
     */
    static Tensor temporal_crop(const Tensor& video_tensor, size_t start_frame, size_t num_frames, VideoFormat format = VideoFormat::BTHWC);
    
    /**
     * @brief Temporal padding (pad with frames at beginning/end)
     * @param video_tensor Input video tensor
     * @param pad_before Number of frames to pad before
     * @param pad_after Number of frames to pad after
     * @param pad_mode Padding mode ("replicate", "zero", "reflect")
     * @param format Video tensor format
     * @return Padded video tensor
     */
    static Tensor temporal_pad(const Tensor& video_tensor, size_t pad_before, size_t pad_after, 
                              const std::string& pad_mode = "replicate", VideoFormat format = VideoFormat::BTHWC);
    
    /**
     * @brief Get video tensor dimensions
     */
    struct VideoTensorInfo {
        size_t batch_size;
        size_t num_frames;
        size_t height;
        size_t width;
        size_t channels;
        VideoFormat format;
    };
      static VideoTensorInfo get_video_info(const Tensor& video_tensor, VideoFormat format);
    
    /**
     * @brief Get tensor info (alias for get_video_info for compatibility)
     */
    static VideoTensorInfo get_tensor_info(const Tensor& video_tensor, VideoFormat format) {
        return get_video_info(video_tensor, format);
    }
    
    /**
     * @brief Validate video tensor shape for given format
     */
    static bool validate_video_tensor(const Tensor& video_tensor, VideoFormat format);

private:
    static std::vector<size_t> get_dimension_order(VideoFormat format);
    static std::vector<size_t> get_inverse_dimension_order(VideoFormat format);
};

/**
 * @brief Temporal convolution operations
 */
class TemporalConvolution {
public:
    /**
     * @brief 3D convolution configuration
     */
    struct Conv3DConfig {
        size_t in_channels;
        size_t out_channels;
        std::vector<size_t> kernel_size;  // [temporal, height, width]
        std::vector<size_t> stride = {1, 1, 1};
        std::vector<size_t> padding = {0, 0, 0};
        bool use_bias = true;
    };
    
    /**
     * @brief 3D convolution layer
     */
    class Conv3DLayer {
    public:
        Conv3DLayer(const Conv3DConfig& config);
        
        /**
         * @brief Forward pass
         * @param input Video tensor in BTCHW format
         * @return Convolved video tensor
         */
        Tensor forward(const Tensor& input);
        
        /**
         * @brief Get number of parameters
         */
        size_t get_param_count() const;
        
        /**
         * @brief Initialize weights (Xavier/He initialization)
         */
        void initialize_weights(const std::string& init_type = "xavier");
        
    private:
        Conv3DConfig config_;
        Tensor weights_;  // [out_channels, in_channels, kernel_t, kernel_h, kernel_w]
        Tensor bias_;     // [out_channels]
        
        void initialize_tensors();
        Tensor apply_conv3d(const Tensor& input) const;
    };
    
    /**
     * @brief Temporal convolution (2D conv + 1D temporal)
     */
    class Conv2Plus1DLayer {
    public:
        Conv2Plus1DLayer(const Conv3DConfig& config);
        
        Tensor forward(const Tensor& input);
        size_t get_param_count() const;
        void initialize_weights(const std::string& init_type = "xavier");
          private:
        Conv3DConfig config_;
        // For now, use simple implementation without separate 2D/1D conv layers
        // TODO: Implement proper 2D and 1D convolution decomposition
    };
};

/**
 * @brief Temporal attention mechanisms
 */
class TemporalAttention {
public:
    /**
     * @brief Configuration for temporal attention
     */
    struct TemporalAttentionConfig {
        size_t embed_dim;
        size_t num_heads = 8;
        size_t max_sequence_length = 32;  // Maximum number of frames
        double dropout_rate = 0.1;
        bool use_positional_encoding = true;
    };
    
    /**
     * @brief Temporal self-attention layer
     */
    class TemporalSelfAttention {
    public:
        TemporalSelfAttention(const TemporalAttentionConfig& config);
        
        /**
         * @brief Apply temporal self-attention
         * @param video_features Video tensor [B, T, H, W, C] -> [B, T, C] after spatial pooling
         * @return Temporally attended features [B, T, C]
         */
        Tensor forward(const Tensor& video_features);
        
        /**
         * @brief Get attention weights for visualization
         */
        Tensor get_attention_weights() const { return attention_weights_; }
        
    private:
        TemporalAttentionConfig config_;
        std::unique_ptr<MultiHeadAttentionLayer> attention_layer_;
        Tensor positional_encoding_;
        Tensor attention_weights_;
        
        void initialize_positional_encoding();
        Tensor apply_spatial_pooling(const Tensor& video_tensor);
    };
    
    /**
     * @brief Cross-temporal attention (between different temporal positions)
     */
    class CrossTemporalAttention {
    public:
        CrossTemporalAttention(const TemporalAttentionConfig& config);
        
        /**
         * @brief Apply cross-temporal attention between frames
         * @param query_frames Query video tensor [B, T1, C]
         * @param key_value_frames Key/Value video tensor [B, T2, C]
         * @return Cross-temporally attended features [B, T1, C]
         */
        Tensor forward(const Tensor& query_frames, const Tensor& key_value_frames);
        
    private:
        TemporalAttentionConfig config_;
        std::unique_ptr<CrossModalAttention> cross_attention_;
    };
};

/**
 * @brief Frame interpolation utilities
 */
class FrameInterpolation {
public:
    /**
     * @brief Linear interpolation between frames
     * @param frame1 First frame tensor [H, W, C]
     * @param frame2 Second frame tensor [H, W, C]
     * @param alpha Interpolation factor (0.0 = frame1, 1.0 = frame2)
     * @return Interpolated frame
     */
    static Tensor linear_interpolate(const Tensor& frame1, const Tensor& frame2, double alpha);
    
    /**
     * @brief Interpolate multiple frames between two keyframes
     * @param frame1 First keyframe
     * @param frame2 Second keyframe
     * @param num_intermediate_frames Number of frames to generate between keyframes
     * @return Vector of interpolated frames (not including keyframes)
     */
    static std::vector<Tensor> interpolate_sequence(const Tensor& frame1, const Tensor& frame2, size_t num_intermediate_frames);
    
    /**
     * @brief Upsample video temporally using interpolation
     * @param video_tensor Input video tensor
     * @param scale_factor Temporal upsampling factor (2 = double frame rate)
     * @param format Video tensor format
     * @return Upsampled video tensor
     */
    static Tensor temporal_upsample(const Tensor& video_tensor, size_t scale_factor, VideoFormat format = VideoFormat::BTHWC);
};

/**
 * @brief Motion and optical flow utilities
 */
class MotionEstimation {
public:
    /**
     * @brief Simple optical flow estimation between two frames
     * @param frame1 First frame [H, W, C]
     * @param frame2 Second frame [H, W, C]
     * @param block_size Block size for motion estimation
     * @return Motion vectors [H/block_size, W/block_size, 2] (dx, dy)
     */
    static Tensor estimate_optical_flow(const Tensor& frame1, const Tensor& frame2, size_t block_size = 8);
    
    /**
     * @brief Apply motion vectors to warp a frame
     * @param frame Input frame [H, W, C]
     * @param motion_vectors Motion field [H, W, 2]
     * @return Warped frame
     */
    static Tensor warp_frame(const Tensor& frame, const Tensor& motion_vectors);
    
    /**
     * @brief Estimate motion for entire video sequence
     * @param video_tensor Video tensor in any format
     * @param format Video tensor format
     * @return Motion vectors for each frame transition [B, T-1, H, W, 2]
     */
    static Tensor estimate_video_motion(const Tensor& video_tensor, VideoFormat format = VideoFormat::BTHWC);
};

/**
 * @brief Video preprocessing pipeline
 */
class VideoPreprocessor {
public:
    /**
     * @brief Video preprocessing configuration
     */
    struct PreprocessConfig {
        size_t target_height = 224;
        size_t target_width = 224;
        size_t target_frames = 16;
        VideoFormat output_format = VideoFormat::BTHWC;
        bool normalize = true;
        std::vector<double> mean = {0.485, 0.456, 0.406};  // ImageNet means
        std::vector<double> std = {0.229, 0.224, 0.225};   // ImageNet stds
        bool temporal_center_crop = true;
    };
    
    VideoPreprocessor(const PreprocessConfig& config);
    
    /**
     * @brief Preprocess video tensor for model input
     * @param video_tensor Raw video tensor
     * @param input_format Format of input video
     * @return Preprocessed video tensor
     */
    Tensor preprocess(const Tensor& video_tensor, VideoFormat input_format = VideoFormat::BTHWC);
    
    /**
     * @brief Postprocess model output back to video format
     * @param model_output Model output tensor
     * @return Postprocessed video tensor
     */
    Tensor postprocess(const Tensor& model_output);
    
private:
    PreprocessConfig config_;
    
    Tensor spatial_resize(const Tensor& video_tensor, VideoFormat format);
    Tensor temporal_resample(const Tensor& video_tensor, VideoFormat format);
    Tensor normalize_video(const Tensor& video_tensor, VideoFormat format);
    Tensor denormalize_video(const Tensor& video_tensor, VideoFormat format);
    
    // Helper method for bilinear interpolation
    Tensor resize_frame_bilinear(const Tensor& frame, size_t target_height, size_t target_width);
};

} // namespace ai
} // namespace asekioml
