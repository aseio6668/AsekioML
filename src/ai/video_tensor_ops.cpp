#include "ai/video_tensor_ops.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <random>

namespace asekioml {
namespace ai {

// VideoTensorUtils Implementation

Tensor VideoTensorUtils::create_video_tensor(const std::vector<Tensor>& frames, VideoFormat format) {
    if (frames.empty()) {
        throw std::invalid_argument("Cannot create video tensor from empty frame sequence");
    }
    
    // Validate that all frames have the same shape
    const auto& first_frame = frames[0];
    const auto& frame_shape = first_frame.shape();
    
    for (size_t i = 1; i < frames.size(); ++i) {
        if (frames[i].shape() != frame_shape) {
            throw std::invalid_argument("All frames must have the same shape");
        }
    }
    
    // Determine video tensor shape based on format
    std::vector<size_t> video_shape;
    size_t batch_size = 1;
    size_t num_frames = frames.size();
    
    // Assume frame shape is [H, W, C] or [C, H, W]
    size_t height, width, channels;
    if (frame_shape.size() == 3) {
        if (frame_shape[2] <= 4) {  // Likely [H, W, C]
            height = frame_shape[0];
            width = frame_shape[1];
            channels = frame_shape[2];
        } else {  // Likely [C, H, W]
            channels = frame_shape[0];
            height = frame_shape[1];
            width = frame_shape[2];
        }
    } else {
        throw std::invalid_argument("Frame tensors must be 3D [H,W,C] or [C,H,W]");
    }
    
    // Create video tensor shape based on format
    switch (format) {
        case VideoFormat::BTHWC:
            video_shape = {batch_size, num_frames, height, width, channels};
            break;
        case VideoFormat::BTCHW:
            video_shape = {batch_size, num_frames, channels, height, width};
            break;
        case VideoFormat::TBHWC:
            video_shape = {num_frames, batch_size, height, width, channels};
            break;
        case VideoFormat::TBCHW:
            video_shape = {num_frames, batch_size, channels, height, width};
            break;
    }
    
    Tensor video_tensor(video_shape);
    auto& video_data = video_tensor.data();
    
    // Copy frame data into video tensor
    size_t frame_size = first_frame.size();
    for (size_t t = 0; t < num_frames; ++t) {
        const auto& frame_data = frames[t].data();
        
        // Calculate offset in video tensor for this frame
        size_t offset;
        switch (format) {
            case VideoFormat::BTHWC:
            case VideoFormat::BTCHW:
                offset = t * frame_size;
                break;
            case VideoFormat::TBHWC:
            case VideoFormat::TBCHW:
                offset = t * batch_size * frame_size;
                break;
        }
        
        // Copy frame data
        std::copy(frame_data.begin(), frame_data.end(), video_data.begin() + offset);
    }
    
    return video_tensor;
}

std::vector<Tensor> VideoTensorUtils::extract_frames(const Tensor& video_tensor, VideoFormat format) {
    if (!validate_video_tensor(video_tensor, format)) {
        throw std::invalid_argument("Invalid video tensor for specified format");
    }
    
    auto info = get_video_info(video_tensor, format);
    std::vector<Tensor> frames;
    frames.reserve(info.num_frames);
    
    const auto& video_data = video_tensor.data();
    size_t frame_size = info.height * info.width * info.channels;
    
    for (size_t t = 0; t < info.num_frames; ++t) {
        // Create frame tensor shape [H, W, C]
        Tensor frame({info.height, info.width, info.channels});
        auto& frame_data = frame.data();
        
        // Calculate offset for this frame in video tensor
        size_t offset;
        switch (format) {
            case VideoFormat::BTHWC:
                offset = t * info.height * info.width * info.channels;
                break;
            case VideoFormat::BTCHW:
                // Need to reorder from [C, H, W] to [H, W, C]
                offset = t * info.channels * info.height * info.width;
                // TODO: Implement channel reordering
                break;
            case VideoFormat::TBHWC:
                offset = t * info.batch_size * info.height * info.width * info.channels;
                break;
            case VideoFormat::TBCHW:
                offset = t * info.batch_size * info.channels * info.height * info.width;
                // TODO: Implement channel reordering
                break;
        }
        
        // Copy frame data
        if (format == VideoFormat::BTHWC || format == VideoFormat::TBHWC) {
            // Direct copy for HWBC formats
            std::copy(video_data.begin() + offset, 
                     video_data.begin() + offset + frame_size, 
                     frame_data.begin());
        } else {
            // TODO: Implement channel reordering for BCHW formats
            // For now, do direct copy and add TODO for proper reordering
            std::copy(video_data.begin() + offset, 
                     video_data.begin() + offset + frame_size, 
                     frame_data.begin());
        }
        
        frames.push_back(std::move(frame));
    }
    
    return frames;
}

Tensor VideoTensorUtils::convert_format(const Tensor& video_tensor, VideoFormat from_format, VideoFormat to_format) {
    if (from_format == to_format) {
        return video_tensor;
    }
    
    auto info = get_video_info(video_tensor, from_format);
    
    // Create target shape
    std::vector<size_t> target_shape;
    switch (to_format) {
        case VideoFormat::BTHWC:
            target_shape = {info.batch_size, info.num_frames, info.height, info.width, info.channels};
            break;
        case VideoFormat::BTCHW:
            target_shape = {info.batch_size, info.num_frames, info.channels, info.height, info.width};
            break;
        case VideoFormat::TBHWC:
            target_shape = {info.num_frames, info.batch_size, info.height, info.width, info.channels};
            break;
        case VideoFormat::TBCHW:
            target_shape = {info.num_frames, info.batch_size, info.channels, info.height, info.width};
            break;
    }
    
    Tensor result(target_shape);
    
    // For now, implement a simple conversion via frame extraction and reconstruction
    // TODO: Implement direct tensor permutation for better performance
    auto frames = extract_frames(video_tensor, from_format);
    
    // Recreate video tensor in target format
    result = create_video_tensor(frames, to_format);
    
    return result;
}

Tensor VideoTensorUtils::temporal_resize(const Tensor& video_tensor, size_t target_frames, VideoFormat format) {
    auto info = get_video_info(video_tensor, format);
    
    if (info.num_frames == target_frames) {
        return video_tensor;
    }
    
    auto frames = extract_frames(video_tensor, format);
    
    if (target_frames < info.num_frames) {
        // Temporal downsampling - select evenly spaced frames
        std::vector<Tensor> downsampled_frames;
        downsampled_frames.reserve(target_frames);
        
        for (size_t i = 0; i < target_frames; ++i) {
            size_t source_idx = (i * info.num_frames) / target_frames;
            downsampled_frames.push_back(frames[source_idx]);
        }
        
        return create_video_tensor(downsampled_frames, format);
    } else {
        // Temporal upsampling - use interpolation
        return FrameInterpolation::temporal_upsample(video_tensor, 
                                                    (target_frames + info.num_frames - 1) / info.num_frames, 
                                                    format);
    }
}

Tensor VideoTensorUtils::temporal_crop(const Tensor& video_tensor, size_t start_frame, size_t num_frames, VideoFormat format) {
    auto info = get_video_info(video_tensor, format);
    
    if (start_frame >= info.num_frames) {
        throw std::invalid_argument("Start frame index out of bounds");
    }
    
    if (start_frame + num_frames > info.num_frames) {
        throw std::invalid_argument("Crop extends beyond video length");
    }
    
    auto frames = extract_frames(video_tensor, format);
    
    std::vector<Tensor> cropped_frames;
    cropped_frames.reserve(num_frames);
    
    for (size_t i = start_frame; i < start_frame + num_frames; ++i) {
        cropped_frames.push_back(frames[i]);
    }
    
    return create_video_tensor(cropped_frames, format);
}

Tensor VideoTensorUtils::temporal_pad(const Tensor& video_tensor, size_t pad_before, size_t pad_after, 
                                     const std::string& pad_mode, VideoFormat format) {
    auto frames = extract_frames(video_tensor, format);
    
    std::vector<Tensor> padded_frames;
    padded_frames.reserve(frames.size() + pad_before + pad_after);
    
    // Pad before
    for (size_t i = 0; i < pad_before; ++i) {
        if (pad_mode == "replicate") {
            padded_frames.push_back(frames[0]);
        } else if (pad_mode == "zero") {
            Tensor zero_frame = Tensor::zeros(frames[0].shape());
            padded_frames.push_back(zero_frame);
        } else if (pad_mode == "reflect") {
            size_t reflect_idx = std::min(pad_before - i - 1, frames.size() - 1);
            padded_frames.push_back(frames[reflect_idx]);
        } else {
            throw std::invalid_argument("Unsupported padding mode: " + pad_mode);
        }
    }
    
    // Original frames
    for (const auto& frame : frames) {
        padded_frames.push_back(frame);
    }
    
    // Pad after
    for (size_t i = 0; i < pad_after; ++i) {
        if (pad_mode == "replicate") {
            padded_frames.push_back(frames.back());
        } else if (pad_mode == "zero") {
            Tensor zero_frame = Tensor::zeros(frames[0].shape());
            padded_frames.push_back(zero_frame);
        } else if (pad_mode == "reflect") {
            size_t reflect_idx = frames.size() - 1 - std::min(i, frames.size() - 1);
            padded_frames.push_back(frames[reflect_idx]);
        }
    }
    
    return create_video_tensor(padded_frames, format);
}

VideoTensorUtils::VideoTensorInfo VideoTensorUtils::get_video_info(const Tensor& video_tensor, VideoFormat format) {
    const auto& shape = video_tensor.shape();
    VideoTensorInfo info;
    info.format = format;
    
    switch (format) {
        case VideoFormat::BTHWC:
            if (shape.size() != 5) throw std::invalid_argument("BTHWC format requires 5D tensor");
            info.batch_size = shape[0];
            info.num_frames = shape[1];
            info.height = shape[2];
            info.width = shape[3];
            info.channels = shape[4];
            break;
            
        case VideoFormat::BTCHW:
            if (shape.size() != 5) throw std::invalid_argument("BTCHW format requires 5D tensor");
            info.batch_size = shape[0];
            info.num_frames = shape[1];
            info.channels = shape[2];
            info.height = shape[3];
            info.width = shape[4];
            break;
            
        case VideoFormat::TBHWC:
            if (shape.size() != 5) throw std::invalid_argument("TBHWC format requires 5D tensor");
            info.num_frames = shape[0];
            info.batch_size = shape[1];
            info.height = shape[2];
            info.width = shape[3];
            info.channels = shape[4];
            break;
            
        case VideoFormat::TBCHW:
            if (shape.size() != 5) throw std::invalid_argument("TBCHW format requires 5D tensor");
            info.num_frames = shape[0];
            info.batch_size = shape[1];
            info.channels = shape[2];
            info.height = shape[3];
            info.width = shape[4];
            break;
    }
    
    return info;
}

bool VideoTensorUtils::validate_video_tensor(const Tensor& video_tensor, VideoFormat format) {
    try {
        get_video_info(video_tensor, format);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<size_t> VideoTensorUtils::get_dimension_order(VideoFormat format) {
    switch (format) {
        case VideoFormat::BTHWC: return {0, 1, 2, 3, 4};  // No reordering needed
        case VideoFormat::BTCHW: return {0, 1, 4, 2, 3};  // Move channels to end
        case VideoFormat::TBHWC: return {1, 0, 2, 3, 4};  // Swap batch and time
        case VideoFormat::TBCHW: return {1, 0, 4, 2, 3};  // Swap batch/time, move channels
        default: return {0, 1, 2, 3, 4};
    }
}

std::vector<size_t> VideoTensorUtils::get_inverse_dimension_order(VideoFormat format) {
    switch (format) {
        case VideoFormat::BTHWC: return {0, 1, 2, 3, 4};
        case VideoFormat::BTCHW: return {0, 1, 3, 4, 2};
        case VideoFormat::TBHWC: return {1, 0, 2, 3, 4};
        case VideoFormat::TBCHW: return {1, 0, 3, 4, 2};
        default: return {0, 1, 2, 3, 4};
    }
}

// FrameInterpolation Implementation

Tensor FrameInterpolation::linear_interpolate(const Tensor& frame1, const Tensor& frame2, double alpha) {
    if (frame1.shape() != frame2.shape()) {
        throw std::invalid_argument("Frames must have the same shape for interpolation");
    }
    
    Tensor result(frame1.shape());
    const auto& data1 = frame1.data();
    const auto& data2 = frame2.data();
    auto& result_data = result.data();
    
    for (size_t i = 0; i < result.size(); ++i) {
        result_data[i] = (1.0 - alpha) * data1[i] + alpha * data2[i];
    }
    
    return result;
}

std::vector<Tensor> FrameInterpolation::interpolate_sequence(const Tensor& frame1, const Tensor& frame2, size_t num_intermediate_frames) {
    std::vector<Tensor> interpolated_frames;
    interpolated_frames.reserve(num_intermediate_frames);
    
    for (size_t i = 1; i <= num_intermediate_frames; ++i) {
        double alpha = static_cast<double>(i) / (num_intermediate_frames + 1);
        interpolated_frames.push_back(linear_interpolate(frame1, frame2, alpha));
    }
    
    return interpolated_frames;
}

Tensor FrameInterpolation::temporal_upsample(const Tensor& video_tensor, size_t scale_factor, VideoFormat format) {
    if (scale_factor <= 1) {
        return video_tensor;
    }
    
    auto frames = VideoTensorUtils::extract_frames(video_tensor, format);
    std::vector<Tensor> upsampled_frames;
    
    // Reserve space for upsampled frames
    size_t total_frames = frames.size() + (frames.size() - 1) * (scale_factor - 1);
    upsampled_frames.reserve(total_frames);
    
    for (size_t i = 0; i < frames.size(); ++i) {
        // Add original frame
        upsampled_frames.push_back(frames[i]);
        
        // Add interpolated frames (except after the last frame)
        if (i < frames.size() - 1) {
            auto interpolated = interpolate_sequence(frames[i], frames[i + 1], scale_factor - 1);
            for (const auto& frame : interpolated) {
                upsampled_frames.push_back(frame);
            }
        }
    }
    
    return VideoTensorUtils::create_video_tensor(upsampled_frames, format);
}

// VideoPreprocessor Implementation

VideoPreprocessor::VideoPreprocessor(const PreprocessConfig& config) : config_(config) {}

Tensor VideoPreprocessor::preprocess(const Tensor& video_tensor, VideoFormat input_format) {
    Tensor processed = video_tensor;
    
    // Convert to target format if needed
    if (input_format != config_.output_format) {
        processed = VideoTensorUtils::convert_format(processed, input_format, config_.output_format);
    }
    
    // Spatial resize
    processed = spatial_resize(processed, config_.output_format);
    
    // Temporal resample
    processed = temporal_resample(processed, config_.output_format);
    
    // Normalize
    if (config_.normalize) {
        processed = normalize_video(processed, config_.output_format);
    }
    
    return processed;
}

Tensor VideoPreprocessor::postprocess(const Tensor& model_output) {
    Tensor result = model_output;
    
    // Denormalize if normalization was applied
    if (config_.normalize) {
        result = denormalize_video(result, config_.output_format);
    }
    
    return result;
}

Tensor VideoPreprocessor::spatial_resize(const Tensor& video_tensor, VideoFormat format) {
    auto info = VideoTensorUtils::get_video_info(video_tensor, format);
    
    if (info.height == config_.target_height && info.width == config_.target_width) {
        return video_tensor;
    }
    
    // Extract frames, resize each, then reconstruct video
    auto frames = VideoTensorUtils::extract_frames(video_tensor, format);
    std::vector<Tensor> resized_frames;
    resized_frames.reserve(frames.size());
    
    for (const auto& frame : frames) {
        // Simple bilinear interpolation resize
        // For a more sophisticated implementation, you could use the image processing utilities
        Tensor resized_frame = resize_frame_bilinear(frame, config_.target_height, config_.target_width);
        resized_frames.push_back(resized_frame);
    }
    
    return VideoTensorUtils::create_video_tensor(resized_frames, format);
}

Tensor VideoPreprocessor::resize_frame_bilinear(const Tensor& frame, size_t target_height, size_t target_width) {
    const auto& shape = frame.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("Frame must be 3D [H, W, C]");
    }
    
    size_t src_height = shape[0];
    size_t src_width = shape[1];
    size_t channels = shape[2];
    
    if (src_height == target_height && src_width == target_width) {
        return frame;
    }
    
    Tensor resized({target_height, target_width, channels});
    const auto& src_data = frame.data();
    auto& dst_data = resized.data();
    
    double scale_y = static_cast<double>(src_height) / target_height;
    double scale_x = static_cast<double>(src_width) / target_width;
    
    for (size_t y = 0; y < target_height; ++y) {
        for (size_t x = 0; x < target_width; ++x) {
            // Calculate source coordinates
            double src_y = (y + 0.5) * scale_y - 0.5;
            double src_x = (x + 0.5) * scale_x - 0.5;
            
            // Clamp to source bounds
            src_y = std::max(0.0, std::min(src_y, static_cast<double>(src_height - 1)));
            src_x = std::max(0.0, std::min(src_x, static_cast<double>(src_width - 1)));
            
            int y0 = static_cast<int>(std::floor(src_y));
            int x0 = static_cast<int>(std::floor(src_x));
            int y1 = std::min(y0 + 1, static_cast<int>(src_height - 1));
            int x1 = std::min(x0 + 1, static_cast<int>(src_width - 1));
            
            double wy = src_y - y0;
            double wx = src_x - x0;
            
            for (size_t c = 0; c < channels; ++c) {
                // Bilinear interpolation
                double v00 = src_data[y0 * src_width * channels + x0 * channels + c];
                double v01 = src_data[y0 * src_width * channels + x1 * channels + c];
                double v10 = src_data[y1 * src_width * channels + x0 * channels + c];
                double v11 = src_data[y1 * src_width * channels + x1 * channels + c];
                
                double v0 = v00 * (1 - wx) + v01 * wx;
                double v1 = v10 * (1 - wx) + v11 * wx;
                double value = v0 * (1 - wy) + v1 * wy;
                
                dst_data[y * target_width * channels + x * channels + c] = value;
            }
        }
    }
    
    return resized;
}

Tensor VideoPreprocessor::temporal_resample(const Tensor& video_tensor, VideoFormat format) {
    auto info = VideoTensorUtils::get_video_info(video_tensor, format);
    
    if (info.num_frames != config_.target_frames) {
        return VideoTensorUtils::temporal_resize(video_tensor, config_.target_frames, format);
    }
    
    return video_tensor;
}

Tensor VideoPreprocessor::normalize_video(const Tensor& video_tensor, VideoFormat format) {
    auto info = VideoTensorUtils::get_video_info(video_tensor, format);
    
    if (config_.mean.size() != info.channels || config_.std.size() != info.channels) {
        throw std::invalid_argument("Mean and std vectors must match number of channels");
    }
    
    Tensor normalized = video_tensor;
    auto& data = normalized.data();
    
    // Apply normalization: (x - mean) / std for each channel
    switch (format) {
        case VideoFormat::BTHWC: {
            for (size_t b = 0; b < info.batch_size; ++b) {
                for (size_t t = 0; t < info.num_frames; ++t) {
                    for (size_t h = 0; h < info.height; ++h) {
                        for (size_t w = 0; w < info.width; ++w) {
                            for (size_t c = 0; c < info.channels; ++c) {
                                size_t idx = b * info.num_frames * info.height * info.width * info.channels +
                                           t * info.height * info.width * info.channels +
                                           h * info.width * info.channels +
                                           w * info.channels + c;
                                
                                data[idx] = (data[idx] - config_.mean[c]) / config_.std[c];
                            }
                        }
                    }
                }
            }
            break;
        }
        case VideoFormat::BTCHW: {
            for (size_t b = 0; b < info.batch_size; ++b) {
                for (size_t t = 0; t < info.num_frames; ++t) {
                    for (size_t c = 0; c < info.channels; ++c) {
                        for (size_t h = 0; h < info.height; ++h) {
                            for (size_t w = 0; w < info.width; ++w) {
                                size_t idx = b * info.num_frames * info.channels * info.height * info.width +
                                           t * info.channels * info.height * info.width +
                                           c * info.height * info.width +
                                           h * info.width + w;
                                
                                data[idx] = (data[idx] - config_.mean[c]) / config_.std[c];
                            }
                        }
                    }
                }
            }
            break;
        }
        default:
            // For TBHWC and TBCHW formats, similar logic with adjusted indexing
            throw std::runtime_error("Normalization for this format not yet implemented");
    }
    
    return normalized;
}

Tensor VideoPreprocessor::denormalize_video(const Tensor& video_tensor, VideoFormat format) {
    auto info = VideoTensorUtils::get_video_info(video_tensor, format);
    
    if (config_.mean.size() != info.channels || config_.std.size() != info.channels) {
        throw std::invalid_argument("Mean and std vectors must match number of channels");
    }
    
    Tensor denormalized = video_tensor;
    auto& data = denormalized.data();
    
    // Apply denormalization: x * std + mean for each channel
    switch (format) {
        case VideoFormat::BTHWC: {
            for (size_t b = 0; b < info.batch_size; ++b) {
                for (size_t t = 0; t < info.num_frames; ++t) {
                    for (size_t h = 0; h < info.height; ++h) {
                        for (size_t w = 0; w < info.width; ++w) {
                            for (size_t c = 0; c < info.channels; ++c) {
                                size_t idx = b * info.num_frames * info.height * info.width * info.channels +
                                           t * info.height * info.width * info.channels +
                                           h * info.width * info.channels +
                                           w * info.channels + c;
                                
                                data[idx] = data[idx] * config_.std[c] + config_.mean[c];
                            }
                        }
                    }
                }
            }
            break;
        }
        case VideoFormat::BTCHW: {
            for (size_t b = 0; b < info.batch_size; ++b) {
                for (size_t t = 0; t < info.num_frames; ++t) {
                    for (size_t c = 0; c < info.channels; ++c) {
                        for (size_t h = 0; h < info.height; ++h) {
                            for (size_t w = 0; w < info.width; ++w) {
                                size_t idx = b * info.num_frames * info.channels * info.height * info.width +
                                           t * info.channels * info.height * info.width +
                                           c * info.height * info.width +
                                           h * info.width + w;
                                
                                data[idx] = data[idx] * config_.std[c] + config_.mean[c];
                            }
                        }
                    }
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Denormalization for this format not yet implemented");
    }
    
    return denormalized;
}

// TemporalConvolution Implementation

// Conv3DLayer Implementation
TemporalConvolution::Conv3DLayer::Conv3DLayer(const Conv3DConfig& config) : config_(config) {
    initialize_tensors();
}

void TemporalConvolution::Conv3DLayer::initialize_tensors() {
    // Weights shape: [out_channels, in_channels, kernel_t, kernel_h, kernel_w]
    std::vector<size_t> weight_shape = {
        config_.out_channels,
        config_.in_channels,
        config_.kernel_size[0],  // temporal
        config_.kernel_size[1],  // height
        config_.kernel_size[2]   // width
    };
    
    weights_ = Tensor(weight_shape);
    
    if (config_.use_bias) {
        bias_ = Tensor({config_.out_channels});
    }
    
    initialize_weights("xavier");
}

void TemporalConvolution::Conv3DLayer::initialize_weights(const std::string& init_type) {
    auto& weight_data = weights_.data();
    size_t fan_in = config_.in_channels * config_.kernel_size[0] * config_.kernel_size[1] * config_.kernel_size[2];
    size_t fan_out = config_.out_channels * config_.kernel_size[0] * config_.kernel_size[1] * config_.kernel_size[2];
    
    if (init_type == "xavier") {
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        std::uniform_real_distribution<double> dist(-limit, limit);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (auto& w : weight_data) {
            w = dist(gen);
        }
    } else if (init_type == "he") {
        double std_dev = std::sqrt(2.0 / fan_in);
        std::normal_distribution<double> dist(0.0, std_dev);
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (auto& w : weight_data) {
            w = dist(gen);
        }
    }
    
    if (config_.use_bias) {
        auto& bias_data = bias_.data();
        std::fill(bias_data.begin(), bias_data.end(), 0.0);
    }
}

Tensor TemporalConvolution::Conv3DLayer::forward(const Tensor& input) {
    return apply_conv3d(input);
}

Tensor TemporalConvolution::Conv3DLayer::apply_conv3d(const Tensor& input) const {
    // Input: [B, T, C, H, W] (BTCHW format)
    const auto& input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("Conv3D input must be 5D [B, T, C, H, W]");
    }
    
    size_t batch_size = input_shape[0];
    size_t input_time = input_shape[1];
    size_t input_channels = input_shape[2];
    size_t input_height = input_shape[3];
    size_t input_width = input_shape[4];
    
    if (input_channels != config_.in_channels) {
        throw std::invalid_argument("Input channels mismatch");
    }
    
    // Calculate output dimensions
    size_t output_time = (input_time + 2 * config_.padding[0] - config_.kernel_size[0]) / config_.stride[0] + 1;
    size_t output_height = (input_height + 2 * config_.padding[1] - config_.kernel_size[1]) / config_.stride[1] + 1;
    size_t output_width = (input_width + 2 * config_.padding[2] - config_.kernel_size[2]) / config_.stride[2] + 1;
    
    std::vector<size_t> output_shape = {batch_size, output_time, config_.out_channels, output_height, output_width};
    Tensor output(output_shape);
    
    const auto& input_data = input.data();
    auto& output_data = output.data();
    const auto& weight_data = weights_.data();
    const auto& bias_data = config_.use_bias ? bias_.data() : std::vector<double>();
    
    // Perform 3D convolution
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < config_.out_channels; ++oc) {
            for (size_t ot = 0; ot < output_time; ++ot) {
                for (size_t oh = 0; oh < output_height; ++oh) {
                    for (size_t ow = 0; ow < output_width; ++ow) {
                        double sum = config_.use_bias ? bias_data[oc] : 0.0;
                        
                        // Convolve over input channels and kernel
                        for (size_t ic = 0; ic < config_.in_channels; ++ic) {
                            for (size_t kt = 0; kt < config_.kernel_size[0]; ++kt) {
                                for (size_t kh = 0; kh < config_.kernel_size[1]; ++kh) {
                                    for (size_t kw = 0; kw < config_.kernel_size[2]; ++kw) {
                                        int it = static_cast<int>(ot * config_.stride[0] + kt) - static_cast<int>(config_.padding[0]);
                                        int ih = static_cast<int>(oh * config_.stride[1] + kh) - static_cast<int>(config_.padding[1]);
                                        int iw = static_cast<int>(ow * config_.stride[2] + kw) - static_cast<int>(config_.padding[2]);
                                        
                                        if (it >= 0 && it < static_cast<int>(input_time) &&
                                            ih >= 0 && ih < static_cast<int>(input_height) &&
                                            iw >= 0 && iw < static_cast<int>(input_width)) {
                                            
                                            size_t input_idx = b * input_time * input_channels * input_height * input_width +
                                                              it * input_channels * input_height * input_width +
                                                              ic * input_height * input_width +
                                                              ih * input_width + iw;
                                            
                                            size_t weight_idx = oc * config_.in_channels * config_.kernel_size[0] * config_.kernel_size[1] * config_.kernel_size[2] +
                                                               ic * config_.kernel_size[0] * config_.kernel_size[1] * config_.kernel_size[2] +
                                                               kt * config_.kernel_size[1] * config_.kernel_size[2] +
                                                               kh * config_.kernel_size[2] + kw;
                                            
                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        
                        size_t output_idx = b * output_time * config_.out_channels * output_height * output_width +
                                           ot * config_.out_channels * output_height * output_width +
                                           oc * output_height * output_width +
                                           oh * output_width + ow;
                        
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
    }
    
    return output;
}

size_t TemporalConvolution::Conv3DLayer::get_param_count() const {
    size_t weight_params = config_.out_channels * config_.in_channels * 
                          config_.kernel_size[0] * config_.kernel_size[1] * config_.kernel_size[2];
    size_t bias_params = config_.use_bias ? config_.out_channels : 0;
    return weight_params + bias_params;
}

// Conv2Plus1DLayer Implementation
TemporalConvolution::Conv2Plus1DLayer::Conv2Plus1DLayer(const Conv3DConfig& config) : config_(config) {
    // For now, use simplified implementation
    // TODO: Implement proper 2+1D convolution decomposition with separate spatial and temporal convolutions
}

Tensor TemporalConvolution::Conv2Plus1DLayer::forward(const Tensor& input) {
    // Simplified implementation: apply 3D convolution for now
    // TODO: Implement proper 2+1D decomposition
    Conv3DLayer conv3d(config_);
    return conv3d.forward(input);
}

size_t TemporalConvolution::Conv2Plus1DLayer::get_param_count() const {
    // Simplified calculation - proper 2+1D would have fewer parameters
    return Conv3DLayer(config_).get_param_count();
}

void TemporalConvolution::Conv2Plus1DLayer::initialize_weights(const std::string& init_type) {
    // TODO: Initialize separate 2D and 1D convolution weights
}

// TemporalAttention Implementation

// TemporalSelfAttention Implementation
TemporalAttention::TemporalSelfAttention::TemporalSelfAttention(const TemporalAttentionConfig& config) 
    : config_(config) {
    
    // Create the underlying attention layer
    attention_layer_ = std::make_unique<MultiHeadAttentionLayer>(
        config_.embed_dim, 
        config_.num_heads, 
        config_.max_sequence_length
    );
    
    attention_layer_->set_dropout_rate(config_.dropout_rate);
    
    if (config_.use_positional_encoding) {
        initialize_positional_encoding();
    }
}

void TemporalAttention::TemporalSelfAttention::initialize_positional_encoding() {
    // Sinusoidal positional encoding
    positional_encoding_ = Tensor({config_.max_sequence_length, config_.embed_dim});
    auto& pos_data = positional_encoding_.data();
    
    for (size_t pos = 0; pos < config_.max_sequence_length; ++pos) {
        for (size_t i = 0; i < config_.embed_dim; ++i) {
            double angle = pos / std::pow(10000.0, 2.0 * (i / 2) / config_.embed_dim);
            
            size_t idx = pos * config_.embed_dim + i;
            
            if (i % 2 == 0) {
                pos_data[idx] = std::sin(angle);
            } else {
                pos_data[idx] = std::cos(angle);
            }
        }
    }
}

Tensor TemporalAttention::TemporalSelfAttention::apply_spatial_pooling(const Tensor& video_tensor) {
    // Input: [B, T, H, W, C] -> Output: [B, T, C]
    const auto& shape = video_tensor.shape();
    if (shape.size() != 5) {
        throw std::invalid_argument("Expected 5D video tensor [B, T, H, W, C]");
    }
    
    size_t batch_size = shape[0];
    size_t num_frames = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    size_t channels = shape[4];
    
    // Global average pooling over spatial dimensions
    Tensor pooled({batch_size, num_frames, channels});
    const auto& input_data = video_tensor.data();
    auto& pooled_data = pooled.data();
    
    size_t spatial_size = height * width;
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < num_frames; ++t) {
            for (size_t c = 0; c < channels; ++c) {
                double sum = 0.0;
                
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t input_idx = b * num_frames * height * width * channels +
                                          t * height * width * channels +
                                          h * width * channels +
                                          w * channels + c;
                        sum += input_data[input_idx];
                    }
                }
                
                size_t pooled_idx = b * num_frames * channels + t * channels + c;
                pooled_data[pooled_idx] = sum / spatial_size;
            }
        }
    }
    
    return pooled;
}

Tensor TemporalAttention::TemporalSelfAttention::forward(const Tensor& video_features) {
    // Apply spatial pooling if input is 5D (has spatial dimensions)
    Tensor temporal_features;
    
    if (video_features.shape().size() == 5) {
        temporal_features = apply_spatial_pooling(video_features);
    } else if (video_features.shape().size() == 3) {
        temporal_features = video_features;  // Already in [B, T, C] format
    } else {
        throw std::invalid_argument("Input must be 5D [B, T, H, W, C] or 3D [B, T, C]");
    }
    
    const auto& shape = temporal_features.shape();
    size_t batch_size = shape[0];
    size_t sequence_length = shape[1];
    size_t embed_dim = shape[2];
    
    if (embed_dim != config_.embed_dim) {
        throw std::invalid_argument("Feature dimension mismatch");
    }
    
    // Add positional encoding if enabled
    if (config_.use_positional_encoding && sequence_length <= config_.max_sequence_length) {
        auto& temp_data = temporal_features.data();
        const auto& pos_data = positional_encoding_.data();
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t t = 0; t < sequence_length; ++t) {
                for (size_t d = 0; d < embed_dim; ++d) {
                    size_t feat_idx = b * sequence_length * embed_dim + t * embed_dim + d;
                    size_t pos_idx = t * embed_dim + d;
                    temp_data[feat_idx] += pos_data[pos_idx];
                }
            }
        }
    }
    
    // Apply self-attention
    Tensor attended = attention_layer_->forward_tensor_self_attention(temporal_features);
    
    // Store attention weights for visualization
    attention_weights_ = attention_layer_->get_last_attention_weights();
    
    return attended;
}

// CrossTemporalAttention Implementation
TemporalAttention::CrossTemporalAttention::CrossTemporalAttention(const TemporalAttentionConfig& config) 
    : config_(config) {
    
    // Create cross-modal attention layer (we'll adapt it for cross-temporal use)
    cross_attention_ = std::make_unique<CrossModalAttention>(
        config_.embed_dim, 
        config_.embed_dim,
        config_.num_heads
    );
}

Tensor TemporalAttention::CrossTemporalAttention::forward(const Tensor& query_frames, const Tensor& key_value_frames) {
    // Input: [B, T1, C] and [B, T2, C]
    if (query_frames.shape().size() != 3 || key_value_frames.shape().size() != 3) {
        throw std::invalid_argument("Inputs must be 3D [B, T, C]");
    }
    
    // Apply cross-temporal attention using the correct API
    auto result = cross_attention_->forward(key_value_frames, query_frames, 
                                           Modality::VIDEO, Modality::VIDEO);
    
    return result;
}

// MotionEstimation Implementation

Tensor MotionEstimation::estimate_optical_flow(const Tensor& frame1, const Tensor& frame2, size_t block_size) {
    // Simple block-matching optical flow estimation
    const auto& shape1 = frame1.shape();
    const auto& shape2 = frame2.shape();
    
    if (shape1 != shape2) {
        throw std::invalid_argument("Frames must have the same shape");
    }
    
    if (shape1.size() != 3) {
        throw std::invalid_argument("Frames must be 3D [H, W, C]");
    }
    
    size_t height = shape1[0];
    size_t width = shape1[1];
    size_t channels = shape1[2];
    
    // Output motion vectors [H/block_size, W/block_size, 2]
    size_t output_height = height / block_size;
    size_t output_width = width / block_size;
    
    Tensor motion_vectors({output_height, output_width, 2});
    
    const auto& data1 = frame1.data();
    const auto& data2 = frame2.data();
    auto& motion_data = motion_vectors.data();
    
    // Search window size
    int search_range = static_cast<int>(block_size);
    
    for (size_t by = 0; by < output_height; ++by) {
        for (size_t bx = 0; bx < output_width; ++bx) {
            double best_sad = std::numeric_limits<double>::max();
            int best_dx = 0, best_dy = 0;
            
            // Search in neighborhood
            for (int dy = -search_range; dy <= search_range; ++dy) {
                for (int dx = -search_range; dx <= search_range; ++dx) {
                    double sad = 0.0;
                    bool valid = true;
                    
                    // Calculate SAD for this displacement
                    for (size_t py = 0; py < block_size && valid; ++py) {
                        for (size_t px = 0; px < block_size && valid; ++px) {
                            int y1 = static_cast<int>(by * block_size + py);
                            int x1 = static_cast<int>(bx * block_size + px);
                            int y2 = y1 + dy;
                            int x2 = x1 + dx;
                            
                            if (y2 < 0 || y2 >= static_cast<int>(height) ||
                                x2 < 0 || x2 >= static_cast<int>(width)) {
                                valid = false;
                                continue;
                            }
                            
                            for (size_t c = 0; c < channels; ++c) {
                                size_t idx1 = y1 * width * channels + x1 * channels + c;
                                size_t idx2 = y2 * width * channels + x2 * channels + c;
                                sad += std::abs(data1[idx1] - data2[idx2]);
                            }
                        }
                    }
                    
                    if (valid && sad < best_sad) {
                        best_sad = sad;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
            
            // Store motion vector
            size_t motion_idx = by * output_width * 2 + bx * 2;
            motion_data[motion_idx] = static_cast<double>(best_dx);
            motion_data[motion_idx + 1] = static_cast<double>(best_dy);
        }
    }
    
    return motion_vectors;
}

Tensor MotionEstimation::warp_frame(const Tensor& frame, const Tensor& motion_vectors) {
    // Warp frame using motion vectors
    const auto& frame_shape = frame.shape();
    const auto& motion_shape = motion_vectors.shape();
    
    if (frame_shape.size() != 3) {
        throw std::invalid_argument("Frame must be 3D [H, W, C]");
    }
    
    if (motion_shape.size() != 3 || motion_shape[2] != 2) {
        throw std::invalid_argument("Motion vectors must be 3D [H, W, 2]");
    }
    
    size_t height = frame_shape[0];
    size_t width = frame_shape[1];
    size_t channels = frame_shape[2];
    
    Tensor warped_frame(frame_shape);
    
    const auto& frame_data = frame.data();
    const auto& motion_data = motion_vectors.data();
    auto& warped_data = warped_frame.data();
    
    // Calculate scale factors between motion grid and frame
    double scale_y = static_cast<double>(motion_shape[0]) / height;
    double scale_x = static_cast<double>(motion_shape[1]) / width;
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            // Find corresponding motion vector
            size_t my = static_cast<size_t>(y * scale_y);
            size_t mx = static_cast<size_t>(x * scale_x);
            
            if (my >= motion_shape[0]) my = motion_shape[0] - 1;
            if (mx >= motion_shape[1]) mx = motion_shape[1] - 1;
            
            size_t motion_idx = my * motion_shape[1] * 2 + mx * 2;
            double dx = motion_data[motion_idx];
            double dy = motion_data[motion_idx + 1];
            
            // Source coordinates
            int src_x = static_cast<int>(x) + static_cast<int>(dx);
            int src_y = static_cast<int>(y) + static_cast<int>(dy);
            
            for (size_t c = 0; c < channels; ++c) {
                size_t dst_idx = y * width * channels + x * channels + c;
                
                if (src_x >= 0 && src_x < static_cast<int>(width) &&
                    src_y >= 0 && src_y < static_cast<int>(height)) {
                    size_t src_idx = src_y * width * channels + src_x * channels + c;
                    warped_data[dst_idx] = frame_data[src_idx];
                } else {
                    // Out of bounds - set to zero or use boundary conditions
                    warped_data[dst_idx] = 0.0;
                }
            }
        }
    }
    
    return warped_frame;
}

Tensor MotionEstimation::estimate_video_motion(const Tensor& video_tensor, VideoFormat format) {
    auto frames = VideoTensorUtils::extract_frames(video_tensor, format);
    
    if (frames.size() < 2) {
        throw std::invalid_argument("Video must have at least 2 frames for motion estimation");
    }
    
    std::vector<Tensor> motion_fields;
    motion_fields.reserve(frames.size() - 1);
    
    // Estimate motion between consecutive frames
    for (size_t i = 0; i < frames.size() - 1; ++i) {
        auto motion = estimate_optical_flow(frames[i], frames[i + 1]);
        motion_fields.push_back(motion);
    }
    
    // Combine motion fields into single tensor
    auto info = VideoTensorUtils::get_video_info(video_tensor, format);
    const auto& motion_shape = motion_fields[0].shape();
    
    // Output: [B, T-1, H, W, 2]
    std::vector<size_t> output_shape = {
        info.batch_size, 
        frames.size() - 1, 
        motion_shape[0], 
        motion_shape[1], 
        2
    };
    
    Tensor video_motion(output_shape);
    auto& video_motion_data = video_motion.data();
    
    size_t motion_frame_size = motion_shape[0] * motion_shape[1] * 2;
    
    for (size_t t = 0; t < motion_fields.size(); ++t) {
        const auto& motion_data = motion_fields[t].data();
        
        // Copy motion data for all batches (assuming single batch for now)
        for (size_t b = 0; b < info.batch_size; ++b) {
            size_t offset = b * (frames.size() - 1) * motion_frame_size + t * motion_frame_size;
            std::copy(motion_data.begin(), motion_data.end(), 
                     video_motion_data.begin() + offset);
        }
    }
      return video_motion;
}

} // namespace ai
} // namespace asekioml
