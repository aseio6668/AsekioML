#include "ai/video_diffusion.hpp"
#include "ai/cnn_layers.hpp"
#include "ai/attention_layers.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {
namespace ai {

// ===== VideoDiffusionModel Implementation =====

std::vector<std::vector<double>> VideoDiffusionModel::create_temporal_schedule(
    int num_steps, 
    int num_frames,
    MotionSchedule motion_schedule,
    double beta_start, 
    double beta_end
) {
    std::vector<std::vector<double>> temporal_betas(num_steps, std::vector<double>(num_frames));
    
    // Base linear schedule
    for (int t = 0; t < num_steps; ++t) {
        double base_beta = beta_start + (beta_end - beta_start) * t / (num_steps - 1);
        
        for (int f = 0; f < num_frames; ++f) {
            switch (motion_schedule) {
                case MotionSchedule::UNIFORM:
                    temporal_betas[t][f] = base_beta;
                    break;
                    
                case MotionSchedule::MOTION_ADAPTIVE:
                    // Slightly higher noise in middle frames (more motion expected)
                    {
                        double frame_factor = 1.0 + 0.2 * std::sin(M_PI * f / (num_frames - 1));
                        temporal_betas[t][f] = base_beta * frame_factor;
                    }
                    break;
                    
                case MotionSchedule::TEMPORAL_DECAY:
                    // Decay noise influence over time (later frames less noisy)
                    {
                        double decay_factor = 1.0 - 0.3 * f / (num_frames - 1);
                        temporal_betas[t][f] = base_beta * decay_factor;
                    }
                    break;
            }
        }
    }
    
    return temporal_betas;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
VideoDiffusionModel::compute_temporal_alphas(const std::vector<std::vector<double>>& temporal_betas) {
    int num_steps = temporal_betas.size();
    int num_frames = temporal_betas[0].size();
    
    std::vector<std::vector<double>> temporal_alphas(num_steps, std::vector<double>(num_frames));
    std::vector<std::vector<double>> temporal_alpha_cumprod(num_steps, std::vector<double>(num_frames));
    
    for (int f = 0; f < num_frames; ++f) {
        double cumprod = 1.0;
        for (int t = 0; t < num_steps; ++t) {
            temporal_alphas[t][f] = 1.0 - temporal_betas[t][f];
            cumprod *= temporal_alphas[t][f];
            temporal_alpha_cumprod[t][f] = cumprod;
        }
    }
    
    return {temporal_alphas, temporal_alpha_cumprod};
}

Tensor VideoDiffusionModel::add_temporal_noise(
    const Tensor& x0, 
    int t, 
    const std::vector<std::vector<double>>& temporal_alpha_cumprod,
    VideoFormat format
) {
    auto info = VideoTensorUtils::get_tensor_info(x0, format);
    
    // Generate temporal noise
    Tensor noise = sample_temporal_noise(x0.shape(), format);
    
    // Apply noise with frame-specific alpha values
    Tensor noisy_video = x0;
    auto x0_data = x0.data();
    auto noise_data = noise.data();
    auto result_data = noisy_video.data();
    
    size_t frame_size = info.height * info.width * info.channels;
    
    for (size_t b = 0; b < info.batch_size; ++b) {
        for (size_t f = 0; f < info.num_frames; ++f) {
            double alpha_cumprod = temporal_alpha_cumprod[t][f];
            double sqrt_alpha_cumprod = std::sqrt(alpha_cumprod);
            double sqrt_one_minus_alpha_cumprod = std::sqrt(1.0 - alpha_cumprod);
            
            size_t frame_offset;
            if (format == VideoFormat::BTCHW) {
                frame_offset = b * info.num_frames * info.channels * info.height * info.width +
                              f * info.channels * info.height * info.width;
            } else { // BTHWC
                frame_offset = b * info.num_frames * info.height * info.width * info.channels +
                              f * info.height * info.width * info.channels;
            }
            
            for (size_t i = 0; i < frame_size; ++i) {
                size_t idx = frame_offset + i;
                result_data[idx] = sqrt_alpha_cumprod * x0_data[idx] + 
                                  sqrt_one_minus_alpha_cumprod * noise_data[idx];
            }
        }
    }
    
    return noisy_video;
}

Tensor VideoDiffusionModel::sample_temporal_noise(
    const std::vector<size_t>& video_shape,
    VideoFormat format
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    Tensor noise(video_shape);
    auto data = noise.data();
    
    // Add slight temporal correlation to noise
    size_t total_size = noise.size();
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dist(gen);
    }
    
    // Apply temporal smoothing for more realistic noise
    if (video_shape.size() >= 5) { // [B, T, C, H, W] or similar
        size_t batch_size = video_shape[0];
        size_t num_frames = (format == VideoFormat::BTCHW) ? video_shape[1] : video_shape[1];
        
        // Light temporal smoothing
        VideoMotionUtils::temporal_smoothing(noise, 0.1, format);
    }
    
    return noise;
}

Tensor VideoDiffusionModel::temporal_denoise_step(
    const Tensor& x_t,
    const Tensor& predicted_noise,
    int t,
    const std::vector<std::vector<double>>& temporal_alphas,
    const std::vector<std::vector<double>>& temporal_alpha_cumprod,
    const std::vector<std::vector<double>>& temporal_betas,
    VideoFormat format
) {
    auto info = VideoTensorUtils::get_tensor_info(x_t, format);
    
    Tensor x_prev = x_t;
    auto x_t_data = x_t.data();
    auto noise_data = predicted_noise.data();
    auto result_data = x_prev.data();
    
    size_t frame_size = info.height * info.width * info.channels;
    
    for (size_t b = 0; b < info.batch_size; ++b) {
        for (size_t f = 0; f < info.num_frames; ++f) {
            if (t >= temporal_alphas.size() || f >= temporal_alphas[t].size()) continue;
            
            double alpha_t = temporal_alphas[t][f];
            double alpha_cumprod_t = temporal_alpha_cumprod[t][f];
            double alpha_cumprod_prev = (t > 0) ? temporal_alpha_cumprod[t-1][f] : 1.0;
            double beta_t = temporal_betas[t][f];
            
            // DDPM denoising formula per frame
            double coeff1 = 1.0 / std::sqrt(alpha_t);
            double coeff2 = beta_t / std::sqrt(1.0 - alpha_cumprod_t);
            
            size_t frame_offset;
            if (format == VideoFormat::BTCHW) {
                frame_offset = b * info.num_frames * info.channels * info.height * info.width +
                              f * info.channels * info.height * info.width;
            } else { // BTHWC
                frame_offset = b * info.num_frames * info.height * info.width * info.channels +
                              f * info.height * info.width * info.channels;
            }
            
            for (size_t i = 0; i < frame_size; ++i) {
                size_t idx = frame_offset + i;
                result_data[idx] = coeff1 * (x_t_data[idx] - coeff2 * noise_data[idx]);
            }
        }
    }
    
    // Add temporal smoothing for consistency
    return VideoMotionUtils::temporal_smoothing(x_prev, 0.2, format);
}

Tensor VideoDiffusionModel::temporal_sample_loop(
    const Tensor& x_t,
    const Tensor& text_embeddings,
    int num_steps,
    GenerationStrategy strategy,
    double guidance_scale,
    VideoFormat format
) {
    Tensor x = x_t;
    auto info = VideoTensorUtils::get_tensor_info(x, format);
    
    // Create temporal schedules
    auto temporal_betas = create_temporal_schedule(num_steps, info.num_frames, MotionSchedule::MOTION_ADAPTIVE);
    auto [temporal_alphas, temporal_alpha_cumprod] = compute_temporal_alphas(temporal_betas);
    
    // Create temporal U-Net for this generation
    TemporalUNet temporal_unet(info.channels, info.channels, 512, text_embeddings.shape()[2], info.num_frames);
    
    for (int t = num_steps - 1; t >= 0; --t) {
        // Predict noise with temporal U-Net
        Tensor predicted_noise;
        
        switch (strategy) {
            case GenerationStrategy::FRAME_BY_FRAME:
                {
                    // Process each frame independently
                    auto frames = VideoTensorUtils::extract_frames(x, format);
                    std::vector<Tensor> denoised_frames;
                    
                    for (const auto& frame : frames) {                        // Use 2D diffusion model for each frame (placeholder)
                        // auto frame_noise = DiffusionModel::simple_unet(frame, t, text_embeddings);
                        Tensor frame_noise = frame; // Simplified placeholder
                        denoised_frames.push_back(frame_noise);
                    }
                    
                    predicted_noise = VideoTensorUtils::create_video_tensor(denoised_frames, format);
                }
                break;
                
            case GenerationStrategy::VIDEO_CHUNK:
            case GenerationStrategy::TEMPORAL_AWARE:
                // Use temporal U-Net for full video processing
                predicted_noise = temporal_unet.forward(x, t, text_embeddings, format);
                break;
        }
        
        // Apply temporal denoising step
        x = temporal_denoise_step(x, predicted_noise, t, temporal_alphas, temporal_alpha_cumprod, temporal_betas, format);
          if (t % 10 == 0) {
            std::cout << "Temporal denoising step " << t << " completed" << std::endl;
        }
    }
    
    return x;
}

// ===== TemporalUNet Implementation =====

TemporalUNet::TemporalUNet(size_t in_channels, size_t out_channels, size_t time_embed_dim, 
                           size_t text_embed_dim, size_t num_frames) 
    : in_channels_(in_channels), out_channels_(out_channels), time_embed_dim_(time_embed_dim),
      text_embed_dim_(text_embed_dim), num_frames_(num_frames) {
}

Tensor TemporalUNet::forward(const Tensor& x_t, int t, const Tensor& text_embeddings, VideoFormat format) {
    // Convert to BTCHW format for processing
    Tensor input = (format == VideoFormat::BTCHW) ? x_t : 
                   VideoTensorUtils::convert_format(x_t, format, VideoFormat::BTCHW);
    
    auto info = VideoTensorUtils::get_tensor_info(input, VideoFormat::BTCHW);
    
    // Time embedding
    Tensor time_emb = time_embedding(t, time_embed_dim_);
    
    // Encoder path with temporal processing
    Tensor h1 = conv3d_block(input, 64);
    h1 = temporal_attention_block(h1, 4);
    
    Tensor h2 = conv3d_block(h1, 128);
    h2 = temporal_attention_block(h2, 4);
    
    Tensor h3 = conv3d_block(h2, 256);
    h3 = temporal_attention_block(h3, 8);
    
    // Bottleneck with cross-attention to text
    Tensor bottleneck = conv3d_block(h3, 512);
    bottleneck = temporal_attention_block(bottleneck, 8);
    
    // Simplified cross-attention with text (reshape for attention)
    size_t B = info.batch_size, T = info.num_frames, C = 512, H = info.height / 8, W = info.width / 8;    Tensor bottleneck_flat({B * T, C, H * W});
    // Copy data (simplified reshaping)
    auto bn_data = bottleneck.data();
    auto flat_data = bottleneck_flat.data();
    std::copy(bn_data.begin(), bn_data.end(), flat_data.begin());
    
    // Cross-attention placeholder (simplified)
    // In real implementation, this would be proper cross-attention
    
    // Decoder path
    Tensor d3 = conv3d_block(bottleneck, 256);
    d3 = temporal_attention_block(d3, 8);
    
    Tensor d2 = conv3d_block(d3, 128); 
    d2 = temporal_attention_block(d2, 4);
    
    Tensor d1 = conv3d_block(d2, 64);
    d1 = temporal_attention_block(d1, 4);
    
    // Output layer
    Tensor output = conv3d_block(d1, out_channels_);
    
    // Convert back to original format if needed
    return (format == VideoFormat::BTCHW) ? output :
           VideoTensorUtils::convert_format(output, VideoFormat::BTCHW, format);
}

Tensor TemporalUNet::conv3d_block(const Tensor& input, size_t out_channels, size_t kernel_size, 
                                  size_t stride, size_t padding) {
    // Simplified 3D convolution using existing Conv2D operations
    // In real implementation, this would be proper 3D convolution
    
    auto info = VideoTensorUtils::get_tensor_info(input, VideoFormat::BTCHW);
    
    // Process each frame with 2D convolution
    auto frames = VideoTensorUtils::extract_frames(input, VideoFormat::BTCHW);
    std::vector<Tensor> conv_frames;
      for (const auto& frame : frames) {        // Create Conv2D layer with correct parameter order
        Conv2DLayer conv_layer(frame.shape()[0], out_channels, kernel_size, stride, padding);
        conv_layer.set_input_dimensions(frame.shape()[1], frame.shape()[2]); // height, width
        
        // Apply convolution
        Tensor conv_frame = conv_layer.forward_tensor(frame);
        conv_frames.push_back(conv_frame);
    }
    
    // Reconstruct video tensor
    Tensor result = VideoTensorUtils::create_video_tensor(conv_frames, VideoFormat::BTCHW);
    
    // Apply temporal convolution for temporal consistency
    // Simplified temporal filtering
    return VideoMotionUtils::temporal_smoothing(result, 0.1, VideoFormat::BTCHW);
}

Tensor TemporalUNet::temporal_attention_block(const Tensor& input, size_t num_heads) {
    // Simplified temporal attention
    // In real implementation, this would be proper temporal self-attention
    
    auto info = VideoTensorUtils::get_tensor_info(input, VideoFormat::BTCHW);
    
    // Reshape for temporal attention: [B, T, C*H*W]
    size_t B = info.batch_size, T = info.num_frames;
    size_t spatial_dim = info.channels * info.height * info.width;
    
    Tensor temporal_input({B, T, spatial_dim});    auto input_data = input.data();
    auto temp_data = temporal_input.data();
    
    // Copy data with proper indexing
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            size_t input_offset = b * T * spatial_dim + t * spatial_dim;
            size_t temp_offset = b * T * spatial_dim + t * spatial_dim;
            // Copy spatial_dim elements from input to temporal_input
            for (size_t i = 0; i < spatial_dim; ++i) {
                if (input_offset + i < input_data.size() && temp_offset + i < temp_data.size()) {
                    temp_data[temp_offset + i] = input_data[input_offset + i];
                }
            }
        }
    }// Apply simplified attention across temporal dimension
    MultiHeadAttentionLayer attention(spatial_dim, num_heads, T);
    
    Tensor attended = temporal_input; // Placeholder for attention output
      // Reshape back to video format
    Tensor result = input; // Start with input shape
    auto result_data = result.data();
    auto attended_data = attended.data();
    
    std::copy(attended_data.begin(), attended_data.end(), result_data.begin());
    
    return result;
}

Tensor TemporalUNet::temporal_residual_block(const Tensor& input, size_t channels) {
    Tensor conv1 = conv3d_block(input, channels);
    Tensor conv2 = conv3d_block(conv1, channels);
    
    // Residual connection
    Tensor result = input;
    auto input_data = input.data();
    auto conv2_data = conv2.data();
    auto result_data = result.data();
    
    for (size_t i = 0; i < input.size(); ++i) {
        result_data[i] = input_data[i] + conv2_data[i];
    }
    
    return result;
}

Tensor TemporalUNet::time_embedding(int t, size_t embed_dim) {
    Tensor embedding({embed_dim});
    auto data = embedding.data();
    
    // Sinusoidal time embedding
    for (size_t i = 0; i < embed_dim; ++i) {
        if (i % 2 == 0) {
            data[i] = std::sin(t / std::pow(10000.0, (double)i / embed_dim));
        } else {
            data[i] = std::cos(t / std::pow(10000.0, (double)(i-1) / embed_dim));
        }
    }
    
    return embedding;
}

Tensor TemporalUNet::temporal_positional_encoding(size_t num_frames, size_t embed_dim) {
    Tensor encoding({num_frames, embed_dim});
    auto data = encoding.data();
    
    for (size_t pos = 0; pos < num_frames; ++pos) {
        for (size_t i = 0; i < embed_dim; ++i) {
            size_t idx = pos * embed_dim + i;
            if (i % 2 == 0) {
                data[idx] = std::sin(pos / std::pow(10000.0, (double)i / embed_dim));
            } else {
                data[idx] = std::cos(pos / std::pow(10000.0, (double)(i-1) / embed_dim));
            }
        }
    }
    
    return encoding;
}

// ===== TextToVideoPipeline Implementation =====

TextToVideoPipeline::TextToVideoPipeline(size_t video_size, size_t num_frames, int num_steps,
                                         VideoDiffusionModel::GenerationStrategy strategy)
    : video_size_(video_size), num_frames_(num_frames), num_steps_(num_steps), strategy_(strategy) {
    
    temporal_unet_ = std::make_unique<TemporalUNet>(4, 4, 512, 768, num_frames);
    initialize_temporal_parameters();
}

Tensor TextToVideoPipeline::generate(const std::string& prompt, double guidance_scale, int seed,
                                     VideoDiffusionModel::MotionSchedule motion_schedule) {
    // Set random seed if provided
    if (seed >= 0) {
        std::srand(seed);
    }
      // Process text prompt (simplified)
    // Create a simple text embedding (placeholder)
    std::vector<std::string> tokens = TextProcessor::tokenize_words(prompt);
    Tensor text_embeddings({1, 768}); // 768-dim embeddings
    // Fill with placeholder values
    auto text_data = text_embeddings.data();
    for (size_t i = 0; i < text_embeddings.size(); ++i) {
        text_data[i] = 0.1; // Simple placeholder
    }
    
    // Create initial noise
    Tensor initial_noise = create_initial_noise(1);
      std::cout << "Starting video generation for prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Video size: " << video_size_ << "x" << video_size_ << ", Frames: " << num_frames_ << std::endl;
    
    // Generate video using temporal diffusion
    Tensor generated_video = VideoDiffusionModel::temporal_sample_loop(
        initial_noise, text_embeddings, num_steps_, strategy_, guidance_scale, VideoFormat::BTCHW
    );
    
    std::cout << "Video generation completed successfully!" << std::endl;
    
    return generated_video;
}

Tensor TextToVideoPipeline::generate_with_interpolation(const std::string& prompt,
                                                        const std::vector<int>& keyframes,
                                                        double guidance_scale, int seed) {
    // Generate base video
    Tensor base_video = generate(prompt, guidance_scale, seed);
    
    // Apply frame interpolation using existing video tensor operations
    for (size_t i = 0; i < keyframes.size() - 1; ++i) {
        int start_frame = keyframes[i];
        int end_frame = keyframes[i + 1];
        
        // Extract keyframes and interpolate between them
        Tensor start_tensor = VideoTensorUtils::temporal_crop(base_video, start_frame, 1, VideoFormat::BTCHW);
        Tensor end_tensor = VideoTensorUtils::temporal_crop(base_video, end_frame, 1, VideoFormat::BTCHW);
          // Use existing frame interpolation (placeholder)
        std::vector<Tensor> frames = {start_tensor, end_tensor};
        // Note: This should use proper frame interpolation from AdvancedOpticalFlow
        Tensor interpolated = start_tensor; // Placeholder - just use start frame
        
        // Replace section in base video (simplified)
        // In real implementation, would properly splice the interpolated frames
    }
    
    return base_video;
}

Tensor TextToVideoPipeline::video_to_video(const Tensor& source_video, const std::string& prompt,
                                          double strength, double guidance_scale) {
    // Add noise to source video based on strength
    auto temporal_betas = VideoDiffusionModel::create_temporal_schedule(num_steps_, num_frames_);
    auto [temporal_alphas, temporal_alpha_cumprod] = VideoDiffusionModel::compute_temporal_alphas(temporal_betas);
    
    int noise_steps = static_cast<int>(strength * num_steps_);
    Tensor noisy_video = VideoDiffusionModel::add_temporal_noise(
        source_video, noise_steps, temporal_alpha_cumprod, VideoFormat::BTCHW
    );
      // Process text prompt
    std::vector<std::string> tokens = TextProcessor::tokenize_words(prompt);
    Tensor text_embeddings({1, 768}); // 768-dim embeddings
    // Fill with placeholder values
    auto text_data = text_embeddings.data();
    for (size_t i = 0; i < text_embeddings.size(); ++i) {
        text_data[i] = 0.1; // Simple placeholder
    }
    
    // Denoise from the noisy video
    Tensor result_video = VideoDiffusionModel::temporal_sample_loop(
        noisy_video, text_embeddings, noise_steps, strategy_, guidance_scale, VideoFormat::BTCHW
    );
    
    return result_video;
}

double TextToVideoPipeline::compute_temporal_consistency(const Tensor& video_tensor) {
    // Simplified temporal consistency metric
    auto frames = VideoTensorUtils::extract_frames(video_tensor, VideoFormat::BTCHW);
    
    double total_similarity = 0.0;
    for (size_t i = 1; i < frames.size(); ++i) {
        // Compute frame-to-frame similarity (simplified)
        auto prev_data = frames[i-1].data();
        auto curr_data = frames[i].data();
        
        double similarity = 0.0;
        size_t frame_size = frames[i].size();
        
        for (size_t j = 0; j < frame_size; ++j) {
            double diff = prev_data[j] - curr_data[j];
            similarity += 1.0 / (1.0 + diff * diff); // Inverse squared difference
        }
        
        total_similarity += similarity / frame_size;
    }
    
    return total_similarity / (frames.size() - 1);
}

void TextToVideoPipeline::initialize_temporal_parameters() {
    temporal_betas_ = VideoDiffusionModel::create_temporal_schedule(
        num_steps_, num_frames_, VideoDiffusionModel::MotionSchedule::MOTION_ADAPTIVE
    );
    
    auto [alphas, alpha_cumprod] = VideoDiffusionModel::compute_temporal_alphas(temporal_betas_);
    temporal_alphas_ = alphas;
    temporal_alpha_cumprod_ = alpha_cumprod;
}

Tensor TextToVideoPipeline::create_initial_noise(size_t batch_size) {
    std::vector<size_t> shape = {batch_size, num_frames_, 4, video_size_, video_size_}; // BTCHW
    return VideoDiffusionModel::sample_temporal_noise(shape, VideoFormat::BTCHW);
}

// ===== VideoMotionUtils Implementation =====

Tensor VideoMotionUtils::estimate_motion_vectors(const Tensor& video_tensor, VideoFormat format) {
    auto info = VideoTensorUtils::get_tensor_info(video_tensor, format);
    
    // Simplified motion estimation using frame differences
    std::vector<size_t> motion_shape = {info.batch_size, info.num_frames - 1, 2, info.height, info.width};
    Tensor motion_vectors(motion_shape);
    
    auto frames = VideoTensorUtils::extract_frames(video_tensor, format);
    auto motion_data = motion_vectors.data();
    
    for (size_t f = 1; f < frames.size(); ++f) {
        auto prev_data = frames[f-1].data();
        auto curr_data = frames[f].data();
        
        // Simplified optical flow estimation
        size_t frame_size = frames[f].size();
        for (size_t i = 0; i < frame_size; ++i) {
            // Very basic motion estimation (frame difference)
            double diff = curr_data[i] - prev_data[i];
            
            // Store as motion vector components (simplified)
            size_t motion_idx = (f-1) * info.height * info.width * 2 + (i % (info.height * info.width)) * 2;
            if (motion_idx < motion_vectors.size()) {
                motion_data[motion_idx] = diff; // x component
                motion_data[motion_idx + 1] = diff * 0.5; // y component
            }
        }
    }
    
    return motion_vectors;
}

std::vector<std::vector<double>> VideoMotionUtils::compute_motion_schedule(
    const Tensor& motion_vectors,
    const std::vector<double>& base_schedule
) {
    // Compute motion magnitude per frame
    auto motion_data = motion_vectors.data();
    auto motion_shape = motion_vectors.shape();
    
    size_t num_frames = motion_shape[1]; // Assuming BTCHW format for motion vectors
    std::vector<double> motion_magnitude(num_frames, 0.0);
    
    // Calculate average motion per frame
    size_t vectors_per_frame = motion_vectors.size() / num_frames;
    for (size_t f = 0; f < num_frames; ++f) {
        double total_motion = 0.0;
        for (size_t i = 0; i < vectors_per_frame; ++i) {
            size_t idx = f * vectors_per_frame + i;
            if (idx < motion_vectors.size()) {
                total_motion += std::abs(motion_data[idx]);
            }
        }
        motion_magnitude[f] = total_motion / vectors_per_frame;
    }
    
    // Adapt schedule based on motion
    std::vector<std::vector<double>> adapted_schedule(base_schedule.size());
    for (size_t t = 0; t < base_schedule.size(); ++t) {
        adapted_schedule[t].resize(num_frames);
        for (size_t f = 0; f < num_frames; ++f) {
            // Higher motion -> slightly more noise for better denoising
            double motion_factor = 1.0 + 0.1 * motion_magnitude[f];
            adapted_schedule[t][f] = base_schedule[t] * motion_factor;
        }
    }
    
    return adapted_schedule;
}

Tensor VideoMotionUtils::temporal_smoothing(const Tensor& video_tensor, double smoothing_factor, VideoFormat format) {
    auto info = VideoTensorUtils::get_tensor_info(video_tensor, format);
    Tensor smoothed = video_tensor;
    
    if (info.num_frames < 2) return smoothed;
    
    auto original_data = video_tensor.data();
    auto smoothed_data = smoothed.data();
    
    size_t frame_size = info.height * info.width * info.channels;
    
    // Apply temporal smoothing
    for (size_t b = 0; b < info.batch_size; ++b) {
        for (size_t f = 1; f < info.num_frames - 1; ++f) { // Skip first and last frame
            size_t curr_offset, prev_offset, next_offset;
            
            if (format == VideoFormat::BTCHW) {
                curr_offset = b * info.num_frames * info.channels * info.height * info.width +
                             f * info.channels * info.height * info.width;
                prev_offset = curr_offset - info.channels * info.height * info.width;
                next_offset = curr_offset + info.channels * info.height * info.width;
            } else { // BTHWC
                curr_offset = b * info.num_frames * info.height * info.width * info.channels +
                             f * info.height * info.width * info.channels;
                prev_offset = curr_offset - info.height * info.width * info.channels;
                next_offset = curr_offset + info.height * info.width * info.channels;
            }
            
            for (size_t i = 0; i < frame_size; ++i) {
                double prev_val = original_data[prev_offset + i];
                double curr_val = original_data[curr_offset + i];
                double next_val = original_data[next_offset + i];
                
                // Temporal smoothing with weighted average
                double smoothed_val = (1.0 - smoothing_factor) * curr_val + 
                                     smoothing_factor * 0.5 * (prev_val + next_val);
                
                smoothed_data[curr_offset + i] = smoothed_val;
            }
        }
    }
    
    return smoothed;
}

Tensor VideoMotionUtils::warp_frame(const Tensor& frame, const Tensor& prev_frame, const Tensor& motion_vectors) {
    // Simplified frame warping
    // In real implementation, this would use proper optical flow warping
    
    Tensor warped = frame;
    auto frame_data = frame.data();
    auto prev_data = prev_frame.data();
    auto motion_data = motion_vectors.data();
    auto warped_data = warped.data();
    
    // Apply simple warping based on motion vectors
    for (size_t i = 0; i < frame.size(); ++i) {
        // Simplified warping: blend current and previous frame based on motion
        double motion_mag = (i < motion_vectors.size()) ? std::abs(motion_data[i]) : 0.0;
        double blend_factor = std::min(motion_mag, 1.0);
        
        warped_data[i] = (1.0 - blend_factor) * frame_data[i] + blend_factor * prev_data[i];
    }
    
    return warped;
}

} // namespace ai
} // namespace asekioml
