#include "ai/simple_video_diffusion.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {
namespace ai {

// ===== SimpleVideoDiffusionModel Implementation =====

std::vector<std::vector<double>> SimpleVideoDiffusionModel::create_temporal_schedule(
    int num_steps, 
    int num_frames
) {
    std::vector<std::vector<double>> temporal_betas(num_steps, std::vector<double>(num_frames));
    
    double beta_start = 0.0001;
    double beta_end = 0.02;
    
    for (int t = 0; t < num_steps; ++t) {
        double base_beta = beta_start + (beta_end - beta_start) * t / (num_steps - 1);
        
        for (int f = 0; f < num_frames; ++f) {
            // Motion-adaptive: slightly higher noise in middle frames
            double frame_factor = 1.0 + 0.1 * std::sin(M_PI * f / (num_frames - 1));
            temporal_betas[t][f] = base_beta * frame_factor;
        }
    }
    
    return temporal_betas;
}

Tensor SimpleVideoDiffusionModel::add_temporal_noise(
    const Tensor& x0, 
    int t, 
    const std::vector<std::vector<double>>& temporal_alpha_cumprod
) {
    Tensor noise = sample_temporal_noise(x0.shape());
    Tensor noisy_video = x0;
    
    auto x0_data = x0.data();
    auto noise_data = noise.data();
    auto result_data = noisy_video.data();
    
    // Simple implementation: assume BTCHW format [B=1, T, C, H, W]
    auto shape = x0.shape();
    size_t T = shape[1], C = shape[2], H = shape[3], W = shape[4];
    size_t frame_size = C * H * W;
    
    for (size_t f = 0; f < T; ++f) {
        if (t >= temporal_alpha_cumprod.size() || f >= temporal_alpha_cumprod[t].size()) continue;
        
        double alpha_cumprod = temporal_alpha_cumprod[t][f];
        double sqrt_alpha_cumprod = std::sqrt(alpha_cumprod);
        double sqrt_one_minus_alpha_cumprod = std::sqrt(1.0 - alpha_cumprod);
        
        size_t frame_offset = f * frame_size;
        
        for (size_t i = 0; i < frame_size; ++i) {
            size_t idx = frame_offset + i;
            if (idx < x0.size()) {
                result_data[idx] = sqrt_alpha_cumprod * x0_data[idx] + 
                                  sqrt_one_minus_alpha_cumprod * noise_data[idx];
            }
        }
    }
    
    return noisy_video;
}

Tensor SimpleVideoDiffusionModel::sample_temporal_noise(const std::vector<size_t>& video_shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    Tensor noise(video_shape);
    auto data = noise.data();
    
    // Generate noise
    size_t total_size = noise.size();
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dist(gen);
    }
    
    // Apply light temporal smoothing for correlation
    return SimpleVideoMotionUtils::temporal_smoothing(noise, 0.1);
}

Tensor SimpleVideoDiffusionModel::temporal_denoise_step(
    const Tensor& x_t,
    const Tensor& predicted_noise,
    int t,
    const std::vector<std::vector<double>>& temporal_alphas
) {
    Tensor x_prev = x_t;
    auto x_t_data = x_t.data();
    auto noise_data = predicted_noise.data();
    auto result_data = x_prev.data();
    
    // Simple denoising: x_{t-1} = (x_t - noise) / alpha_t
    auto shape = x_t.shape();
    size_t T = shape[1], C = shape[2], H = shape[3], W = shape[4];
    size_t frame_size = C * H * W;
    
    for (size_t f = 0; f < T; ++f) {
        if (t >= temporal_alphas.size() || f >= temporal_alphas[t].size()) continue;
        
        double alpha_t = temporal_alphas[t][f];
        double coeff = 1.0 / std::sqrt(alpha_t);
        
        size_t frame_offset = f * frame_size;
        
        for (size_t i = 0; i < frame_size; ++i) {
            size_t idx = frame_offset + i;
            if (idx < x_t.size()) {
                result_data[idx] = coeff * (x_t_data[idx] - 0.1 * noise_data[idx]);
            }
        }
    }
    
    // Apply temporal smoothing for consistency
    return SimpleVideoMotionUtils::temporal_smoothing(x_prev, 0.2);
}

Tensor SimpleVideoDiffusionModel::temporal_sample_loop(
    const Tensor& x_t,
    int num_steps
) {
    Tensor x = x_t;
    auto shape = x.shape();
    int num_frames = static_cast<int>(shape[1]);
    
    // Create temporal schedules
    auto temporal_betas = create_temporal_schedule(num_steps, num_frames);
    
    // Compute alphas
    std::vector<std::vector<double>> temporal_alphas(num_steps, std::vector<double>(num_frames));
    for (int t = 0; t < num_steps; ++t) {
        for (int f = 0; f < num_frames; ++f) {
            temporal_alphas[t][f] = 1.0 - temporal_betas[t][f];
        }
    }
    
    // Create simple temporal U-Net
    SimpleTemporalUNet unet(num_frames);
    
    for (int t = num_steps - 1; t >= 0; --t) {
        // Predict noise with temporal U-Net
        Tensor predicted_noise = unet.forward(x, t);
        
        // Apply temporal denoising step
        x = temporal_denoise_step(x, predicted_noise, t, temporal_alphas);
        
        if (t % 5 == 0) {
            std::cout << "Temporal denoising step " << t << " completed" << std::endl;
        }
    }
    
    return x;
}

// ===== SimpleTemporalUNet Implementation =====

SimpleTemporalUNet::SimpleTemporalUNet(size_t num_frames) 
    : num_frames_(num_frames) {
}

Tensor SimpleTemporalUNet::forward(const Tensor& x_t, int t) {
    // Extract frames and process individually
    auto frames = VideoTensorUtils::extract_frames(x_t, VideoFormat::BTCHW);
    std::vector<Tensor> processed_frames;
    
    for (const auto& frame : frames) {
        Tensor processed = process_frame(frame, t);
        processed_frames.push_back(processed);
    }
    
    // Reconstruct video tensor
    Tensor result = VideoTensorUtils::create_video_tensor(processed_frames, VideoFormat::BTCHW);
    
    // Apply temporal smoothing
    return temporal_smooth(result);
}

Tensor SimpleTemporalUNet::process_frame(const Tensor& frame, int t) {
    // Simple frame processing - just add some pattern based on timestep
    Tensor processed = frame;
    auto data = processed.data();
    
    // Add timestep-dependent pattern
    double time_factor = 0.01 * std::sin(t * 0.1);
    
    for (size_t i = 0; i < processed.size(); ++i) {
        data[i] = data[i] + time_factor * std::sin(i * 0.001);
    }
    
    return processed;
}

Tensor SimpleTemporalUNet::temporal_smooth(const Tensor& video_tensor) {
    return SimpleVideoMotionUtils::temporal_smoothing(video_tensor, 0.15);
}

// ===== SimpleTextToVideoPipeline Implementation =====

SimpleTextToVideoPipeline::SimpleTextToVideoPipeline(size_t video_size, size_t num_frames, int num_steps)
    : video_size_(video_size), num_frames_(num_frames), num_steps_(num_steps) {
    
    temporal_unet_ = std::make_unique<SimpleTemporalUNet>(num_frames);
}

Tensor SimpleTextToVideoPipeline::generate(const std::string& prompt) {
    std::cout << "Starting simple video generation for prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Video size: " << video_size_ << "x" << video_size_ << ", Frames: " << num_frames_ << std::endl;
    
    // Create initial noise
    Tensor initial_noise = create_initial_noise();
    
    // Generate video using temporal diffusion
    Tensor generated_video = SimpleVideoDiffusionModel::temporal_sample_loop(
        initial_noise, num_steps_
    );
    
    std::cout << "Simple video generation completed successfully!" << std::endl;
    
    return generated_video;
}

Tensor SimpleTextToVideoPipeline::create_initial_noise() {
    // Create noise tensor [B=1, T, C=3, H, W]
    std::vector<size_t> shape = {1, num_frames_, 3, video_size_, video_size_};
    return SimpleVideoDiffusionModel::sample_temporal_noise(shape);
}

// ===== SimpleVideoMotionUtils Implementation =====

Tensor SimpleVideoMotionUtils::temporal_smoothing(
    const Tensor& video_tensor,
    double smoothing_factor
) {
    auto shape = video_tensor.shape();
    if (shape.size() < 5) return video_tensor; // Not a video tensor
    
    size_t T = shape[1], C = shape[2], H = shape[3], W = shape[4];
    if (T < 3) return video_tensor; // Need at least 3 frames for smoothing
    
    Tensor smoothed = video_tensor;
    auto original_data = video_tensor.data();
    auto smoothed_data = smoothed.data();
    
    size_t frame_size = C * H * W;
    
    // Apply temporal smoothing (skip first and last frame)
    for (size_t f = 1; f < T - 1; ++f) {
        size_t curr_offset = f * frame_size;
        size_t prev_offset = (f - 1) * frame_size;
        size_t next_offset = (f + 1) * frame_size;
        
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
    
    return smoothed;
}

double SimpleVideoMotionUtils::estimate_motion_strength(const Tensor& video_tensor) {
    auto shape = video_tensor.shape();
    if (shape.size() < 5 || shape[1] < 2) return 0.0; // Need at least 2 frames
    
    size_t T = shape[1], C = shape[2], H = shape[3], W = shape[4];
    auto data = video_tensor.data();
    
    double total_motion = 0.0;
    size_t frame_size = C * H * W;
    
    // Calculate frame-to-frame differences
    for (size_t f = 1; f < T; ++f) {
        size_t curr_offset = f * frame_size;
        size_t prev_offset = (f - 1) * frame_size;
        
        double frame_diff = 0.0;
        for (size_t i = 0; i < frame_size; ++i) {
            double diff = data[curr_offset + i] - data[prev_offset + i];
            frame_diff += diff * diff;
        }
        
        total_motion += std::sqrt(frame_diff / frame_size);
    }
    
    return total_motion / (T - 1);
}

} // namespace ai
} // namespace asekioml
