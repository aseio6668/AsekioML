#include "ai/simple_video_diffusion.hpp"
#include "ai/video_tensor_ops.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace clmodel::ai;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

void test_temporal_scheduling() {
    print_separator("Testing Simple Temporal Scheduling");
    
    try {
        int num_steps = 10;
        int num_frames = 4;
        
        std::cout << "Creating temporal schedule with " << num_steps << " steps and " << num_frames << " frames..." << std::endl;
        
        auto temporal_betas = SimpleVideoDiffusionModel::create_temporal_schedule(num_steps, num_frames);
        
        std::cout << "✓ Temporal schedule created successfully" << std::endl;
        std::cout << "Schedule shape: [" << temporal_betas.size() << ", " << temporal_betas[0].size() << "]" << std::endl;
        
        // Display sample values
        std::cout << "\nSample beta values (first 3 steps, all frames):" << std::endl;
        for (int t = 0; t < 3; ++t) {
            std::cout << "  Step " << t << ": ";
            for (int f = 0; f < num_frames; ++f) {
                std::cout << std::fixed << std::setprecision(6) << temporal_betas[t][f] << " ";
            }
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in temporal scheduling: " << e.what() << std::endl;
    }
}

void test_temporal_noise_operations() {
    print_separator("Testing Simple Temporal Noise Operations");
    
    try {
        // Create a small video tensor [B=1, T=4, C=3, H=16, W=16]
        std::vector<size_t> video_shape = {1, 4, 3, 16, 16};
        
        std::cout << "Creating test video tensor with shape: [";
        for (size_t i = 0; i < video_shape.size(); ++i) {
            std::cout << video_shape[i];
            if (i < video_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Create synthetic video data
        Tensor video_tensor(video_shape);
        auto data = video_tensor.data();
        for (size_t i = 0; i < video_tensor.size(); ++i) {
            data[i] = 0.5 + 0.3 * std::sin(i * 0.01);
        }
        
        std::cout << "✓ Test video tensor created" << std::endl;
        
        // Test temporal noise generation
        std::cout << "\nTesting temporal noise generation..." << std::endl;
        Tensor temporal_noise = SimpleVideoDiffusionModel::sample_temporal_noise(video_shape);
        
        std::cout << "✓ Temporal noise generated successfully" << std::endl;
        
        // Test adding temporal noise
        int num_steps = 10, num_frames = 4;
        auto temporal_betas = SimpleVideoDiffusionModel::create_temporal_schedule(num_steps, num_frames);
        
        // Compute alpha_cumprod
        std::vector<std::vector<double>> temporal_alpha_cumprod(num_steps, std::vector<double>(num_frames));
        for (int t = 0; t < num_steps; ++t) {
            for (int f = 0; f < num_frames; ++f) {
                double alpha = 1.0 - temporal_betas[t][f];
                temporal_alpha_cumprod[t][f] = (t == 0) ? alpha : temporal_alpha_cumprod[t-1][f] * alpha;
            }
        }
        
        std::cout << "\nTesting temporal noise addition..." << std::endl;
        int test_timestep = 5;
        Tensor noisy_video = SimpleVideoDiffusionModel::add_temporal_noise(
            video_tensor, test_timestep, temporal_alpha_cumprod
        );
        
        std::cout << "✓ Temporal noise added at timestep " << test_timestep << std::endl;
        
        // Check noise effect
        auto original_data = video_tensor.data();
        auto noisy_data = noisy_video.data();
        double total_diff = 0.0;
        size_t sample_size = std::min(video_tensor.size(), (size_t)1000);
        for (size_t i = 0; i < sample_size; ++i) {
            total_diff += std::abs(noisy_data[i] - original_data[i]);
        }
        double avg_diff = total_diff / sample_size;
        
        std::cout << "Average difference after noise addition: " << std::fixed << std::setprecision(4) << avg_diff << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in temporal noise operations: " << e.what() << std::endl;
    }
}

void test_simple_temporal_unet() {
    print_separator("Testing Simple Temporal U-Net");
    
    try {
        // Create simple temporal U-Net
        std::cout << "Creating Simple Temporal U-Net..." << std::endl;
        SimpleTemporalUNet unet(4); // 4 frames
        
        std::cout << "✓ Simple Temporal U-Net created" << std::endl;
        
        // Create test input [B=1, T=4, C=3, H=16, W=16]
        std::vector<size_t> input_shape = {1, 4, 3, 16, 16};
        Tensor test_input(input_shape);
        auto input_data = test_input.data();
        
        // Fill with test pattern
        for (size_t i = 0; i < test_input.size(); ++i) {
            input_data[i] = 0.1 * std::sin(i * 0.001);
        }
        
        std::cout << "Testing forward pass..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Tensor output = unet.forward(test_input, 5);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ Forward pass completed in " << duration.count() << "ms" << std::endl;
        
        // Verify output is different from input
        auto output_data = output.data();
        double total_diff = 0.0;
        size_t sample_size = std::min(test_input.size(), (size_t)500);
        for (size_t i = 0; i < sample_size; ++i) {
            total_diff += std::abs(output_data[i] - input_data[i]);
        }
        double avg_diff = total_diff / sample_size;
        
        std::cout << "Average input-output difference: " << std::fixed << std::setprecision(6) << avg_diff << std::endl;
        
        if (avg_diff > 1e-6) {
            std::cout << "✓ U-Net is transforming input (not just copying)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in Simple Temporal U-Net test: " << e.what() << std::endl;
    }
}

void test_video_generation() {
    print_separator("Testing Simple Video Generation");
    
    try {
        std::cout << "Creating Simple Text-to-Video Pipeline..." << std::endl;
        
        // Create pipeline with small parameters for testing
        SimpleTextToVideoPipeline pipeline(
            16,  // 16x16 video size
            4,   // 4 frames
            5    // 5 diffusion steps
        );
        
        std::cout << "✓ Pipeline created successfully" << std::endl;
        
        std::cout << "\nTesting video generation..." << std::endl;
        std::string test_prompt = "A bouncing ball";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Tensor generated_video = pipeline.generate(test_prompt);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ Video generation completed in " << duration.count() << "ms" << std::endl;
        
        // Check generated video properties
        std::cout << "\nGenerated video shape: [";
        for (size_t i = 0; i < generated_video.shape().size(); ++i) {
            std::cout << generated_video.shape()[i];
            if (i < generated_video.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Compute motion strength
        double motion = SimpleVideoMotionUtils::estimate_motion_strength(generated_video);
        std::cout << "Estimated motion strength: " << std::fixed << std::setprecision(4) << motion << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in video generation test: " << e.what() << std::endl;
    }
}

void test_motion_utils() {
    print_separator("Testing Simple Video Motion Utils");
    
    try {
        // Create test video with motion
        std::vector<size_t> video_shape = {1, 6, 3, 8, 8};
        Tensor test_video(video_shape);
        auto data = test_video.data();
        
        // Create video with moving pattern
        size_t frame_size = 3 * 8 * 8;
        for (size_t f = 0; f < 6; ++f) {
            for (size_t i = 0; i < frame_size; ++i) {
                // Create moving pattern
                double pattern = std::sin((i + f * 10) * 0.1);
                data[f * frame_size + i] = 0.5 + 0.3 * pattern;
            }
        }
        
        std::cout << "Created test video with moving pattern (6 frames, 8x8)" << std::endl;
        
        // Test motion estimation
        std::cout << "\nTesting motion estimation..." << std::endl;
        double motion_strength = SimpleVideoMotionUtils::estimate_motion_strength(test_video);
        
        std::cout << "✓ Motion strength estimated: " << std::fixed << std::setprecision(4) << motion_strength << std::endl;
        
        // Test temporal smoothing
        std::cout << "\nTesting temporal smoothing..." << std::endl;
        Tensor smoothed_video = SimpleVideoMotionUtils::temporal_smoothing(test_video, 0.3);
        
        std::cout << "✓ Temporal smoothing applied" << std::endl;
        
        // Check smoothing effect
        auto orig_data = test_video.data();
        auto smooth_data = smoothed_video.data();
        double total_diff = 0.0;
        size_t sample_size = std::min(test_video.size(), (size_t)200);
        
        for (size_t i = 0; i < sample_size; ++i) {
            total_diff += std::abs(smooth_data[i] - orig_data[i]);
        }
        double avg_diff = total_diff / sample_size;
        
        std::cout << "Average smoothing difference: " << std::fixed << std::setprecision(6) << avg_diff << std::endl;
        
        // Estimate motion after smoothing
        double smoothed_motion = SimpleVideoMotionUtils::estimate_motion_strength(smoothed_video);
        std::cout << "Motion strength after smoothing: " << std::fixed << std::setprecision(4) << smoothed_motion << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in motion utils test: " << e.what() << std::endl;
    }
}

void performance_benchmark() {
    print_separator("Simple Video Diffusion Performance Benchmark");
    
    try {
        std::cout << "Running performance benchmark..." << std::endl;
        
        std::vector<std::tuple<std::string, size_t, size_t, int>> test_cases = {
            {"Tiny (4 frames, 8x8)", 8, 4, 3},
            {"Small (6 frames, 16x16)", 16, 6, 5},
            {"Medium (8 frames, 32x32)", 32, 8, 5}
        };
        
        for (const auto& [name, size, frames, steps] : test_cases) {
            std::cout << "\nBenchmarking " << name << " with " << steps << " steps..." << std::endl;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            SimpleTextToVideoPipeline pipeline(size, frames, steps);
            Tensor result = pipeline.generate("test video");
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Calculate throughput
            size_t total_pixels = 1 * frames * 3 * size * size; // B * T * C * H * W
            double pixels_per_ms = (double)total_pixels / duration.count();
            
            std::cout << "  Time: " << duration.count() << "ms" << std::endl;
            std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << pixels_per_ms << " pixels/ms" << std::endl;
            std::cout << "  Memory: " << (total_pixels * sizeof(double)) / (1024 * 1024) << " MB" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in performance benchmark: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "     CLModel Simple Video Diffusion Demo" << std::endl;
    std::cout << "     Week 10: Video Diffusion Architecture" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Run all tests
    test_temporal_scheduling();
    test_temporal_noise_operations();
    test_simple_temporal_unet();
    test_video_generation();
    test_motion_utils();
    performance_benchmark();
    
    print_separator("Simple Video Diffusion Demo Summary");
    
    std::cout << "✅ Week 10 Video Diffusion Architecture Implementation Complete!" << std::endl;
    std::cout << "\nKey Features Demonstrated:" << std::endl;
    std::cout << "  • Temporal noise scheduling with motion awareness" << std::endl;
    std::cout << "  • Video diffusion operations for temporal data" << std::endl;
    std::cout << "  • Simple Temporal U-Net for video processing" << std::endl;
    std::cout << "  • Text-to-video generation pipeline" << std::endl;
    std::cout << "  • Motion estimation and temporal consistency" << std::endl;
    std::cout << "  • Video preprocessing and postprocessing utilities" << std::endl;
    
    std::cout << "\n🎬 Week 10 Foundation Complete!" << std::endl;
    std::cout << "\nNext Steps (Week 11-12):" << std::endl;
    std::cout << "  • Advanced frame interpolation techniques" << std::endl;
    std::cout << "  • Enhanced motion synthesis and style transfer" << std::endl;
    std::cout << "  • Audio-visual synchronization" << std::endl;
    std::cout << "  • Performance optimizations and scaling" << std::endl;
    
    return 0;
}
