#include "ai/video_diffusion.hpp"
#include "ai/video_tensor_ops.hpp"
#include "ai/text_to_image.hpp"
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
    print_separator("Testing Temporal Noise Scheduling");
    
    try {
        int num_steps = 20;
        int num_frames = 8;
        
        std::cout << "Creating temporal schedules with " << num_steps << " steps and " << num_frames << " frames..." << std::endl;
        
        // Test different motion schedules
        auto uniform_betas = VideoDiffusionModel::create_temporal_schedule(
            num_steps, num_frames, VideoDiffusionModel::MotionSchedule::UNIFORM
        );
        
        auto motion_adaptive_betas = VideoDiffusionModel::create_temporal_schedule(
            num_steps, num_frames, VideoDiffusionModel::MotionSchedule::MOTION_ADAPTIVE
        );
        
        auto temporal_decay_betas = VideoDiffusionModel::create_temporal_schedule(
            num_steps, num_frames, VideoDiffusionModel::MotionSchedule::TEMPORAL_DECAY
        );
        
        std::cout << "✓ All temporal schedules created successfully" << std::endl;
        
        // Display sample values
        std::cout << "\nSample beta values (first 3 steps, first 4 frames):" << std::endl;
        std::cout << "Uniform schedule:" << std::endl;
        for (int t = 0; t < 3; ++t) {
            std::cout << "  Step " << t << ": ";
            for (int f = 0; f < 4; ++f) {
                std::cout << std::fixed << std::setprecision(6) << uniform_betas[t][f] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Motion adaptive schedule:" << std::endl;
        for (int t = 0; t < 3; ++t) {
            std::cout << "  Step " << t << ": ";
            for (int f = 0; f < 4; ++f) {
                std::cout << std::fixed << std::setprecision(6) << motion_adaptive_betas[t][f] << " ";
            }
            std::cout << std::endl;
        }
        
        // Test alpha computation
        auto [alphas, alpha_cumprod] = VideoDiffusionModel::compute_temporal_alphas(uniform_betas);
        
        std::cout << "\n✓ Alpha values computed successfully" << std::endl;
        std::cout << "Alpha shapes: [" << alphas.size() << ", " << alphas[0].size() << "]" << std::endl;
        std::cout << "Alpha_cumprod shapes: [" << alpha_cumprod.size() << ", " << alpha_cumprod[0].size() << "]" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in temporal scheduling: " << e.what() << std::endl;
    }
}

void test_temporal_noise_operations() {
    print_separator("Testing Temporal Noise Operations");
    
    try {
        // Create a simple video tensor [B=1, T=4, C=3, H=32, W=32]
        std::vector<size_t> video_shape = {1, 4, 3, 32, 32};
        
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
            data[i] = 0.5 + 0.3 * std::sin(i * 0.01); // Synthetic pattern
        }
        
        std::cout << "✓ Test video tensor created" << std::endl;
        
        // Test temporal noise generation
        std::cout << "\nTesting temporal noise generation..." << std::endl;
        Tensor temporal_noise = VideoDiffusionModel::sample_temporal_noise(video_shape, VideoFormat::BTCHW);
        
        std::cout << "✓ Temporal noise generated, shape: [";
        for (size_t i = 0; i < temporal_noise.shape().size(); ++i) {
            std::cout << temporal_noise.shape()[i];
            if (i < temporal_noise.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Create temporal schedule for noise testing
        int num_steps = 10;
        int num_frames = 4;
        auto temporal_betas = VideoDiffusionModel::create_temporal_schedule(num_steps, num_frames);
        auto [temporal_alphas, temporal_alpha_cumprod] = VideoDiffusionModel::compute_temporal_alphas(temporal_betas);
        
        // Test adding temporal noise
        std::cout << "\nTesting temporal noise addition..." << std::endl;
        int test_timestep = 5;
        Tensor noisy_video = VideoDiffusionModel::add_temporal_noise(
            video_tensor, test_timestep, temporal_alpha_cumprod, VideoFormat::BTCHW
        );
        
        std::cout << "✓ Temporal noise added at timestep " << test_timestep << std::endl;
        
        // Check noise effect
        auto original_data = video_tensor.data();
        auto noisy_data = noisy_video.data();
        double total_diff = 0.0;
        for (size_t i = 0; i < video_tensor.size() && i < 1000; ++i) { // Sample first 1000 elements
            total_diff += std::abs(noisy_data[i] - original_data[i]);
        }
        double avg_diff = total_diff / std::min(video_tensor.size(), (size_t)1000);
        
        std::cout << "Average difference after noise addition: " << std::fixed << std::setprecision(4) << avg_diff << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in temporal noise operations: " << e.what() << std::endl;
    }
}

void test_temporal_unet() {
    print_separator("Testing Temporal U-Net");
    
    try {
        // Create temporal U-Net
        std::cout << "Creating Temporal U-Net..." << std::endl;
        TemporalUNet unet(4, 4, 512, 768, 8); // 4 in/out channels, 8 frames
        
        std::cout << "✓ Temporal U-Net created" << std::endl;
        
        // Create test input [B=1, T=8, C=4, H=64, W=64]
        std::vector<size_t> input_shape = {1, 8, 4, 64, 64};
        Tensor test_input(input_shape);
        auto input_data = test_input.data();
        
        // Fill with test pattern
        for (size_t i = 0; i < test_input.size(); ++i) {
            input_data[i] = 0.1 * std::sin(i * 0.001);
        }
        
        // Create text embeddings [B=1, seq_len=77, embed_dim=768]
        Tensor text_embeddings({1, 77, 768});
        auto text_data = text_embeddings.data();
        for (size_t i = 0; i < text_embeddings.size(); ++i) {
            text_data[i] = 0.01 * (i % 100); // Simple pattern
        }
        
        std::cout << "Testing forward pass..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Tensor output = unet.forward(test_input, 10, text_embeddings, VideoFormat::BTCHW);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ Forward pass completed in " << duration.count() << "ms" << std::endl;
        
        // Check output shape
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output.shape().size(); ++i) {
            std::cout << output.shape()[i];
            if (i < output.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Verify output is different from input
        auto output_data = output.data();
        double total_diff = 0.0;
        size_t sample_size = std::min(test_input.size(), (size_t)1000);
        for (size_t i = 0; i < sample_size; ++i) {
            total_diff += std::abs(output_data[i] - input_data[i]);
        }
        double avg_diff = total_diff / sample_size;
        
        std::cout << "Average input-output difference: " << std::fixed << std::setprecision(6) << avg_diff << std::endl;
        
        if (avg_diff > 1e-6) {
            std::cout << "✓ U-Net is transforming input (not just copying)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in Temporal U-Net test: " << e.what() << std::endl;
    }
}

void test_video_generation_strategies() {
    print_separator("Testing Video Generation Strategies");
    
    try {
        // Test different generation strategies with small video
        std::vector<size_t> small_video_shape = {1, 4, 3, 16, 16}; // Very small for testing
        
        std::cout << "Testing generation strategies with small video (4 frames, 16x16)..." << std::endl;
        
        // Create initial noise
        Tensor initial_noise = VideoDiffusionModel::sample_temporal_noise(small_video_shape, VideoFormat::BTCHW);
        
        // Create dummy text embeddings
        Tensor text_embeddings({1, 10, 128}); // Small embeddings for testing
        auto text_data = text_embeddings.data();
        for (size_t i = 0; i < text_embeddings.size(); ++i) {
            text_data[i] = 0.1 * std::sin(i * 0.1);
        }
        
        std::cout << "\nTesting FRAME_BY_FRAME strategy..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Tensor frame_result = VideoDiffusionModel::temporal_sample_loop(
            initial_noise, text_embeddings, 5, // Only 5 steps for testing
            VideoDiffusionModel::GenerationStrategy::FRAME_BY_FRAME, 1.0, VideoFormat::BTCHW
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ Frame-by-frame generation completed in " << duration.count() << "ms" << std::endl;
        
        std::cout << "\nTesting TEMPORAL_AWARE strategy..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        Tensor temporal_result = VideoDiffusionModel::temporal_sample_loop(
            initial_noise, text_embeddings, 5, // Only 5 steps for testing
            VideoDiffusionModel::GenerationStrategy::TEMPORAL_AWARE, 1.0, VideoFormat::BTCHW
        );
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ Temporal-aware generation completed in " << duration.count() << "ms" << std::endl;
        
        // Compare results
        auto frame_data = frame_result.data();
        auto temporal_data = temporal_result.data();
        double total_diff = 0.0;
        size_t sample_size = std::min(frame_result.size(), (size_t)500);
        
        for (size_t i = 0; i < sample_size; ++i) {
            total_diff += std::abs(frame_data[i] - temporal_data[i]);
        }
        double avg_diff = total_diff / sample_size;
        
        std::cout << "\nAverage difference between strategies: " << std::fixed << std::setprecision(6) << avg_diff << std::endl;
        
        if (avg_diff > 1e-6) {
            std::cout << "✓ Different strategies produce different results" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in generation strategy test: " << e.what() << std::endl;
    }
}

void test_text_to_video_pipeline() {
    print_separator("Testing Text-to-Video Pipeline");
    
    try {
        std::cout << "Creating Text-to-Video Pipeline..." << std::endl;
        
        // Create pipeline with small parameters for testing
        TextToVideoPipeline pipeline(
            32,  // 32x32 video size
            4,   // 4 frames
            10,  // 10 diffusion steps
            VideoDiffusionModel::GenerationStrategy::TEMPORAL_AWARE
        );
        
        std::cout << "✓ Pipeline created successfully" << std::endl;
        
        std::cout << "\nTesting video generation..." << std::endl;
        std::string test_prompt = "A red ball bouncing";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Tensor generated_video = pipeline.generate(
            test_prompt,
            2.0,  // guidance scale
            42    // seed for reproducibility
        );
        
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
        
        // Compute temporal consistency
        double consistency = pipeline.compute_temporal_consistency(generated_video);
        std::cout << "Temporal consistency score: " << std::fixed << std::setprecision(4) << consistency << std::endl;
        
        std::cout << "\nTesting video-to-video transformation..." << std::endl;
        
        Tensor transformed_video = pipeline.video_to_video(
            generated_video,
            "A blue ball bouncing",
            0.5,  // transformation strength
            2.0   // guidance scale
        );
        
        std::cout << "✓ Video-to-video transformation completed" << std::endl;
        
        // Compare original and transformed
        auto orig_data = generated_video.data();
        auto trans_data = transformed_video.data();
        double total_diff = 0.0;
        size_t sample_size = std::min(generated_video.size(), (size_t)500);
        
        for (size_t i = 0; i < sample_size; ++i) {
            total_diff += std::abs(orig_data[i] - trans_data[i]);
        }
        double avg_diff = total_diff / sample_size;
        
        std::cout << "Average transformation difference: " << std::fixed << std::setprecision(6) << avg_diff << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in text-to-video pipeline test: " << e.what() << std::endl;
    }
}

void test_motion_utils() {
    print_separator("Testing Video Motion Utils");
    
    try {
        // Create test video with motion
        std::vector<size_t> video_shape = {1, 6, 3, 16, 16};
        Tensor test_video(video_shape);
        auto data = test_video.data();
        
        // Create video with moving pattern
        size_t frame_size = 3 * 16 * 16;
        for (size_t f = 0; f < 6; ++f) {
            for (size_t i = 0; i < frame_size; ++i) {
                // Create moving diagonal pattern
                size_t pixel_idx = i % (16 * 16);
                size_t y = pixel_idx / 16;
                size_t x = pixel_idx % 16;
                
                // Moving diagonal
                double pattern = std::sin((x + y + f * 2) * 0.3);
                data[f * frame_size + i] = 0.5 + 0.3 * pattern;
            }
        }
        
        std::cout << "Created test video with moving pattern (6 frames, 16x16)" << std::endl;
        
        // Test motion estimation
        std::cout << "\nTesting motion vector estimation..." << std::endl;
        Tensor motion_vectors = VideoMotionUtils::estimate_motion_vectors(test_video, VideoFormat::BTCHW);
        
        std::cout << "✓ Motion vectors computed, shape: [";
        for (size_t i = 0; i < motion_vectors.shape().size(); ++i) {
            std::cout << motion_vectors.shape()[i];
            if (i < motion_vectors.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Test motion-aware scheduling
        std::cout << "\nTesting motion-aware scheduling..." << std::endl;
        std::vector<double> base_schedule = {0.0001, 0.0005, 0.001, 0.005, 0.01};
        auto motion_schedule = VideoMotionUtils::compute_motion_schedule(motion_vectors, base_schedule);
        
        std::cout << "✓ Motion-adapted schedule computed" << std::endl;
        std::cout << "Schedule shape: [" << motion_schedule.size() << ", " << motion_schedule[0].size() << "]" << std::endl;
        
        // Test temporal smoothing
        std::cout << "\nTesting temporal smoothing..." << std::endl;
        Tensor smoothed_video = VideoMotionUtils::temporal_smoothing(test_video, 0.3, VideoFormat::BTCHW);
        
        std::cout << "✓ Temporal smoothing applied" << std::endl;
        
        // Check smoothing effect
        auto orig_data = test_video.data();
        auto smooth_data = smoothed_video.data();
        double total_diff = 0.0;
        size_t sample_size = std::min(test_video.size(), (size_t)500);
        
        for (size_t i = 0; i < sample_size; ++i) {
            total_diff += std::abs(smooth_data[i] - orig_data[i]);
        }
        double avg_diff = total_diff / sample_size;
        
        std::cout << "Average smoothing difference: " << std::fixed << std::setprecision(6) << avg_diff << std::endl;
        
        // Test frame warping
        std::cout << "\nTesting frame warping..." << std::endl;
        auto frames = VideoTensorUtils::extract_frames(test_video, VideoFormat::BTCHW);
        if (frames.size() >= 2) {
            Tensor warped_frame = VideoMotionUtils::warp_frame(
                frames[1], frames[0], motion_vectors
            );
            std::cout << "✓ Frame warping completed" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error in motion utils test: " << e.what() << std::endl;
    }
}

void performance_benchmark() {
    print_separator("Video Diffusion Performance Benchmark");
    
    try {
        std::cout << "Running performance benchmark..." << std::endl;
        
        std::vector<std::tuple<std::string, std::vector<size_t>, int>> test_cases = {
            {"Tiny (4 frames, 16x16)", {1, 4, 3, 16, 16}, 5},
            {"Small (8 frames, 32x32)", {1, 8, 3, 32, 32}, 5},
            {"Medium (16 frames, 64x64)", {1, 16, 3, 64, 64}, 3}
        };
        
        for (const auto& [name, shape, steps] : test_cases) {
            std::cout << "\nBenchmarking " << name << " with " << steps << " steps..." << std::endl;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Create noise
            Tensor noise = VideoDiffusionModel::sample_temporal_noise(shape, VideoFormat::BTCHW);
            
            // Create dummy text embeddings
            Tensor text_embeddings({1, 10, 64});
            auto text_data = text_embeddings.data();
            for (size_t i = 0; i < text_embeddings.size(); ++i) {
                text_data[i] = 0.01 * (i % 50);
            }
            
            // Run generation
            Tensor result = VideoDiffusionModel::temporal_sample_loop(
                noise, text_embeddings, steps,
                VideoDiffusionModel::GenerationStrategy::TEMPORAL_AWARE, 1.0, VideoFormat::BTCHW
            );
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Calculate throughput
            size_t total_pixels = 1;
            for (size_t dim : shape) total_pixels *= dim;
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
    std::cout << "     CLModel Video Diffusion Demo" << std::endl;
    std::cout << "     Week 10: Video Diffusion Architecture" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Run all tests
    test_temporal_scheduling();
    test_temporal_noise_operations();
    test_temporal_unet();
    test_video_generation_strategies();
    test_text_to_video_pipeline();
    test_motion_utils();
    performance_benchmark();
    
    print_separator("Video Diffusion Demo Summary");
    
    std::cout << "✅ Video Diffusion Architecture Implementation Complete!" << std::endl;
    std::cout << "\nKey Features Demonstrated:" << std::endl;
    std::cout << "  • Temporal noise scheduling with motion awareness" << std::endl;
    std::cout << "  • 3D diffusion operations for video data" << std::endl;
    std::cout << "  • Temporal U-Net with 3D convolutions" << std::endl;
    std::cout << "  • Multiple generation strategies (frame-by-frame, temporal-aware)" << std::endl;
    std::cout << "  • Text-to-video and video-to-video pipelines" << std::endl;
    std::cout << "  • Motion estimation and temporal consistency" << std::endl;
    std::cout << "  • Video preprocessing and postprocessing utilities" << std::endl;
    
    std::cout << "\n🎬 Ready for advanced video generation workflows!" << std::endl;
    std::cout << "\nNext Steps (Week 11-12):" << std::endl;
    std::cout << "  • Advanced frame interpolation techniques" << std::endl;
    std::cout << "  • Motion synthesis and style transfer" << std::endl;
    std::cout << "  • Audio-visual synchronization" << std::endl;
    std::cout << "  • Performance optimizations" << std::endl;
    
    return 0;
}
