#include "../include/ai/text_to_image.hpp"
#include "../include/ai/image_processing.hpp"
#include "../include/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace clmodel::ai;

void print_test_result(const std::string& test_name, bool passed, double time_ms = -1) {
    std::cout << "  " << std::setw(35) << std::left << test_name << ": ";
    if (passed) {
        std::cout << "PASS";
    } else {
        std::cout << "FAIL";
    }
    if (time_ms >= 0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << time_ms << "ms)";
    }
    std::cout << std::endl;
}

void test_text_processing() {
    std::cout << "\nTesting Text Processing..." << std::endl;
    
    // Test tokenization
    bool tokenize_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string test_text = "A beautiful cat sitting in the garden";
        auto tokens = TextProcessor::tokenize_words(test_text);
        
        if (tokens.size() != 7) {
            tokenize_pass = false;
        }
        
        // Test vocabulary building
        std::vector<std::string> corpus = {
            "a cat sat on the mat",
            "the dog ran in the park", 
            "a beautiful day in the garden"
        };
        
        auto vocab = TextProcessor::build_vocabulary(corpus, 100, 1);
        if (vocab.size() < 4) { // Should have at least special tokens
            tokenize_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Text Tokenization", tokenize_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Text Tokenization", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test embeddings
    bool embedding_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<int> token_ids = {2, 4, 5, 6, 3}; // BOS, a, cat, dog, EOS
        auto embeddings = TextProcessor::tokens_to_embeddings(token_ids, 128, 1000, 10);
        auto shape = embeddings.shape();
        
        if (shape.size() != 2 || shape[0] != 10 || shape[1] != 128) {
            embedding_pass = false;
        }
        
        // Test positional encoding
        auto pos_enc = TextProcessor::positional_encoding(10, 128);
        auto pos_shape = pos_enc.shape();
        
        if (pos_shape.size() != 2 || pos_shape[0] != 10 || pos_shape[1] != 128) {
            embedding_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Text Embeddings", embedding_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Text Embeddings", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test text cleaning
    bool cleaning_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string dirty_text = "Hello, World! This is a TEST...";
        auto cleaned = TextProcessor::clean_text(dirty_text);
        
        // Should be lowercase and no punctuation
        if (cleaned.find(',') != std::string::npos || 
            cleaned.find('!') != std::string::npos ||
            cleaned.find('.') != std::string::npos) {
            cleaning_pass = false;
        }
        
        // Should be lowercase
        if (cleaned.find('H') != std::string::npos || 
            cleaned.find('W') != std::string::npos ||
            cleaned.find("TEST") != std::string::npos) {
            cleaning_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Text Cleaning", cleaning_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Text Cleaning", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void test_diffusion_model() {
    std::cout << "\nTesting Diffusion Model..." << std::endl;
    
    // Test noise scheduling
    bool schedule_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto linear_betas = DiffusionModel::linear_schedule(100, 0.0001, 0.02);
        if (linear_betas.size() != 100 || linear_betas[0] != 0.0001 || linear_betas[99] != 0.02) {
            schedule_pass = false;
        }
        
        auto cosine_betas = DiffusionModel::cosine_schedule(50);
        if (cosine_betas.size() != 50) {
            schedule_pass = false;
        }
        
        auto [alphas, alpha_cumprod] = DiffusionModel::compute_alphas(linear_betas);
        if (alphas.size() != 100 || alpha_cumprod.size() != 100) {
            schedule_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Noise Scheduling", schedule_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Noise Scheduling", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test noise addition
    bool noise_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create a test image
        Tensor clean_image({3, 64, 64});
        for (size_t c = 0; c < 3; ++c) {
            for (size_t h = 0; h < 64; ++h) {
                for (size_t w = 0; w < 64; ++w) {
                    clean_image({c, h, w}) = 0.5; // Gray image
                }
            }
        }
        
        auto betas = DiffusionModel::linear_schedule(50);
        auto [alphas, alpha_cumprod] = DiffusionModel::compute_alphas(betas);
        
        // Add noise at different timesteps
        auto noisy_t10 = DiffusionModel::add_noise(clean_image, 10, alpha_cumprod);
        auto noisy_t40 = DiffusionModel::add_noise(clean_image, 40, alpha_cumprod);
        
        auto shape = noisy_t10.shape();
        if (shape.size() != 3 || shape[0] != 3 || shape[1] != 64 || shape[2] != 64) {
            noise_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Noise Addition", noise_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Noise Addition", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test noise sampling
    bool sampling_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto noise = DiffusionModel::sample_noise({2, 32, 32});
        auto shape = noise.shape();
        
        if (shape.size() != 3 || shape[0] != 2 || shape[1] != 32 || shape[2] != 32) {
            sampling_pass = false;
        }
        
        // Check that noise has reasonable variance (not all zeros or constant)
        double sum = 0.0;
        double sum_sq = 0.0;
        size_t count = 0;
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 32; ++j) {
                for (size_t k = 0; k < 32; ++k) {
                    double val = noise({i, j, k});
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
        }
        
        double mean = sum / count;
        double variance = (sum_sq / count) - (mean * mean);
        
        if (variance < 0.1) { // Should have reasonable variance
            sampling_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Noise Sampling", sampling_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Noise Sampling", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void test_text_to_image_pipeline() {
    std::cout << "\nTesting Text-to-Image Pipeline..." << std::endl;
    
    // Test pipeline initialization
    bool init_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        TextToImagePipeline pipeline(128, 20); // Small size and few steps for testing
        
        // Test vocabulary setting
        std::unordered_map<std::string, int> custom_vocab;
        custom_vocab["<UNK>"] = 0;
        custom_vocab["<PAD>"] = 1;
        custom_vocab["cat"] = 2;
        custom_vocab["dog"] = 3;
        pipeline.set_vocabulary(custom_vocab);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Pipeline Initialization", init_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Pipeline Initialization", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test single image generation
    bool generation_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        TextToImagePipeline pipeline(64, 5); // Very small for testing
        
        std::cout << "  Starting image generation test..." << std::endl;
        auto generated_image = pipeline.generate("a cat", 1.0, 42);
        
        auto shape = generated_image.shape();
        if (shape.size() != 3 || shape[0] != 3 || shape[1] != 64 || shape[2] != 64) {
            generation_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Single Image Generation", generation_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Single Image Generation", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
    
    // Test batch generation
    bool batch_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        TextToImagePipeline pipeline(32, 3); // Even smaller for batch test
        
        std::vector<std::string> prompts = {"a cat", "a dog"};
        std::cout << "  Starting batch generation test..." << std::endl;
        auto batch_images = pipeline.generate_batch(prompts, 1.0, 123);
        
        if (batch_images.size() != 2) {
            batch_pass = false;
        }
        
        for (const auto& image : batch_images) {
            auto shape = image.shape();
            if (shape.size() != 3 || shape[0] != 3 || shape[1] != 32 || shape[2] != 32) {
                batch_pass = false;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Batch Image Generation", batch_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Batch Image Generation", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void test_integration_with_image_processing() {
    std::cout << "\nTesting Integration with Image Processing..." << std::endl;
    
    bool integration_pass = true;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Generate a small image
        TextToImagePipeline pipeline(32, 2);
        auto generated_image = pipeline.generate("test", 1.0, 456);        // Test that we can use image processing on generated image
        auto normalized = ImageProcessor::normalize_to_float(generated_image);
        // Generated images are in channels-first format [3, height, width]
        auto resized = ImageProcessor::resize(normalized, 64, 64, InterpolationMethod::BILINEAR, true);
        
        auto shape = resized.shape();
        if (shape.size() != 3 || shape[0] != 3 || shape[1] != 64 || shape[2] != 64) {
            integration_pass = false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        print_test_result("Image Processing Integration", integration_pass, duration.count() / 1000.0);
    } catch (const std::exception& e) {
        print_test_result("Image Processing Integration", false);
        std::cout << "    Error: " << e.what() << std::endl;
    }
}

void demonstrate_text_to_image() {
    std::cout << "\n=== Text-to-Image Generation Demo ===" << std::endl;
    
    try {
        // Create pipeline with reasonable settings
        TextToImagePipeline pipeline(128, 10);
        
        std::vector<std::string> demo_prompts = {
            "a cute cat",
            "a red car",
            "a blue house"
        };
        
        std::cout << "Generating images for demonstration prompts..." << std::endl;
        
        for (const auto& prompt : demo_prompts) {
            std::cout << "\nPrompt: \"" << prompt << "\"" << std::endl;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto image = pipeline.generate(prompt, 2.0, 789);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            auto shape = image.shape();
            std::cout << "Generated image shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
            std::cout << "Generation time: " << duration.count() << " ms" << std::endl;
            
            // Show some basic statistics about the generated image
            double min_val = 1e6, max_val = -1e6, sum = 0.0;
            size_t count = 0;
            
            for (size_t c = 0; c < shape[0]; ++c) {
                for (size_t h = 0; h < shape[1]; ++h) {
                    for (size_t w = 0; w < shape[2]; ++w) {
                        double val = image({c, h, w});
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                        sum += val;
                        count++;
                    }
                }
            }
            
            double mean = sum / count;
            std::cout << "Image statistics - Min: " << std::fixed << std::setprecision(4) << min_val 
                     << ", Max: " << max_val << ", Mean: " << mean << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Demo error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "CLModel Phase 2: Text-to-Image Pipeline Tests" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    test_text_processing();
    test_diffusion_model();
    test_text_to_image_pipeline();
    test_integration_with_image_processing();
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\nText-to-Image Pipeline Tests Complete" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    
    // Run demonstration
    demonstrate_text_to_image();
    
    std::cout << "\n🎨 Phase 2 Week 5-8: Text-to-Image Pipeline foundation implemented!" << std::endl;
    std::cout << "📝 Note: This is a simplified educational implementation" << std::endl;
    std::cout << "🚀 Next: Text-to-Audio pipeline (Week 9-12)" << std::endl;
    
    return 0;
}
