#include "include/ai/text_to_image.hpp"
#include "include/ai/image_processing.hpp"
#include <iostream>

using namespace clmodel::ai;

int main() {
    std::cout << "Testing Text-to-Image Integration Debug\n";
    
    try {
        // Generate a small image
        std::cout << "Creating pipeline...\n";
        TextToImagePipeline pipeline(32, 2);
        
        std::cout << "Generating image...\n";
        auto generated_image = pipeline.generate("test", 1.0, 456);
        
        std::cout << "Generated image shape: ";
        auto gen_shape = generated_image.shape();
        for (size_t i = 0; i < gen_shape.size(); ++i) {
            std::cout << gen_shape[i];
            if (i < gen_shape.size() - 1) std::cout << "x";
        }
        std::cout << "\n";
        
        // Check value ranges
        double min_val = generated_image.data()[0];
        double max_val = generated_image.data()[0];
        for (size_t i = 0; i < generated_image.size(); ++i) {
            min_val = std::min(min_val, generated_image.data()[i]);
            max_val = std::max(max_val, generated_image.data()[i]);
        }
        std::cout << "Value range: [" << min_val << ", " << max_val << "]\n";
        
        // Test normalization
        std::cout << "Testing normalization...\n";
        auto normalized = ImageProcessor::normalize_to_float(generated_image);
        std::cout << "Normalized shape: ";
        auto norm_shape = normalized.shape();
        for (size_t i = 0; i < norm_shape.size(); ++i) {
            std::cout << norm_shape[i];
            if (i < norm_shape.size() - 1) std::cout << "x";
        }
        std::cout << "\n";
        
        // Check normalized ranges
        min_val = normalized.data()[0];
        max_val = normalized.data()[0];
        for (size_t i = 0; i < normalized.size(); ++i) {
            min_val = std::min(min_val, normalized.data()[i]);
            max_val = std::max(max_val, normalized.data()[i]);
        }
        std::cout << "Normalized range: [" << min_val << ", " << max_val << "]\n";
          // Test resizing (assume channels-first format: [3, 32, 32])
        std::cout << "Testing resize (channels-first)...\n";
        auto resized = ImageProcessor::resize(normalized, 64, 64, InterpolationMethod::BILINEAR, true);
        std::cout << "Resized shape: ";
        auto resize_shape = resized.shape();
        for (size_t i = 0; i < resize_shape.size(); ++i) {
            std::cout << resize_shape[i];
            if (i < resize_shape.size() - 1) std::cout << "x";
        }
        std::cout << "\n";
        
        // Final check
        if (resize_shape.size() == 3 && resize_shape[0] == 3 && resize_shape[1] == 64 && resize_shape[2] == 64) {
            std::cout << "Integration test: PASS\n";
        } else {
            std::cout << "Integration test: FAIL\n";
            std::cout << "Expected shape: [3, 64, 64]\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
