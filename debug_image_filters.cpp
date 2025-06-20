#include "include/ai/image_processing.hpp"
#include <iostream>

using namespace clmodel::ai;

int main() {
    std::cout << "Testing Image Filters Debug\n";
    
    try {
        // Create test image
        Tensor image({5, 5, 1});
        
        // Create a simple pattern (center spike)
        for (size_t h = 0; h < 5; ++h) {
            for (size_t w = 0; w < 5; ++w) {
                image({h, w, 0}) = (h == 2 && w == 2) ? 1.0 : 0.0;
            }
        }
        
        std::cout << "Original image:\n";
        for (size_t h = 0; h < 5; ++h) {
            for (size_t w = 0; w < 5; ++w) {
                std::cout << image({h, w, 0}) << " ";
            }
            std::cout << "\n";
        }
          // Test Gaussian kernel (using public API)
        std::cout << "\nTesting Gaussian blur directly:\n";
        
        // Test Gaussian blur
        std::cout << "\nApplying Gaussian blur:\n";
        Tensor blurred = ImageProcessor::gaussian_blur(image, 3, 1.0, false);
        
        std::cout << "Blurred image:\n";
        for (size_t h = 0; h < 5; ++h) {
            for (size_t w = 0; w < 5; ++w) {
                std::cout << blurred({h, w, 0}) << " ";
            }
            std::cout << "\n";
        }
        
        // Center should still be highest but not 1.0
        double center_val = blurred({2, 2, 0});
        std::cout << "\nCenter value: " << center_val << "\n";
        std::cout << "Test condition: " << (center_val > 0.5 && center_val < 1.0) << "\n";
        
        // Test brightness adjustment
        std::cout << "\nTesting brightness adjustment:\n";
        Tensor brightened = ImageProcessor::adjust_brightness(image, 2.0);
        double bright_center = brightened({2, 2, 0});
        std::cout << "Original center: " << image({2, 2, 0}) << "\n";
        std::cout << "Brightened center: " << bright_center << "\n";
        std::cout << "Should be clamped to 1.0: " << (bright_center <= 1.0) << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
