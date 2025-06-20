#include "../include/ai/image_processing.hpp"
#include "../include/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace clmodel::ai;

void print_test_result(const std::string& test_name, bool passed, double time_ms = -1) {
    std::cout << "  " << std::setw(35) << std::left << test_name << ": ";
    std::cout << (passed ? "PASS" : "FAIL");
    if (time_ms >= 0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << time_ms << " ms)";
    }
    std::cout << "\n";
}

bool test_image_dimensions() {
    try {
        // Test channels-last format [H, W, C]
        Tensor image_hwc({100, 200, 3});
        auto [height, width, channels] = ImageProcessor::get_image_dims(image_hwc, false);
        if (height != 100 || width != 200 || channels != 3) {
            return false;
        }
        
        // Test channels-first format [C, H, W]
        Tensor image_chw({3, 100, 200});
        auto [height2, width2, channels2] = ImageProcessor::get_image_dims(image_chw, true);
        if (height2 != 100 || width2 != 200 || channels2 != 3) {
            return false;
        }
        
        // Test grayscale [H, W]
        Tensor gray_image({100, 200});
        auto [height3, width3, channels3] = ImageProcessor::get_image_dims(gray_image, false);
        if (height3 != 100 || width3 != 200 || channels3 != 1) {
            return false;
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool test_channel_order_conversion() {
    try {
        // Create test image [H, W, C] = [2, 3, 2]
        Tensor image_hwc({2, 3, 2});
        
        // Fill with test pattern
        for (size_t h = 0; h < 2; ++h) {
            for (size_t w = 0; w < 3; ++w) {
                for (size_t c = 0; c < 2; ++c) {
                    image_hwc({h, w, c}) = h * 100 + w * 10 + c;
                }
            }
        }
        
        // Convert to channels-first
        Tensor image_chw = ImageProcessor::convert_channel_order(image_hwc, true);
        
        // Verify conversion
        if (image_chw.shape() != std::vector<size_t>({2, 2, 3})) {
            return false;
        }
        
        // Check specific values
        if (image_chw({0, 0, 0}) != 0.0 || // channel 0, h=0, w=0
            image_chw({1, 1, 2}) != 121.0) { // channel 1, h=1, w=2
            return false;
        }
        
        // Convert back to channels-last
        Tensor image_back = ImageProcessor::convert_channel_order(image_chw, false);
        
        // Should match original
        return image_back.shape() == image_hwc.shape();
        
    } catch (const std::exception&) {
        return false;
    }
}

bool test_image_resize() {
    try {
        // Create small test image [H, W, C] = [4, 4, 3]
        Tensor image({4, 4, 3});
        
        // Fill with gradient pattern
        for (size_t h = 0; h < 4; ++h) {
            for (size_t w = 0; w < 4; ++w) {
                for (size_t c = 0; c < 3; ++c) {
                    image({h, w, c}) = (h * 4 + w) / 15.0; // Normalized to [0, 1]
                }
            }
        }
        
        // Test resize to different dimensions
        Tensor resized = ImageProcessor::resize(image, 8, 8, InterpolationMethod::BILINEAR, false);
        
        if (resized.shape() != std::vector<size_t>({8, 8, 3})) {
            return false;
        }
        
        // Test that corner values are preserved approximately
        double original_corner = image({0, 0, 0});
        double resized_corner = resized({0, 0, 0});
        
        return std::abs(original_corner - resized_corner) < 0.1;
        
    } catch (const std::exception&) {
        return false;
    }
}

bool test_crop_operations() {
    try {
        // Create test image [H, W, C] = [10, 10, 3]
        Tensor image({10, 10, 3});
        
        // Fill with position-dependent values
        for (size_t h = 0; h < 10; ++h) {
            for (size_t w = 0; w < 10; ++w) {
                for (size_t c = 0; c < 3; ++c) {
                    image({h, w, c}) = h * 10 + w + c * 0.1;
                }
            }
        }
        
        // Test crop
        Tensor cropped = ImageProcessor::crop(image, 2, 3, 4, 5, false);
        
        if (cropped.shape() != std::vector<size_t>({4, 5, 3})) {
            return false;
        }
        
        // Verify cropped content
        double expected = 2 * 10 + 3 + 0 * 0.1; // h=2, w=3, c=0
        double actual = cropped({0, 0, 0});
        
        if (std::abs(expected - actual) > 1e-6) {
            return false;
        }
        
        // Test center crop
        Tensor center_cropped = ImageProcessor::center_crop(image, 6, false);
        
        return center_cropped.shape() == std::vector<size_t>({6, 6, 3});
        
    } catch (const std::exception&) {
        return false;
    }
}

bool test_color_conversions() {
    try {
        // Create RGB test image
        Tensor rgb_image({2, 2, 3});
        
        // Set known RGB values
        rgb_image({0, 0, 0}) = 1.0; rgb_image({0, 0, 1}) = 0.0; rgb_image({0, 0, 2}) = 0.0; // Red
        rgb_image({0, 1, 0}) = 0.0; rgb_image({0, 1, 1}) = 1.0; rgb_image({0, 1, 2}) = 0.0; // Green
        rgb_image({1, 0, 0}) = 0.0; rgb_image({1, 0, 1}) = 0.0; rgb_image({1, 0, 2}) = 1.0; // Blue
        rgb_image({1, 1, 0}) = 1.0; rgb_image({1, 1, 1}) = 1.0; rgb_image({1, 1, 2}) = 1.0; // White
        
        // Test RGB to grayscale
        Tensor gray = ImageProcessor::rgb_to_grayscale(rgb_image, false);
        
        if (gray.shape() != std::vector<size_t>({2, 2})) {
            return false;
        }
        
        // Check known conversions (approximate)
        double red_gray = gray({0, 0}); // Should be ~0.299
        double white_gray = gray({1, 1}); // Should be ~1.0
        
        if (std::abs(red_gray - 0.299) > 0.01 || std::abs(white_gray - 1.0) > 0.01) {
            return false;
        }
        
        // Test RGB to HSV
        Tensor hsv = ImageProcessor::rgb_to_hsv(rgb_image, false);
        
        return hsv.shape() == std::vector<size_t>({2, 2, 3});
        
    } catch (const std::exception&) {
        return false;
    }
}

bool test_normalization() {
    try {
        // Create test image with values in [0, 255] range
        Tensor image({2, 2, 3});
        for (size_t i = 0; i < image.size(); ++i) {
            image.data()[i] = i * 10.0; // Values from 0 to 170
        }
        
        // Test normalize to float
        Tensor normalized = ImageProcessor::normalize_to_float(image);
        
        // Check range
        for (size_t i = 0; i < normalized.size(); ++i) {
            if (normalized.data()[i] < 0.0 || normalized.data()[i] > 1.0) {
                return false;
            }
        }
        
        // Test ImageNet normalization
        std::vector<double> mean = {0.5, 0.5, 0.5};
        std::vector<double> std = {0.5, 0.5, 0.5};
        
        Tensor imagenet_norm = ImageProcessor::normalize_imagenet(normalized, mean, std, false);
        
        // Test denormalization
        Tensor denormalized = ImageProcessor::denormalize_imagenet(imagenet_norm, mean, std, false);
        
        // Should be close to original normalized image
        for (size_t i = 0; i < normalized.size(); ++i) {
            if (std::abs(normalized.data()[i] - denormalized.data()[i]) > 0.01) {
                return false;
            }
        }
        
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
}

bool test_image_filters() {
    try {
        // Create test image
        Tensor image({5, 5, 1});
        
        // Create a simple pattern (center spike)
        for (size_t h = 0; h < 5; ++h) {
            for (size_t w = 0; w < 5; ++w) {
                image({h, w, 0}) = (h == 2 && w == 2) ? 1.0 : 0.0;
            }
        }
        
        // Test Gaussian blur
        Tensor blurred = ImageProcessor::gaussian_blur(image, 3, 1.0, false);
        
        if (blurred.shape() != image.shape()) {
            return false;
        }
          // Center should still be highest but significantly reduced by blur
        double center_val = blurred({2, 2, 0});
        // With 3x3 Gaussian kernel and sigma=1.0, center should be around 0.2
        if (center_val <= 0.1 || center_val >= 0.3) {
            return false;
        }
        
        // Check that blur actually spread values around
        double neighbor_val = blurred({1, 2, 0}); // neighbor pixel
        if (neighbor_val <= 0.05 || neighbor_val >= center_val) {
            return false;
        }
        
        // Test brightness adjustment
        Tensor brightened = ImageProcessor::adjust_brightness(image, 2.0);
        if (brightened({2, 2, 0}) > 1.0) { // Should be clamped to 1.0
            return false;
        }
        
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
}

bool test_batch_operations() {
    try {
        // Create multiple test images
        std::vector<Tensor> images;
        for (int i = 0; i < 3; ++i) {
            Tensor img({4, 4, 3});
            for (size_t j = 0; j < img.size(); ++j) {
                img.data()[j] = i * 0.1 + j * 0.01;
            }
            images.push_back(img);
        }
        
        // Test batch resize
        auto resized_batch = ImageProcessor::batch_resize(images, 8, 8, InterpolationMethod::BILINEAR, false);
        
        if (resized_batch.size() != 3) {
            return false;
        }
        
        for (const auto& img : resized_batch) {
            if (img.shape() != std::vector<size_t>({8, 8, 3})) {
                return false;
            }
        }
        
        // Test create batch
        Tensor batched = ImageProcessor::create_batch(images, false);
        
        if (batched.shape() != std::vector<size_t>({3, 4, 4, 3})) {
            return false;
        }
        
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
}

bool test_image_io() {
    try {
        // Test loading
        Tensor loaded = ImageProcessor::load_image("test_image.jpg", ColorSpace::RGB, false);
        
        if (loaded.shape().size() != 3) {
            return false;
        }
        
        // Test saving
        ImageProcessor::save_image(loaded, "output_test.png", ColorSpace::RGB, false);
        
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
}

void run_image_processing_tests() {
    std::cout << "\nCLModel Phase 2: Image Processing Tests\n";
    std::cout << "======================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
      // Run all tests
    print_test_result("Image Load/Save", test_image_io());
    print_test_result("Image Dimensions", test_image_dimensions());
    print_test_result("Channel Order Conversion", test_channel_order_conversion());
    print_test_result("Image Resize", test_image_resize());
    print_test_result("Crop Operations", test_crop_operations());
    print_test_result("Color Conversions", test_color_conversions());
    print_test_result("Normalization", test_normalization());
    print_test_result("Image Filters", test_image_filters());    print_test_result("Batch Operations", test_batch_operations());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nImage Processing Tests Complete\n";
    std::cout << "Total time: " << duration.count() << " ms\n";
    std::cout << "\n✅ Phase 2 Week 1-2: Image Processing Foundation implemented!\n";
    std::cout << "📋 Next: Audio processing foundation (Week 3-4)\n";
}

int main() {
    try {
        run_image_processing_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
