#pragma once

#include "../tensor.hpp"
#include <string>
#include <vector>
#include <memory>

namespace asekioml {
namespace ai {

/**
 * @brief Color space definitions
 */
enum class ColorSpace {
    RGB,
    BGR,
    RGBA,
    BGRA,
    GRAYSCALE,
    HSV,
    LAB
};

/**
 * @brief Image interpolation methods for resizing
 */
enum class InterpolationMethod {
    NEAREST,
    BILINEAR,
    BICUBIC
};

/**
 * @brief Advanced image processing operations built on the Tensor class
 * 
 * Provides high-performance image manipulation capabilities for AI pipelines,
 * including preprocessing for computer vision models and image generation.
 */
class ImageProcessor {
public:
    // ===== Image Loading and Saving =====
    
    /**
     * @brief Load image from file to tensor
     * @param filepath Path to image file (PNG, JPEG, etc.)
     * @param color_space Target color space for loading
     * @return Tensor with shape [height, width, channels] or [channels, height, width]
     */
    static Tensor load_image(const std::string& filepath, 
                           ColorSpace color_space = ColorSpace::RGB,
                           bool channels_first = false);
    
    /**
     * @brief Save tensor as image file
     * @param tensor Image tensor to save
     * @param filepath Output file path
     * @param color_space Color space of the tensor data
     * @param channels_first Whether tensor is in [C, H, W] format
     */
    static void save_image(const Tensor& tensor, 
                          const std::string& filepath,
                          ColorSpace color_space = ColorSpace::RGB,
                          bool channels_first = false);
    
    // ===== Geometric Transformations =====
    
    /**
     * @brief Resize image tensor to new dimensions
     * @param image Input image tensor [H, W, C] or [C, H, W]
     * @param new_height Target height
     * @param new_width Target width
     * @param method Interpolation method
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Resized image tensor
     */
    static Tensor resize(const Tensor& image, 
                        size_t new_height, 
                        size_t new_width,
                        InterpolationMethod method = InterpolationMethod::BILINEAR,
                        bool channels_first = false);
    
    /**
     * @brief Crop image tensor to specified region
     * @param image Input image tensor
     * @param top Top coordinate
     * @param left Left coordinate
     * @param height Crop height
     * @param width Crop width
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Cropped image tensor
     */
    static Tensor crop(const Tensor& image,
                      size_t top, size_t left,
                      size_t height, size_t width,
                      bool channels_first = false);
    
    /**
     * @brief Center crop image to square aspect ratio
     * @param image Input image tensor
     * @param size Target square size
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Center-cropped square image
     */
    static Tensor center_crop(const Tensor& image, 
                             size_t size,
                             bool channels_first = false);
    
    // ===== Color Space Conversions =====
    
    /**
     * @brief Convert between color spaces
     * @param image Input image tensor
     * @param from_space Source color space
     * @param to_space Target color space
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Converted image tensor
     */
    static Tensor convert_color_space(const Tensor& image,
                                    ColorSpace from_space,
                                    ColorSpace to_space,
                                    bool channels_first = false);
    
    /**
     * @brief Convert RGB to grayscale
     * @param rgb_image RGB image tensor
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Grayscale image tensor
     */
    static Tensor rgb_to_grayscale(const Tensor& rgb_image,
                                  bool channels_first = false);
    
    /**
     * @brief Convert RGB to HSV
     * @param rgb_image RGB image tensor
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return HSV image tensor
     */
    static Tensor rgb_to_hsv(const Tensor& rgb_image,
                            bool channels_first = false);
    
    // ===== Normalization and Preprocessing =====
    
    /**
     * @brief Normalize image tensor to [0, 1] range
     * @param image Input image tensor (typically 0-255)
     * @return Normalized image tensor
     */
    static Tensor normalize_to_float(const Tensor& image);
    
    /**
     * @brief Normalize image with mean and std (ImageNet-style)
     * @param image Input image tensor [0, 1]
     * @param mean Per-channel mean values
     * @param std Per-channel standard deviation values
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Normalized image tensor
     */
    static Tensor normalize_imagenet(const Tensor& image,
                                   const std::vector<double>& mean = {0.485, 0.456, 0.406},
                                   const std::vector<double>& std = {0.229, 0.224, 0.225},
                                   bool channels_first = false);
    
    /**
     * @brief Denormalize ImageNet-normalized tensor back to [0, 1]
     * @param normalized_image Normalized image tensor
     * @param mean Per-channel mean values used for normalization
     * @param std Per-channel standard deviation values used for normalization
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Denormalized image tensor
     */
    static Tensor denormalize_imagenet(const Tensor& normalized_image,
                                     const std::vector<double>& mean = {0.485, 0.456, 0.406},
                                     const std::vector<double>& std = {0.229, 0.224, 0.225},
                                     bool channels_first = false);
    
    // ===== Image Filters and Effects =====
    
    /**
     * @brief Apply Gaussian blur to image
     * @param image Input image tensor
     * @param kernel_size Blur kernel size (odd number)
     * @param sigma Gaussian sigma value
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Blurred image tensor
     */
    static Tensor gaussian_blur(const Tensor& image,
                               size_t kernel_size = 5,
                               double sigma = 1.0,
                               bool channels_first = false);
    
    /**
     * @brief Apply edge detection filter
     * @param image Input image tensor (preferably grayscale)
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Edge-detected image tensor
     */
    static Tensor edge_detection(const Tensor& image,
                                bool channels_first = false);
    
    /**
     * @brief Adjust image brightness
     * @param image Input image tensor
     * @param factor Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
     * @return Brightness-adjusted image tensor
     */
    static Tensor adjust_brightness(const Tensor& image, double factor);
    
    /**
     * @brief Adjust image contrast
     * @param image Input image tensor
     * @param factor Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)
     * @return Contrast-adjusted image tensor
     */
    static Tensor adjust_contrast(const Tensor& image, double factor);
    
    // ===== Batch Operations =====
    
    /**
     * @brief Batch resize multiple images
     * @param images Vector of image tensors
     * @param new_height Target height
     * @param new_width Target width
     * @param method Interpolation method
     * @param channels_first Whether tensors are in [C, H, W] format
     * @return Vector of resized image tensors
     */
    static std::vector<Tensor> batch_resize(const std::vector<Tensor>& images,
                                           size_t new_height,
                                           size_t new_width,
                                           InterpolationMethod method = InterpolationMethod::BILINEAR,
                                           bool channels_first = false);
    
    /**
     * @brief Create a batched tensor from multiple images
     * @param images Vector of image tensors (must have same dimensions)
     * @param channels_first Whether tensors are in [C, H, W] format
     * @return Batched tensor [N, C, H, W] or [N, H, W, C]
     */
    static Tensor create_batch(const std::vector<Tensor>& images,
                              bool channels_first = false);
    
    // ===== Data Augmentation =====
    
    /**
     * @brief Randomly flip image horizontally
     * @param image Input image tensor
     * @param probability Probability of flipping (0.0 to 1.0)
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Possibly flipped image tensor
     */
    static Tensor random_horizontal_flip(const Tensor& image,
                                        double probability = 0.5,
                                        bool channels_first = false);
    
    /**
     * @brief Randomly rotate image
     * @param image Input image tensor
     * @param max_angle Maximum rotation angle in degrees
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Rotated image tensor
     */
    static Tensor random_rotation(const Tensor& image,
                                 double max_angle = 15.0,
                                 bool channels_first = false);
    
    // ===== Utility Functions =====
    
    /**
     * @brief Get image dimensions from tensor
     * @param image Image tensor
     * @param channels_first Whether tensor is in [C, H, W] format
     * @return Tuple of (height, width, channels)
     */
    static std::tuple<size_t, size_t, size_t> get_image_dims(const Tensor& image,
                                                            bool channels_first = false);
    
    /**
     * @brief Convert between channels-first and channels-last formats
     * @param image Input image tensor
     * @param to_channels_first Target format
     * @return Converted image tensor
     */
    static Tensor convert_channel_order(const Tensor& image, bool to_channels_first);
    
    /**
     * @brief Clamp tensor values to valid image range [0, 1] or [0, 255]
     * @param image Input image tensor
     * @param min_val Minimum value
     * @param max_val Maximum value
     * @return Clamped image tensor
     */
    static Tensor clamp_values(const Tensor& image, double min_val = 0.0, double max_val = 1.0);

private:
    // Helper functions for internal use
    static Tensor apply_2d_convolution(const Tensor& image, const Tensor& kernel, bool channels_first = false);
    static std::pair<size_t, size_t> get_hw_dims(const Tensor& image, bool channels_first);
    static size_t get_channel_dim(const Tensor& image, bool channels_first);
    static Tensor create_gaussian_kernel(size_t size, double sigma);
    static double bilinear_interpolate(const Tensor& image, double y, double x, size_t channel, bool channels_first);
};

} // namespace ai
} // namespace asekioml
