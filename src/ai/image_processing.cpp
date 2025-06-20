#include "../../include/ai/image_processing.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {
namespace ai {

// ===== Utility Functions =====

std::tuple<size_t, size_t, size_t> ImageProcessor::get_image_dims(const Tensor& image, bool channels_first) {
    auto shape = image.shape();
    
    if (shape.size() == 3) {
        if (channels_first) {
            return std::make_tuple(shape[1], shape[2], shape[0]); // [C, H, W] -> (H, W, C)
        } else {
            return std::make_tuple(shape[0], shape[1], shape[2]); // [H, W, C] -> (H, W, C)
        }
    } else if (shape.size() == 2) {
        return std::make_tuple(shape[0], shape[1], 1); // [H, W] -> (H, W, 1)
    } else {
        throw std::invalid_argument("Image tensor must be 2D or 3D");
    }
}

std::pair<size_t, size_t> ImageProcessor::get_hw_dims(const Tensor& image, bool channels_first) {
    auto shape = image.shape();
    
    if (shape.size() == 3) {
        if (channels_first) {
            return {shape[1], shape[2]}; // [C, H, W]
        } else {
            return {shape[0], shape[1]}; // [H, W, C]
        }
    } else if (shape.size() == 2) {
        return {shape[0], shape[1]}; // [H, W]
    } else {
        throw std::invalid_argument("Image tensor must be 2D or 3D");
    }
}

size_t ImageProcessor::get_channel_dim(const Tensor& image, bool channels_first) {
    auto shape = image.shape();
    
    if (shape.size() == 3) {
        if (channels_first) {
            return shape[0]; // [C, H, W]
        } else {
            return shape[2]; // [H, W, C]
        }
    } else if (shape.size() == 2) {
        return 1; // [H, W] - grayscale
    } else {
        throw std::invalid_argument("Image tensor must be 2D or 3D");
    }
}

Tensor ImageProcessor::convert_channel_order(const Tensor& image, bool to_channels_first) {
    auto shape = image.shape();
    
    if (shape.size() == 2) {
        return image; // No change needed for 2D images
    } else if (shape.size() == 3) {
        if (to_channels_first) {
            // [H, W, C] -> [C, H, W]
            return image.transpose({2, 0, 1});
        } else {
            // [C, H, W] -> [H, W, C]
            return image.transpose({1, 2, 0});
        }
    } else {
        throw std::invalid_argument("Image tensor must be 2D or 3D");
    }
}

// ===== Image Loading and Saving =====

Tensor ImageProcessor::load_image(const std::string& filepath, ColorSpace color_space, bool channels_first) {
    // Placeholder implementation - generates a test pattern
    // TODO: Implement actual image loading with third-party libraries (e.g., stb_image, OpenCV, FreeImage)
    
    // For now, create a test checkerboard pattern
    size_t height = 256;
    size_t width = 256;
    size_t channels = (color_space == ColorSpace::GRAYSCALE) ? 1 : 3;
    
    std::vector<size_t> shape;
    if (channels_first) {
        shape = {channels, height, width};
    } else {
        shape = {height, width, channels};
    }
    
    Tensor image(shape);
    
    // Create checkerboard pattern
    size_t square_size = 32;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            bool is_white = ((y / square_size) + (x / square_size)) % 2 == 0;
            double value = is_white ? 0.8 : 0.2;
            
            for (size_t c = 0; c < channels; ++c) {
                if (channels_first) {
                    image({c, y, x}) = value;
                } else {
                    image({y, x, c}) = value;
                }
            }
        }
    }
    
    return image;
}

void ImageProcessor::save_image(const Tensor& tensor, const std::string& filepath, 
                               ColorSpace color_space, bool channels_first) {
    // Placeholder implementation - logs the save operation
    // TODO: Implement actual image saving with third-party libraries
    
    auto shape = tensor.shape();
    auto [height, width, channels] = get_image_dims(tensor, channels_first);
    
    // Calculate basic statistics
    double min_val = tensor.data()[0];
    double max_val = tensor.data()[0];
    double sum = 0.0;
    
    for (size_t i = 0; i < tensor.size(); ++i) {
        double val = tensor.data()[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    double mean = sum / tensor.size();
    
    // Log the operation (in a real implementation, this would save the actual file)
    std::cout << "ImageProcessor::save_image() - Placeholder Implementation\n";
    std::cout << "  File: " << filepath << "\n";
    std::cout << "  Dimensions: " << height << "x" << width << "x" << channels << "\n";
    std::cout << "  Format: " << (channels_first ? "CHW" : "HWC") << "\n";
    std::cout << "  Value range: [" << min_val << ", " << max_val << "] (mean: " << mean << ")\n";
    std::cout << "  Note: Actual file saving not implemented yet\n";
}

// ===== Geometric Transformations =====

Tensor ImageProcessor::resize(const Tensor& image, size_t new_height, size_t new_width,
                             InterpolationMethod method, bool channels_first) {
    auto [height, width, channels] = get_image_dims(image, channels_first);
    
    if (height == new_height && width == new_width) {
        return image; // No resize needed
    }
    
    // Create output tensor
    std::vector<size_t> output_shape;
    if (channels_first) {
        output_shape = {channels, new_height, new_width};
    } else {
        output_shape = {new_height, new_width, channels};
    }
    
    Tensor result(output_shape);
    
    // Calculate scaling factors
    double y_scale = static_cast<double>(height) / new_height;
    double x_scale = static_cast<double>(width) / new_width;
    
    // Resize each channel
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < new_height; ++y) {
            for (size_t x = 0; x < new_width; ++x) {
                double src_y = (y + 0.5) * y_scale - 0.5;
                double src_x = (x + 0.5) * x_scale - 0.5;
                
                double value = 0.0;
                
                if (method == InterpolationMethod::NEAREST) {
                    size_t nearest_y = static_cast<size_t>(std::round(src_y));
                    size_t nearest_x = static_cast<size_t>(std::round(src_x));
                    
                    nearest_y = std::min(nearest_y, height - 1);
                    nearest_x = std::min(nearest_x, width - 1);
                    
                    if (channels_first) {
                        value = image({c, nearest_y, nearest_x});
                    } else {
                        value = image({nearest_y, nearest_x, c});
                    }
                } else if (method == InterpolationMethod::BILINEAR) {
                    value = bilinear_interpolate(image, src_y, src_x, c, channels_first);
                }
                
                if (channels_first) {
                    result({c, y, x}) = value;
                } else {
                    result({y, x, c}) = value;
                }
            }
        }
    }
    
    return result;
}

double ImageProcessor::bilinear_interpolate(const Tensor& image, double y, double x, size_t channel, bool channels_first) {
    auto [height, width, channels] = get_image_dims(image, channels_first);
    
    // Clamp coordinates
    y = std::max(0.0, std::min(static_cast<double>(height - 1), y));
    x = std::max(0.0, std::min(static_cast<double>(width - 1), x));
    
    size_t y0 = static_cast<size_t>(std::floor(y));
    size_t y1 = std::min(y0 + 1, height - 1);
    size_t x0 = static_cast<size_t>(std::floor(x));
    size_t x1 = std::min(x0 + 1, width - 1);
    
    double dy = y - y0;
    double dx = x - x0;
    
    // Get corner values
    double val00, val01, val10, val11;
    
    if (channels_first) {
        val00 = image({channel, y0, x0});
        val01 = image({channel, y0, x1});
        val10 = image({channel, y1, x0});
        val11 = image({channel, y1, x1});
    } else {
        val00 = image({y0, x0, channel});
        val01 = image({y0, x1, channel});
        val10 = image({y1, x0, channel});
        val11 = image({y1, x1, channel});
    }
    
    // Bilinear interpolation
    double val0 = val00 * (1 - dx) + val01 * dx;
    double val1 = val10 * (1 - dx) + val11 * dx;
    
    return val0 * (1 - dy) + val1 * dy;
}

Tensor ImageProcessor::crop(const Tensor& image, size_t top, size_t left,
                           size_t height, size_t width, bool channels_first) {
    auto [img_height, img_width, channels] = get_image_dims(image, channels_first);
    
    // Validate crop parameters
    if (top + height > img_height || left + width > img_width) {
        throw std::invalid_argument("Crop region exceeds image boundaries");
    }
    
    // Create output tensor
    std::vector<size_t> output_shape;
    if (channels_first) {
        output_shape = {channels, height, width};
    } else {
        output_shape = {height, width, channels};
    }
    
    Tensor result(output_shape);
    
    // Copy cropped region
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                double value;
                if (channels_first) {
                    value = image({c, top + y, left + x});
                    result({c, y, x}) = value;
                } else {
                    value = image({top + y, left + x, c});
                    result({y, x, c}) = value;
                }
            }
        }
    }
    
    return result;
}

Tensor ImageProcessor::center_crop(const Tensor& image, size_t size, bool channels_first) {
    auto [height, width, channels] = get_image_dims(image, channels_first);
    
    if (height < size || width < size) {
        throw std::invalid_argument("Center crop size larger than image dimensions");
    }
    
    size_t top = (height - size) / 2;
    size_t left = (width - size) / 2;
    
    return crop(image, top, left, size, size, channels_first);
}

// ===== Color Space Conversions =====

Tensor ImageProcessor::rgb_to_grayscale(const Tensor& rgb_image, bool channels_first) {
    auto [height, width, channels] = get_image_dims(rgb_image, channels_first);
    
    if (channels != 3) {
        throw std::invalid_argument("RGB to grayscale conversion requires 3-channel image");
    }
    
    Tensor result({height, width});
    
    // Standard RGB to grayscale conversion weights
    const double r_weight = 0.299;
    const double g_weight = 0.587;
    const double b_weight = 0.114;
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            double r, g, b;
            
            if (channels_first) {
                r = rgb_image({0, y, x});
                g = rgb_image({1, y, x});
                b = rgb_image({2, y, x});
            } else {
                r = rgb_image({y, x, 0});
                g = rgb_image({y, x, 1});
                b = rgb_image({y, x, 2});
            }
            
            double gray = r * r_weight + g * g_weight + b * b_weight;
            result({y, x}) = gray;
        }
    }
    
    return result;
}

Tensor ImageProcessor::rgb_to_hsv(const Tensor& rgb_image, bool channels_first) {
    auto [height, width, channels] = get_image_dims(rgb_image, channels_first);
    
    if (channels != 3) {
        throw std::invalid_argument("RGB to HSV conversion requires 3-channel image");
    }
    
    std::vector<size_t> output_shape;
    if (channels_first) {
        output_shape = {3, height, width};
    } else {
        output_shape = {height, width, 3};
    }
    
    Tensor result(output_shape);
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            double r, g, b;
            
            if (channels_first) {
                r = rgb_image({0, y, x});
                g = rgb_image({1, y, x});
                b = rgb_image({2, y, x});
            } else {
                r = rgb_image({y, x, 0});
                g = rgb_image({y, x, 1});
                b = rgb_image({y, x, 2});
            }
            
            double max_val = std::max({r, g, b});
            double min_val = std::min({r, g, b});
            double delta = max_val - min_val;
            
            // Hue calculation
            double h = 0.0;
            if (delta != 0.0) {
                if (max_val == r) {
                    h = 60.0 * fmod(((g - b) / delta), 6.0);
                } else if (max_val == g) {
                    h = 60.0 * (((b - r) / delta) + 2.0);
                } else if (max_val == b) {
                    h = 60.0 * (((r - g) / delta) + 4.0);
                }
            }
            if (h < 0.0) h += 360.0;
            h /= 360.0; // Normalize to [0, 1]
            
            // Saturation calculation
            double s = (max_val == 0.0) ? 0.0 : (delta / max_val);
            
            // Value calculation
            double v = max_val;
            
            if (channels_first) {
                result({0, y, x}) = h;
                result({1, y, x}) = s;
                result({2, y, x}) = v;
            } else {
                result({y, x, 0}) = h;
                result({y, x, 1}) = s;
                result({y, x, 2}) = v;
            }
        }
    }
    
    return result;
}

// ===== Normalization and Preprocessing =====

Tensor ImageProcessor::normalize_to_float(const Tensor& image) {
    Tensor result = image / 255.0;
    return clamp_values(result, 0.0, 1.0);
}

Tensor ImageProcessor::normalize_imagenet(const Tensor& image,
                                        const std::vector<double>& mean,
                                        const std::vector<double>& std,
                                        bool channels_first) {
    auto [height, width, channels] = get_image_dims(image, channels_first);
    
    if (mean.size() != channels || std.size() != channels) {
        throw std::invalid_argument("Mean and std vectors must match number of channels");
    }
    
    Tensor result = image;
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                double value;
                if (channels_first) {
                    value = (result({c, y, x}) - mean[c]) / std[c];
                    result({c, y, x}) = value;
                } else {
                    value = (result({y, x, c}) - mean[c]) / std[c];
                    result({y, x, c}) = value;
                }
            }
        }
    }
    
    return result;
}

Tensor ImageProcessor::denormalize_imagenet(const Tensor& normalized_image,
                                          const std::vector<double>& mean,
                                          const std::vector<double>& std,
                                          bool channels_first) {
    auto [height, width, channels] = get_image_dims(normalized_image, channels_first);
    
    if (mean.size() != channels || std.size() != channels) {
        throw std::invalid_argument("Mean and std vectors must match number of channels");
    }
    
    Tensor result = normalized_image;
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                double value;
                if (channels_first) {
                    value = result({c, y, x}) * std[c] + mean[c];
                    result({c, y, x}) = value;
                } else {
                    value = result({y, x, c}) * std[c] + mean[c];
                    result({y, x, c}) = value;
                }
            }
        }
    }
    
    return clamp_values(result, 0.0, 1.0);
}

// ===== Image Filters and Effects =====

Tensor ImageProcessor::create_gaussian_kernel(size_t size, double sigma) {
    if (size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd");
    }
    
    Tensor kernel({size, size});
    int center = static_cast<int>(size) / 2;
    double sum = 0.0;
    
    for (int y = 0; y < static_cast<int>(size); ++y) {
        for (int x = 0; x < static_cast<int>(size); ++x) {
            int dy = y - center;
            int dx = x - center;
            double value = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            kernel({static_cast<size_t>(y), static_cast<size_t>(x)}) = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (size_t i = 0; i < kernel.size(); ++i) {
        kernel.data()[i] /= sum;
    }
    
    return kernel;
}

Tensor ImageProcessor::gaussian_blur(const Tensor& image, size_t kernel_size, double sigma, bool channels_first) {
    Tensor kernel = create_gaussian_kernel(kernel_size, sigma);
    return apply_2d_convolution(image, kernel, channels_first);
}

Tensor ImageProcessor::apply_2d_convolution(const Tensor& image, const Tensor& kernel, bool channels_first) {
    auto [height, width, channels] = get_image_dims(image, channels_first);
    auto kernel_shape = kernel.shape();
    
    if (kernel_shape.size() != 2 || kernel_shape[0] != kernel_shape[1]) {
        throw std::invalid_argument("Kernel must be square 2D tensor");
    }
    
    size_t kernel_size = kernel_shape[0];
    int pad = static_cast<int>(kernel_size) / 2;
    
    std::vector<size_t> output_shape;
    if (channels_first) {
        output_shape = {channels, height, width};
    } else {
        output_shape = {height, width, channels};
    }
    
    Tensor result(output_shape);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                double sum = 0.0;
                
                for (size_t ky = 0; ky < kernel_size; ++ky) {
                    for (size_t kx = 0; kx < kernel_size; ++kx) {
                        int img_y = static_cast<int>(y) + static_cast<int>(ky) - pad;
                        int img_x = static_cast<int>(x) + static_cast<int>(kx) - pad;
                        
                        // Handle boundaries with padding (replicate edge pixels)
                        img_y = std::max(0, std::min(img_y, static_cast<int>(height - 1)));
                        img_x = std::max(0, std::min(img_x, static_cast<int>(width - 1)));
                        
                        double pixel_value;
                        if (channels_first) {
                            pixel_value = image({c, static_cast<size_t>(img_y), static_cast<size_t>(img_x)});
                        } else {
                            pixel_value = image({static_cast<size_t>(img_y), static_cast<size_t>(img_x), c});
                        }
                        
                        sum += pixel_value * kernel({ky, kx});
                    }
                }
                
                if (channels_first) {
                    result({c, y, x}) = sum;
                } else {
                    result({y, x, c}) = sum;
                }
            }
        }
    }
    
    return result;
}

Tensor ImageProcessor::adjust_brightness(const Tensor& image, double factor) {
    return clamp_values(image * factor, 0.0, 1.0);
}

Tensor ImageProcessor::adjust_contrast(const Tensor& image, double factor) {
    // Calculate mean brightness
    double mean = 0.0;
    size_t total_pixels = image.size();
    for (size_t i = 0; i < total_pixels; ++i) {
        mean += image.data()[i];
    }
    mean /= total_pixels;
    
    // Apply contrast adjustment: new_value = mean + factor * (old_value - mean)
    Tensor result = image;
    for (size_t i = 0; i < total_pixels; ++i) {
        result.data()[i] = mean + factor * (image.data()[i] - mean);
    }
    
    return clamp_values(result, 0.0, 1.0);
}

// ===== Utility Functions =====

Tensor ImageProcessor::clamp_values(const Tensor& image, double min_val, double max_val) {
    Tensor result = image;
    for (size_t i = 0; i < result.size(); ++i) {
        result.data()[i] = std::max(min_val, std::min(max_val, result.data()[i]));
    }
    return result;
}

// ===== Data Augmentation =====

Tensor ImageProcessor::random_horizontal_flip(const Tensor& image, double probability, bool channels_first) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    if (dist(gen) > probability) {
        return image; // No flip
    }
    
    auto [height, width, channels] = get_image_dims(image, channels_first);
    
    std::vector<size_t> output_shape;
    if (channels_first) {
        output_shape = {channels, height, width};
    } else {
        output_shape = {height, width, channels};
    }
    
    Tensor result(output_shape);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t flipped_x = width - 1 - x;
                
                if (channels_first) {
                    result({c, y, x}) = image({c, y, flipped_x});
                } else {
                    result({y, x, c}) = image({y, flipped_x, c});
                }
            }
        }
    }
    
    return result;
}

// ===== Batch Operations =====

std::vector<Tensor> ImageProcessor::batch_resize(const std::vector<Tensor>& images,
                                                 size_t new_height, size_t new_width,
                                                 InterpolationMethod method,
                                                 bool channels_first) {
    std::vector<Tensor> result;
    result.reserve(images.size());
    
    for (const auto& image : images) {
        result.push_back(resize(image, new_height, new_width, method, channels_first));
    }
    
    return result;
}

Tensor ImageProcessor::create_batch(const std::vector<Tensor>& images, bool channels_first) {
    if (images.empty()) {
        throw std::invalid_argument("Cannot create batch from empty image vector");
    }
    
    // Verify all images have the same shape
    auto first_shape = images[0].shape();
    for (size_t i = 1; i < images.size(); ++i) {
        if (images[i].shape() != first_shape) {
            throw std::invalid_argument("All images must have the same shape for batching");
        }
    }
    
    // Create batch tensor
    std::vector<size_t> batch_shape = {images.size()};
    batch_shape.insert(batch_shape.end(), first_shape.begin(), first_shape.end());
    
    Tensor batch(batch_shape);
    
    // Copy each image into the batch
    size_t image_size = images[0].size();
    for (size_t i = 0; i < images.size(); ++i) {
        std::copy(images[i].data().begin(), images[i].data().end(),
                 batch.data().begin() + i * image_size);
    }
    
    return batch;
}

} // namespace ai
} // namespace asekioml
