#pragma once

#include "matrix.hpp"
#include <memory>
#include <stdexcept>
#include <functional>
#include <vector>

// Include headers based on available GPU support
#ifdef ASEKIOML_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef ASEKIOML_CUDNN_SUPPORT
#include <cudnn.h>
#endif
#endif

#ifdef ASEKIOML_ROCM_SUPPORT
#include <hip/hip_runtime.h>
#ifdef ASEKIOML_ROCBLAS_SUPPORT
#include <rocblas.h>
#endif
#endif

#ifdef ASEKIOML_OPENCL_SUPPORT
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#endif

namespace asekioml {
namespace gpu {

enum class DeviceType {
    CPU,
    CUDA,      // NVIDIA GPUs
    ROCM,      // AMD GPUs (HIP/ROCm)
    OPENCL,    // Cross-platform (Intel, AMD, NVIDIA)
    METAL,     // Apple GPUs
    VULKAN,    // Cross-platform compute
    AUTO       // Automatically detect best available
};

// GPU device information
struct GPUDeviceInfo {
    DeviceType type;
    int device_id;
    std::string name;
    size_t memory_mb;
    int compute_capability_major;
    int compute_capability_minor;
    bool supports_double_precision;
    bool supports_unified_memory;
};

// Abstract GPU backend interface
class GPUBackend {
public:
    virtual ~GPUBackend() = default;
    virtual DeviceType get_type() const = 0;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copy_to_device(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_to_host(void* dst, const void* src, size_t bytes) = 0;
    virtual void synchronize() = 0;
    virtual void matrix_multiply(const void* a, const void* b, void* c, 
                               size_t m, size_t n, size_t k) = 0;
    virtual void matrix_add(const void* a, const void* b, void* c, size_t size) = 0;
    virtual void relu_inplace(void* data, size_t size) = 0;
    virtual void sigmoid_inplace(void* data, size_t size) = 0;
};

// GPU Manager for device detection and backend creation
class GPUManager {
private:
    static std::unique_ptr<GPUManager> instance_;
    std::vector<GPUDeviceInfo> available_devices_;
    
public:
    static GPUManager& get_instance();
    
    // Device discovery
    void scan_devices();
    const std::vector<GPUDeviceInfo>& get_available_devices() const;
    DeviceType get_best_device_type() const;
    
    // Backend creation
    std::unique_ptr<GPUBackend> create_backend(DeviceType type, int device_id = 0);
    
    // Utility functions
    bool has_cuda_support() const;
    bool has_rocm_support() const;
    bool has_opencl_support() const;
    size_t get_total_gpu_memory(DeviceType type) const;
};

class GPUMatrix {
private:
    DeviceType device_type_;
    void* device_data_;
    size_t rows_;
    size_t cols_;
    bool owns_data_;
    std::unique_ptr<GPUBackend> backend_;
    
public:
    GPUMatrix(size_t rows, size_t cols, DeviceType device = DeviceType::AUTO);
    ~GPUMatrix();
    
    // Move semantics for efficient transfers
    GPUMatrix(GPUMatrix&& other) noexcept;
    GPUMatrix& operator=(GPUMatrix&& other) noexcept;
    
    // Disable copy (expensive GPU memory operations)
    GPUMatrix(const GPUMatrix&) = delete;
    GPUMatrix& operator=(const GPUMatrix&) = delete;
    
    // Host-Device transfers
    void from_host(const Matrix& host_matrix);
    Matrix to_host() const;
    
    // GPU operations
    GPUMatrix multiply(const GPUMatrix& other) const;
    GPUMatrix add(const GPUMatrix& other) const;
    GPUMatrix subtract(const GPUMatrix& other) const;
    
    // GPU-accelerated activation functions
    GPUMatrix relu() const;
    GPUMatrix sigmoid() const;
    GPUMatrix softmax() const;
    
    // Batch operations
    GPUMatrix batch_multiply(const GPUMatrix& other) const;    // Memory management
    static void set_device(int device_id);
    void synchronize();  // Instance method, not static
    static size_t get_free_memory();
    static std::vector<GPUDeviceInfo> get_available_devices();
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    DeviceType device() const { return device_type_; }
    
    // Performance optimization hints
    void pin_memory();
    void enable_async_operations(bool enable = true);
};

// GPU-accelerated neural network layers
class GPUDenseLayer {
private:
    GPUMatrix weights_;
    GPUMatrix biases_;
    GPUMatrix last_input_;
    bool training_;
    DeviceType device_type_;
    
public:
    GPUDenseLayer(size_t input_size, size_t output_size, DeviceType device = DeviceType::AUTO);
    
    GPUMatrix forward(const GPUMatrix& input);
    GPUMatrix backward(const GPUMatrix& gradient);
    
    void update_weights(const GPUMatrix& weight_gradients, const GPUMatrix& bias_gradients, double learning_rate);
    
    // Transfer to/from CPU
    void from_cpu_layer(const class DenseLayer& cpu_layer);
    void to_cpu_layer(class DenseLayer& cpu_layer) const;
    
    DeviceType get_device_type() const { return device_type_; }
};

// Memory-efficient GPU batch processor
class GPUBatchProcessor {
private:
    size_t max_batch_size_;
    DeviceType device_;
    std::vector<GPUMatrix> batch_buffers_;
    
public:
    GPUBatchProcessor(size_t max_batch_size, DeviceType device = DeviceType::AUTO);
    
    // Process batches efficiently without frequent CPU-GPU transfers
    std::vector<GPUMatrix> process_batches(const std::vector<Matrix>& cpu_batches,
                                          std::function<GPUMatrix(const GPUMatrix&)> processor);
};

// High-level GPU acceleration functions
namespace gpu_ops {
    // Matrix operations with automatic device selection
    Matrix multiply_gpu(const Matrix& a, const Matrix& b, DeviceType device = DeviceType::AUTO);
    Matrix add_gpu(const Matrix& a, const Matrix& b, DeviceType device = DeviceType::AUTO);
    
    // Activation functions
    Matrix relu_gpu(const Matrix& input, DeviceType device = DeviceType::AUTO);
    Matrix sigmoid_gpu(const Matrix& input, DeviceType device = DeviceType::AUTO);
    Matrix softmax_gpu(const Matrix& input, DeviceType device = DeviceType::AUTO);
    
    // Convolution operations (for CNNs)
    Matrix conv2d_gpu(const Matrix& input, const Matrix& kernel, 
                     int stride = 1, int padding = 0, DeviceType device = DeviceType::AUTO);
    
    // Batch operations
    std::vector<Matrix> batch_multiply_gpu(const std::vector<Matrix>& inputs,
                                         const Matrix& weights, DeviceType device = DeviceType::AUTO);
}

#ifdef ASEKIOML_CUDA_SUPPORT
// CUDA-specific backend implementation
class CUDABackend : public GPUBackend {
private:
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;
    int device_id_;
    
public:
    CUDABackend(int device_id = 0);
    ~CUDABackend();
    
    DeviceType get_type() const override { return DeviceType::CUDA; }
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const void* src, size_t bytes) override;
    void synchronize() override;
    void matrix_multiply(const void* a, const void* b, void* c, 
                        size_t m, size_t n, size_t k) override;
    void matrix_add(const void* a, const void* b, void* c, size_t size) override;
    void relu_inplace(void* data, size_t size) override;
    void sigmoid_inplace(void* data, size_t size) override;
};
#endif

#ifdef ASEKIOML_ROCM_SUPPORT
// ROCm/HIP backend for AMD GPUs
class ROCmBackend : public GPUBackend {
private:
    hipStream_t stream_;
    int device_id_;
#ifdef ASEKIOML_ROCBLAS_SUPPORT
    rocblas_handle rocblas_handle_;
#endif
    
public:
    ROCmBackend(int device_id = 0);
    ~ROCmBackend();
    
    DeviceType get_type() const override { return DeviceType::ROCM; }
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const void* src, size_t bytes) override;
    void synchronize() override;
    void matrix_multiply(const void* a, const void* b, void* c, 
                        size_t m, size_t n, size_t k) override;
    void matrix_add(const void* a, const void* b, void* c, size_t size) override;
    void relu_inplace(void* data, size_t size) override;
    void sigmoid_inplace(void* data, size_t size) override;
};
#endif

#ifdef ASEKIOML_OPENCL_SUPPORT
// OpenCL backend for cross-platform support
class OpenCLBackend : public GPUBackend {
private:
    cl::Context context_;
    cl::Device device_;
    cl::CommandQueue queue_;
    cl::Program program_;
    std::map<std::string, cl::Kernel> kernels_;
    
public:
    OpenCLBackend(int device_id = 0);
    ~OpenCLBackend();
    
    DeviceType get_type() const override { return DeviceType::OPENCL; }
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    void copy_to_device(void* dst, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const void* src, size_t bytes) override;
    void synchronize() override;
    void matrix_multiply(const void* a, const void* b, void* c, 
                        size_t m, size_t n, size_t k) override;
    void matrix_add(const void* a, const void* b, void* c, size_t size) override;
    void relu_inplace(void* data, size_t size) override;
    void sigmoid_inplace(void* data, size_t size) override;

private:
    void build_kernels();
    cl::Kernel& get_kernel(const std::string& name);
};
#endif

} // namespace gpu
} // namespace asekioml
