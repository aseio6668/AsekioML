# CLModel Multi-Vendor GPU Support

## Overview

CLModel provides comprehensive GPU acceleration support across multiple vendors, ensuring that users can leverage their available hardware regardless of the manufacturer. This document explains the multi-vendor GPU architecture and how to use it effectively.

## Supported GPU Vendors

### NVIDIA GPUs (CUDA)
- **Support Level**: Full native support
- **Requirements**: CUDA Toolkit 11.0+
- **Libraries**: CUDA Runtime, cuBLAS, cuDNN (optional)
- **Features**:
  - High-performance matrix operations via cuBLAS
  - Deep learning optimizations via cuDNN
  - Memory pooling and unified memory
  - Asynchronous operations and streams
  - Support for all modern NVIDIA GPUs (GTX 10 series and newer)

### AMD GPUs (ROCm/HIP)
- **Support Level**: Full native support
- **Requirements**: ROCm 4.0+
- **Libraries**: HIP Runtime, rocBLAS
- **Features**:
  - Native AMD GPU acceleration via HIP
  - Optimized BLAS operations via rocBLAS
  - Memory management and async operations
  - Support for modern AMD GPUs (RX 5000 series, RX 6000 series, RDNA/RDNA2)
  - Compatibility with AMD Instinct accelerators

### Intel GPUs (OpenCL)
- **Support Level**: Cross-platform support
- **Requirements**: OpenCL 2.0+
- **Libraries**: OpenCL drivers
- **Features**:
  - Cross-platform GPU compute via OpenCL
  - Support for Intel integrated and discrete GPUs
  - Custom kernel compilation and execution
  - Compatible with Intel Arc GPUs and integrated graphics

### Other Vendors (OpenCL Fallback)
- **Support Level**: Best-effort compatibility
- **Coverage**: Any GPU with OpenCL support
- **Features**:
  - Automatic fallback for unsupported vendors
  - Basic matrix operations and activations
  - Memory management via OpenCL buffers

## Architecture Overview

### Device Detection and Selection

CLModel automatically detects available GPU devices and selects the best option:

```cpp
// Automatic device selection (recommended)
GPUMatrix matrix(1024, 1024, DeviceType::AUTO);

// Manual device selection
GPUMatrix cuda_matrix(1024, 1024, DeviceType::CUDA);
GPUMatrix rocm_matrix(1024, 1024, DeviceType::ROCM);
GPUMatrix opencl_matrix(1024, 1024, DeviceType::OPENCL);
```

### Device Priority

The framework prioritizes devices in the following order:
1. **NVIDIA CUDA** - Best performance for most workloads
2. **AMD ROCm** - Native AMD GPU performance
3. **OpenCL** - Cross-platform compatibility
4. **CPU** - Fallback for systems without GPU support

### Backend Abstraction

CLModel uses a unified backend interface that abstracts vendor-specific details:

```cpp
class GPUBackend {
public:
    virtual DeviceType get_type() const = 0;
    virtual void* allocate(size_t bytes) = 0;
    virtual void matrix_multiply(const void* a, const void* b, void* c, 
                               size_t m, size_t n, size_t k) = 0;
    // ... other operations
};
```

## Installation and Setup

### NVIDIA CUDA Setup

1. **Install CUDA Toolkit**:
   ```bash
   # Download from https://developer.nvidia.com/cuda-downloads
   # Windows: Run the installer
   # Linux: Follow distribution-specific instructions
   ```

2. **Verify Installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **CMake Configuration**:
   ```bash
   cmake -DCMAKE_CUDA_COMPILER=nvcc ..
   ```

### AMD ROCm Setup

1. **Install ROCm**:
   ```bash
   # Ubuntu/Debian
   wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dev rocblas

   # Windows
   # Download ROCm for Windows from AMD website
   ```

2. **Verify Installation**:
   ```bash
   rocm-smi
   hipcc --version
   ```

3. **CMake Configuration**:
   ```bash
   cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
   ```

### OpenCL Setup

1. **Install OpenCL**:
   ```bash
   # Ubuntu/Debian
   sudo apt install opencl-headers ocl-icd-opencl-dev

   # Windows
   # Usually included with GPU drivers
   ```

2. **Verify Installation**:
   ```bash
   clinfo  # List available OpenCL devices
   ```

## Usage Examples

### Basic GPU Operations

```cpp
#include "clmodel.hpp"
using namespace clmodel::gpu;

// Matrix operations with automatic device selection
Matrix a(1024, 1024);
Matrix b(1024, 1024);
// ... initialize matrices ...

// GPU-accelerated multiplication
Matrix result = gpu_ops::multiply_gpu(a, b);

// Device-specific operations
Matrix cuda_result = gpu_ops::multiply_gpu(a, b, DeviceType::CUDA);
Matrix rocm_result = gpu_ops::multiply_gpu(a, b, DeviceType::ROCM);
```

### Neural Network Training

```cpp
// Create GPU-accelerated network layers
GPUDenseLayer layer1(784, 128, DeviceType::AUTO);
GPUDenseLayer layer2(128, 10, DeviceType::AUTO);

// Training loop with GPU acceleration
for (const auto& batch : training_data) {
    GPUMatrix gpu_input(batch.rows(), batch.cols());
    gpu_input.from_host(batch);
    
    // Forward pass
    GPUMatrix hidden = layer1.forward(gpu_input);
    GPUMatrix output = layer2.forward(hidden);
    
    // ... backward pass and optimization
}
```

### Device Information and Monitoring

```cpp
// Get available devices
auto devices = GPUMatrix::get_available_devices();
for (const auto& device : devices) {
    std::cout << "Device: " << device.name 
              << ", Memory: " << device.memory_mb << " MB"
              << ", Type: " << static_cast<int>(device.type) << std::endl;
}

// Check specific vendor support
auto& manager = GPUManager::get_instance();
if (manager.has_cuda_support()) {
    std::cout << "NVIDIA CUDA available" << std::endl;
}
if (manager.has_rocm_support()) {
    std::cout << "AMD ROCm available" << std::endl;
}
```

## Performance Considerations

### Memory Management

- **Unified Memory**: NVIDIA GPUs support unified memory for easier programming
- **Memory Pooling**: CLModel includes memory pool optimization for reduced allocation overhead
- **Async Transfers**: Overlap CPU-GPU transfers with computation

### Batch Processing

- Use `GPUBatchProcessor` for efficient batch operations
- Minimize CPU-GPU transfers by keeping data on GPU between operations
- Leverage async operations where possible

### Vendor-Specific Optimizations

#### NVIDIA CUDA
- Utilize Tensor Cores on RTX/V100+ GPUs for mixed-precision training
- Enable cuDNN for optimized convolution operations
- Use CUDA streams for concurrent kernel execution

#### AMD ROCm
- Leverage rocBLAS for optimized linear algebra operations
- Use HIP streams for asynchronous operations
- Optimize for RDNA/RDNA2 architecture characteristics

#### OpenCL
- Write efficient kernels for specific operations
- Optimize work group sizes for target hardware
- Use local memory effectively for cache optimization

## Troubleshooting

### Common Issues

1. **Device Not Detected**:
   - Verify driver installation
   - Check library paths (LD_LIBRARY_PATH on Linux)
   - Ensure user has permissions for GPU access

2. **Compilation Errors**:
   - Verify SDK/toolkit installation
   - Check CMake configuration
   - Ensure compatible compiler versions

3. **Runtime Errors**:
   - Check GPU memory availability
   - Verify device compatibility
   - Enable debug output for detailed error messages

### Performance Issues

1. **Slow GPU Operations**:
   - Profile memory transfer overhead
   - Check for memory fragmentation
   - Verify optimal batch sizes

2. **AMD GPU Performance**:
   - Ensure ROCm is properly configured
   - Check for latest driver updates
   - Verify rocBLAS installation

### Debugging

Enable verbose logging:
```cpp
// Set environment variable
export CLMODEL_GPU_DEBUG=1

// Or in code
GPUManager::get_instance().set_debug_mode(true);
```

## Future Enhancements

### Planned Features

1. **Apple Metal Support**: Native acceleration for Apple Silicon Macs
2. **Vulkan Compute**: Cross-platform compute via Vulkan
3. **Intel oneAPI**: Native Intel GPU support via DPC++
4. **Multi-GPU Support**: Automatic load balancing across multiple GPUs
5. **Mixed Precision Training**: FP16/BF16 support for faster training

### Contribution Guidelines

To add support for new GPU vendors:

1. Implement the `GPUBackend` interface
2. Add device detection in `GPUManager::scan_devices()`
3. Update CMake configuration for new dependencies
4. Add comprehensive tests for the new backend
5. Update documentation and examples

## Conclusion

CLModel's multi-vendor GPU support ensures that users can leverage their available hardware effectively, regardless of the GPU vendor. The unified API abstracts away vendor-specific details while still providing access to platform-specific optimizations when needed.

For AMD GPU users specifically, the framework provides first-class support through ROCm/HIP, ensuring that AMD hardware can achieve competitive performance for machine learning workloads.
