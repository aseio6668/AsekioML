#include "gpu_acceleration.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace clmodel {
namespace gpu {

// GPUManager singleton implementation
std::unique_ptr<GPUManager> GPUManager::instance_ = nullptr;

GPUManager& GPUManager::get_instance() {
    if (!instance_) {
        instance_ = std::make_unique<GPUManager>();
        instance_->scan_devices();
    }
    return *instance_;
}

void GPUManager::scan_devices() {
    available_devices_.clear();
    
    // Add CPU as fallback
    GPUDeviceInfo cpu_info;
    cpu_info.type = DeviceType::CPU;
    cpu_info.device_id = 0;
    cpu_info.name = "CPU";
    cpu_info.memory_mb = 0; // Use system RAM
    cpu_info.compute_capability_major = 0;
    cpu_info.compute_capability_minor = 0;
    cpu_info.supports_double_precision = true;
    cpu_info.supports_unified_memory = true;
    available_devices_.push_back(cpu_info);
    
#ifdef CLMODEL_CUDA_SUPPORT
    // Scan CUDA devices
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                GPUDeviceInfo info;
                info.type = DeviceType::CUDA;
                info.device_id = i;
                info.name = prop.name;
                info.memory_mb = prop.totalGlobalMem / (1024 * 1024);
                info.compute_capability_major = prop.major;
                info.compute_capability_minor = prop.minor;
                info.supports_double_precision = prop.major >= 2 || (prop.major == 1 && prop.minor >= 3);
                info.supports_unified_memory = prop.managedMemory;
                available_devices_.push_back(info);
            }
        }
    }
#endif

#ifdef CLMODEL_ROCM_SUPPORT
    // Scan ROCm devices
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0) {
        for (int i = 0; i < device_count; ++i) {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
                GPUDeviceInfo info;
                info.type = DeviceType::ROCM;
                info.device_id = i;
                info.name = prop.name;
                info.memory_mb = prop.totalGlobalMem / (1024 * 1024);
                info.compute_capability_major = prop.major;
                info.compute_capability_minor = prop.minor;
                info.supports_double_precision = true; // Most modern AMD GPUs support FP64
                info.supports_unified_memory = prop.managedMemory;
                available_devices_.push_back(info);
            }
        }
    }
#endif

#ifdef CLMODEL_OPENCL_SUPPORT
    // Scan OpenCL devices
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        for (const auto& platform : platforms) {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            
            for (size_t i = 0; i < devices.size(); ++i) {
                GPUDeviceInfo info;
                info.type = DeviceType::OPENCL;
                info.device_id = static_cast<int>(i);
                info.name = devices[i].getInfo<CL_DEVICE_NAME>();
                info.memory_mb = devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024);
                info.compute_capability_major = 1;
                info.compute_capability_minor = 0;
                info.supports_double_precision = devices[i].getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>() != 0;
                info.supports_unified_memory = devices[i].getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
                available_devices_.push_back(info);
            }
        }
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error during device scan: " << e.what() << std::endl;
    }
#endif
}

const std::vector<GPUDeviceInfo>& GPUManager::get_available_devices() const {
    return available_devices_;
}

DeviceType GPUManager::get_best_device_type() const {
    // Priority: CUDA > ROCm > OpenCL > CPU
    for (const auto& device : available_devices_) {
        if (device.type == DeviceType::CUDA) return DeviceType::CUDA;
    }
    for (const auto& device : available_devices_) {
        if (device.type == DeviceType::ROCM) return DeviceType::ROCM;
    }
    for (const auto& device : available_devices_) {
        if (device.type == DeviceType::OPENCL) return DeviceType::OPENCL;
    }
    return DeviceType::CPU;
}

std::unique_ptr<GPUBackend> GPUManager::create_backend(DeviceType type, int device_id) {
    if (type == DeviceType::AUTO) {
        type = get_best_device_type();
    }
    
    switch (type) {
#ifdef CLMODEL_CUDA_SUPPORT
        case DeviceType::CUDA:
            return std::make_unique<CUDABackend>(device_id);
#endif
#ifdef CLMODEL_ROCM_SUPPORT
        case DeviceType::ROCM:
            return std::make_unique<ROCmBackend>(device_id);
#endif
#ifdef CLMODEL_OPENCL_SUPPORT
        case DeviceType::OPENCL:
            return std::make_unique<OpenCLBackend>(device_id);
#endif
        default:
            throw std::runtime_error("Requested GPU backend not available");
    }
}

bool GPUManager::has_cuda_support() const {
#ifdef CLMODEL_CUDA_SUPPORT
    return std::any_of(available_devices_.begin(), available_devices_.end(),
                      [](const GPUDeviceInfo& device) { return device.type == DeviceType::CUDA; });
#else
    return false;
#endif
}

bool GPUManager::has_rocm_support() const {
#ifdef CLMODEL_ROCM_SUPPORT
    return std::any_of(available_devices_.begin(), available_devices_.end(),
                      [](const GPUDeviceInfo& device) { return device.type == DeviceType::ROCM; });
#else
    return false;
#endif
}

bool GPUManager::has_opencl_support() const {
#ifdef CLMODEL_OPENCL_SUPPORT
    return std::any_of(available_devices_.begin(), available_devices_.end(),
                      [](const GPUDeviceInfo& device) { return device.type == DeviceType::OPENCL; });
#else
    return false;
#endif
}

size_t GPUManager::get_total_gpu_memory(DeviceType type) const {
    size_t total = 0;
    for (const auto& device : available_devices_) {
        if (device.type == type) {
            total += device.memory_mb;
        }
    }
    return total;
}

// GPUMatrix implementation
GPUMatrix::GPUMatrix(size_t rows, size_t cols, DeviceType device)
    : device_type_(device), device_data_(nullptr), rows_(rows), cols_(cols), owns_data_(true) {
    
    if (device_type_ == DeviceType::AUTO) {
        device_type_ = GPUManager::get_instance().get_best_device_type();
    }
    
    backend_ = GPUManager::get_instance().create_backend(device_type_);
    device_data_ = backend_->allocate(rows * cols * sizeof(double));
}

GPUMatrix::~GPUMatrix() {
    if (owns_data_ && device_data_ && backend_) {
        backend_->deallocate(device_data_);
    }
}

GPUMatrix::GPUMatrix(GPUMatrix&& other) noexcept
    : device_type_(other.device_type_), device_data_(other.device_data_),
      rows_(other.rows_), cols_(other.cols_), owns_data_(other.owns_data_),
      backend_(std::move(other.backend_)) {
    other.device_data_ = nullptr;
    other.owns_data_ = false;
}

GPUMatrix& GPUMatrix::operator=(GPUMatrix&& other) noexcept {
    if (this != &other) {
        if (owns_data_ && device_data_ && backend_) {
            backend_->deallocate(device_data_);
        }
        
        device_type_ = other.device_type_;
        device_data_ = other.device_data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        owns_data_ = other.owns_data_;
        backend_ = std::move(other.backend_);
        
        other.device_data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

void GPUMatrix::from_host(const Matrix& host_matrix) {
    if (host_matrix.rows() != rows_ || host_matrix.cols() != cols_) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    
    // Create a contiguous copy of the host data
    std::vector<double> host_data;
    host_data.reserve(rows_ * cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            host_data.push_back(host_matrix(i, j));
        }
    }
    
    backend_->copy_to_device(device_data_, host_data.data(), rows_ * cols_ * sizeof(double));
}

Matrix GPUMatrix::to_host() const {
    std::vector<double> host_data(rows_ * cols_);
    backend_->copy_to_host(host_data.data(), device_data_, rows_ * cols_ * sizeof(double));
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) = host_data[i * cols_ + j];
        }
    }
    
    return result;
}

GPUMatrix GPUMatrix::multiply(const GPUMatrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    GPUMatrix result(rows_, other.cols_, device_type_);
    backend_->matrix_multiply(device_data_, other.device_data_, result.device_data_,
                             rows_, other.cols_, cols_);
    return result;
}

GPUMatrix GPUMatrix::add(const GPUMatrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    GPUMatrix result(rows_, cols_, device_type_);
    backend_->matrix_add(device_data_, other.device_data_, result.device_data_, rows_ * cols_);
    return result;
}

GPUMatrix GPUMatrix::subtract(const GPUMatrix& other) const {
    // Implement subtraction using add with negative values
    GPUMatrix neg_other(other.rows_, other.cols_, device_type_);
    // TODO: Implement negation kernel
    return add(neg_other);
}

GPUMatrix GPUMatrix::relu() const {
    GPUMatrix result(rows_, cols_, device_type_);
    backend_->copy_to_device(result.device_data_, device_data_, rows_ * cols_ * sizeof(double));
    backend_->relu_inplace(result.device_data_, rows_ * cols_);
    return result;
}

GPUMatrix GPUMatrix::sigmoid() const {
    GPUMatrix result(rows_, cols_, device_type_);
    backend_->copy_to_device(result.device_data_, device_data_, rows_ * cols_ * sizeof(double));
    backend_->sigmoid_inplace(result.device_data_, rows_ * cols_);
    return result;
}

GPUMatrix GPUMatrix::softmax() const {
    // TODO: Implement proper softmax
    return sigmoid(); // Placeholder
}

void GPUMatrix::synchronize() {
    if (backend_) {
        backend_->synchronize();
    }
}

std::vector<GPUDeviceInfo> GPUMatrix::get_available_devices() {
    return GPUManager::get_instance().get_available_devices();
}

void GPUMatrix::pin_memory() {
    // TODO: Implement memory pinning for faster transfers
}

void GPUMatrix::enable_async_operations(bool enable) {
    (void)enable;  // Suppress unused parameter warning
    // TODO: Implement async operation control
}

#ifdef CLMODEL_CUDA_SUPPORT
// CUDA Backend Implementation
CUDABackend::CUDABackend(int device_id) : device_id_(device_id) {
    cudaSetDevice(device_id_);
    cublasCreate(&cublas_handle_);
    cudaStreamCreate(&stream_);
    cublasSetStream(cublas_handle_, stream_);
}

CUDABackend::~CUDABackend() {
    cublasDestroy(cublas_handle_);
    cudaStreamDestroy(stream_);
}

void* CUDABackend::allocate(size_t bytes) {
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, bytes);
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(error)));
    }
    return ptr;
}

void CUDABackend::deallocate(void* ptr) {
    cudaFree(ptr);
}

void CUDABackend::copy_to_device(void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_);
}

void CUDABackend::copy_to_host(void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

void CUDABackend::synchronize() {
    cudaStreamSynchronize(stream_);
}

void CUDABackend::matrix_multiply(const void* a, const void* b, void* c, size_t m, size_t n, size_t k) {
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                &alpha,
                (const double*)b, static_cast<int>(n),
                (const double*)a, static_cast<int>(k),
                &beta,
                (double*)c, static_cast<int>(n));
}

void CUDABackend::matrix_add(const void* a, const void* b, void* c, size_t size) {
    // Simple CUDA kernel for element-wise addition
    // TODO: Implement actual CUDA kernel
    copy_to_device(c, a, size * sizeof(double));
    cublasDaxpy(cublas_handle_, static_cast<int>(size), &(const double&)1.0, (const double*)b, 1, (double*)c, 1);
}

void CUDABackend::relu_inplace(void* data, size_t size) {
    (void)data;   // Suppress unused parameter warning
    (void)size;   // Suppress unused parameter warning
    // TODO: Implement CUDA ReLU kernel
}

void CUDABackend::sigmoid_inplace(void* data, size_t size) {
    (void)data;
    (void)size;
    // TODO: Implement CUDA sigmoid kernel
}
#endif

#ifdef CLMODEL_ROCM_SUPPORT
// ROCm Backend Implementation
ROCmBackend::ROCmBackend(int device_id) : device_id_(device_id) {
    hipSetDevice(device_id_);
    hipStreamCreate(&stream_);
#ifdef CLMODEL_ROCBLAS_SUPPORT
    rocblas_create_handle(&rocblas_handle_);
    rocblas_set_stream(rocblas_handle_, stream_);
#endif
}

ROCmBackend::~ROCmBackend() {
#ifdef CLMODEL_ROCBLAS_SUPPORT
    rocblas_destroy_handle(rocblas_handle_);
#endif
    hipStreamDestroy(stream_);
}

void* ROCmBackend::allocate(size_t bytes) {
    void* ptr;
    hipError_t error = hipMalloc(&ptr, bytes);
    if (error != hipSuccess) {
        throw std::runtime_error("HIP memory allocation failed");
    }
    return ptr;
}

void ROCmBackend::deallocate(void* ptr) {
    hipFree(ptr);
}

void ROCmBackend::copy_to_device(void* dst, const void* src, size_t bytes) {
    hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream_);
}

void ROCmBackend::copy_to_host(void* dst, const void* src, size_t bytes) {
    hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream_);
    hipStreamSynchronize(stream_);
}

void ROCmBackend::synchronize() {
    hipStreamSynchronize(stream_);
}

void ROCmBackend::matrix_multiply(const void* a, const void* b, void* c, size_t m, size_t n, size_t k) {
#ifdef CLMODEL_ROCBLAS_SUPPORT
    const double alpha = 1.0, beta = 0.0;
    rocblas_dgemm(rocblas_handle_, rocblas_operation_none, rocblas_operation_none,
                  static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                  &alpha,
                  (const double*)b, static_cast<int>(n),
                  (const double*)a, static_cast<int>(k),
                  &beta,
                  (double*)c, static_cast<int>(n));
#else
    // Fallback implementation without rocBLAS
    throw std::runtime_error("Matrix multiplication requires rocBLAS");
#endif
}

void ROCmBackend::matrix_add(const void* a, const void* b, void* c, size_t size) {
#ifdef CLMODEL_ROCBLAS_SUPPORT
    copy_to_device(c, a, size * sizeof(double));
    rocblas_daxpy(rocblas_handle_, static_cast<int>(size), &(const double&)1.0, (const double*)b, 1, (double*)c, 1);
#else
    throw std::runtime_error("Matrix addition requires rocBLAS");
#endif
}

void ROCmBackend::relu_inplace(void* data, size_t size) {
    // TODO: Implement HIP ReLU kernel
}

void ROCmBackend::sigmoid_inplace(void* data, size_t size) {
    // TODO: Implement HIP sigmoid kernel
}
#endif

#ifdef CLMODEL_OPENCL_SUPPORT
// OpenCL Backend Implementation
OpenCLBackend::OpenCLBackend(int device_id) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    
    if (devices.empty() || device_id >= static_cast<int>(devices.size())) {
        throw std::runtime_error("No suitable OpenCL GPU devices found");
    }
    
    device_ = devices[device_id];
    context_ = cl::Context({device_});
    queue_ = cl::CommandQueue(context_, device_);
    
    build_kernels();
}

OpenCLBackend::~OpenCLBackend() {
    // OpenCL objects are automatically cleaned up
}

void* OpenCLBackend::allocate(size_t bytes) {
    cl::Buffer* buffer = new cl::Buffer(context_, CL_MEM_READ_WRITE, bytes);
    return static_cast<void*>(buffer);
}

void OpenCLBackend::deallocate(void* ptr) {
    delete static_cast<cl::Buffer*>(ptr);
}

void OpenCLBackend::copy_to_device(void* dst, const void* src, size_t bytes) {
    cl::Buffer* buffer = static_cast<cl::Buffer*>(dst);
    queue_.enqueueWriteBuffer(*buffer, CL_FALSE, 0, bytes, src);
}

void OpenCLBackend::copy_to_host(void* dst, const void* src, size_t bytes) {
    const cl::Buffer* buffer = static_cast<const cl::Buffer*>(src);
    queue_.enqueueReadBuffer(*buffer, CL_TRUE, 0, bytes, dst);
}

void OpenCLBackend::synchronize() {
    queue_.finish();
}

void OpenCLBackend::matrix_multiply(const void* a, const void* b, void* c, size_t m, size_t n, size_t k) {
    // TODO: Implement OpenCL matrix multiplication kernel
    throw std::runtime_error("OpenCL matrix multiplication not yet implemented");
}

void OpenCLBackend::matrix_add(const void* a, const void* b, void* c, size_t size) {
    // TODO: Implement OpenCL matrix addition kernel
    throw std::runtime_error("OpenCL matrix addition not yet implemented");
}

void OpenCLBackend::relu_inplace(void* data, size_t size) {
    // TODO: Implement OpenCL ReLU kernel
}

void OpenCLBackend::sigmoid_inplace(void* data, size_t size) {
    // TODO: Implement OpenCL sigmoid kernel
}

void OpenCLBackend::build_kernels() {
    // TODO: Build OpenCL kernels for various operations
    const std::string kernel_source = R"(
        __kernel void matrix_add(__global double* a, __global double* b, __global double* c, int size) {
            int id = get_global_id(0);
            if (id < size) {
                c[id] = a[id] + b[id];
            }
        }
        
        __kernel void relu(__global double* data, int size) {
            int id = get_global_id(0);
            if (id < size) {
                data[id] = fmax(0.0, data[id]);
            }
        }
    )";
    
    try {
        program_ = cl::Program(context_, kernel_source);
        program_.build({device_});
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL kernel build error: " << e.what() << std::endl;
        std::cerr << "Build log: " << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_) << std::endl;
    }
}

cl::Kernel& OpenCLBackend::get_kernel(const std::string& name) {
    auto it = kernels_.find(name);
    if (it == kernels_.end()) {
        kernels_[name] = cl::Kernel(program_, name.c_str());
    }
    return kernels_[name];
}
#endif

// High-level GPU operations
namespace gpu_ops {

Matrix multiply_gpu(const Matrix& a, const Matrix& b, DeviceType device) {
    GPUMatrix gpu_a(a.rows(), a.cols(), device);
    GPUMatrix gpu_b(b.rows(), b.cols(), device);
    
    gpu_a.from_host(a);
    gpu_b.from_host(b);
    
    GPUMatrix result = gpu_a.multiply(gpu_b);
    return result.to_host();
}

Matrix add_gpu(const Matrix& a, const Matrix& b, DeviceType device) {
    GPUMatrix gpu_a(a.rows(), a.cols(), device);
    GPUMatrix gpu_b(b.rows(), b.cols(), device);
    
    gpu_a.from_host(a);
    gpu_b.from_host(b);
    
    GPUMatrix result = gpu_a.add(gpu_b);
    return result.to_host();
}

Matrix relu_gpu(const Matrix& input, DeviceType device) {
    GPUMatrix gpu_input(input.rows(), input.cols(), device);
    gpu_input.from_host(input);
    
    GPUMatrix result = gpu_input.relu();
    return result.to_host();
}

Matrix sigmoid_gpu(const Matrix& input, DeviceType device) {
    GPUMatrix gpu_input(input.rows(), input.cols(), device);
    gpu_input.from_host(input);
    
    GPUMatrix result = gpu_input.sigmoid();
    return result.to_host();
}

Matrix softmax_gpu(const Matrix& input, DeviceType device) {
    GPUMatrix gpu_input(input.rows(), input.cols(), device);
    gpu_input.from_host(input);
    
    GPUMatrix result = gpu_input.softmax();
    return result.to_host();
}

Matrix conv2d_gpu(const Matrix& input, const Matrix& kernel, int stride, int padding, DeviceType device) {
    (void)input;
    (void)kernel;
    (void)stride;
    (void)padding;
    (void)device;
    // TODO: Implement GPU convolution
    throw std::runtime_error("GPU convolution not yet implemented");
}

std::vector<Matrix> batch_multiply_gpu(const std::vector<Matrix>& inputs, const Matrix& weights, DeviceType device) {
    std::vector<Matrix> results;
    results.reserve(inputs.size());
    
    GPUMatrix gpu_weights(weights.rows(), weights.cols(), device);
    gpu_weights.from_host(weights);
    
    for (const auto& input : inputs) {
        GPUMatrix gpu_input(input.rows(), input.cols(), device);
        gpu_input.from_host(input);
        
        GPUMatrix result = gpu_input.multiply(gpu_weights);
        results.push_back(result.to_host());
    }
    
    return results;
}

} // namespace gpu_ops

} // namespace gpu
} // namespace clmodel
