#pragma once

#include <vector>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <memory>

namespace asekioml {
namespace ai {

/**
 * @brief Thread pool for CPU parallelization
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    
    size_t size() const { return workers_.size(); }
    void wait_for_all();
};

/**
 * @brief Advanced parallel compute engine for tensor operations
 * 
 * Provides CPU vectorization, multi-threading, and GPU acceleration
 * for common tensor operations used in neural networks.
 */
class AIComputeEngine {
public:
    enum class Device {
        CPU,
        CUDA,
        OPENCL,
        ROCm
    };
    
    struct ComputeStats {
        size_t cpu_operations = 0;
        size_t gpu_operations = 0;
        double cpu_time = 0.0;
        double gpu_time = 0.0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
    };

private:
    std::unique_ptr<ThreadPool> thread_pool_;
    ComputeStats stats_;
    std::mutex stats_mutex_;
    
    // Configuration
    size_t num_threads_;
    bool use_simd_ = true;
    bool use_gpu_ = false;
    Device preferred_device_ = Device::CPU;
    size_t gpu_threshold_ = 1024 * 1024; // 1M elements
    
    // GPU context management
    bool cuda_initialized_ = false;
    bool opencl_initialized_ = false;
    bool rocm_initialized_ = false;
    
public:
    static AIComputeEngine& instance();
    
    // Configuration
    void set_num_threads(size_t num_threads);
    void enable_simd(bool enable) { use_simd_ = enable; }
    void enable_gpu(bool enable) { use_gpu_ = enable; }
    void set_preferred_device(Device device) { preferred_device_ = device; }
    void set_gpu_threshold(size_t threshold) { gpu_threshold_ = threshold; }
    
    // Device management
    bool initialize_cuda();
    bool initialize_opencl();
    bool initialize_rocm();
    void cleanup_gpu_contexts();
    
    // Parallel operations
    template<typename T>
    void parallel_for(size_t start, size_t end, std::function<void(size_t)> func);
    
    template<typename T>
    void parallel_transform(const T* input, T* output, size_t size, 
                          std::function<T(T)> func);
    
    // Vector operations (SIMD optimized)
    void vector_add(const double* a, const double* b, double* result, size_t size);
    void vector_sub(const double* a, const double* b, double* result, size_t size);
    void vector_mul(const double* a, const double* b, double* result, size_t size);
    void vector_div(const double* a, const double* b, double* result, size_t size);
    void vector_scale(const double* a, double scalar, double* result, size_t size);
    
    // Reduction operations
    double vector_sum(const double* data, size_t size);
    double vector_max(const double* data, size_t size);
    double vector_min(const double* data, size_t size);
    double vector_dot(const double* a, const double* b, size_t size);
    
    // Matrix operations
    void matrix_multiply(const double* a, const double* b, double* c,
                        size_t m, size_t n, size_t k);
    void matrix_transpose(const double* input, double* output,
                         size_t rows, size_t cols);
    
    // Convolution operations
    void convolution_2d(const double* input, const double* kernel, double* output,
                       size_t batch_size, size_t in_channels, size_t out_channels,
                       size_t in_height, size_t in_width,
                       size_t kernel_size, size_t stride, size_t padding);
    
    void im2col(const double* input, double* output,
               size_t channels, size_t height, size_t width,
               size_t kernel_size, size_t stride, size_t padding,
               size_t out_height, size_t out_width);
    
    // Pooling operations
    void max_pool_2d(const double* input, double* output, size_t* indices,
                    size_t batch_size, size_t channels,
                    size_t in_height, size_t in_width,
                    size_t kernel_size, size_t stride, size_t padding);
    
    void avg_pool_2d(const double* input, double* output,
                    size_t batch_size, size_t channels,
                    size_t in_height, size_t in_width,
                    size_t kernel_size, size_t stride, size_t padding);
    
    // Activation functions
    void relu(const double* input, double* output, size_t size);
    void relu_derivative(const double* input, double* output, size_t size);
    void sigmoid(const double* input, double* output, size_t size);
    void tanh_activation(const double* input, double* output, size_t size);
    void softmax(const double* input, double* output, size_t size);
    
    // Statistics and profiling
    const ComputeStats& get_stats() const { return stats_; }
    void reset_stats();
    std::string get_performance_report() const;
    
    // Device selection
    Device select_optimal_device(size_t operation_size) const;
    bool is_gpu_available(Device device) const;
    
private:
    AIComputeEngine();
    ~AIComputeEngine();
    
    // Disable copy/move
    AIComputeEngine(const AIComputeEngine&) = delete;
    AIComputeEngine& operator=(const AIComputeEngine&) = delete;
    
    // SIMD implementations
    void vector_add_simd(const double* a, const double* b, double* result, size_t size);
    void vector_mul_simd(const double* a, const double* b, double* result, size_t size);
    
    // GPU implementations
    void vector_add_cuda(const double* a, const double* b, double* result, size_t size);
    void matrix_multiply_cuda(const double* a, const double* b, double* c,
                             size_t m, size_t n, size_t k);
    
    // Threading utilities
    size_t get_optimal_chunk_size(size_t total_size) const;
    void update_stats(bool is_gpu_operation, double elapsed_time);
};

/**
 * @brief GPU kernel manager for custom compute kernels
 */
class GPUKernelManager {
public:
    struct KernelInfo {
        std::string name;
        std::string source_code;
        size_t local_work_size[3];
        size_t preferred_multiple;
    };
    
private:
    std::unordered_map<std::string, KernelInfo> kernels_;
    bool cuda_available_ = false;
    bool opencl_available_ = false;
    
public:
    static GPUKernelManager& instance();
    
    // Kernel management
    bool load_kernel(const std::string& name, const std::string& source_file);
    bool compile_kernel(const std::string& name);
    bool execute_kernel(const std::string& name, const std::vector<void*>& args,
                       size_t global_work_size[3], size_t local_work_size[3] = nullptr);
    
    // Built-in kernels
    void initialize_builtin_kernels();
    
    // Utility
    std::vector<std::string> get_available_kernels() const;
    bool is_kernel_available(const std::string& name) const;
};

/**
 * @brief Performance profiler for compute operations
 */
class ComputeProfiler {
public:
    struct ProfileEntry {
        std::string operation_name;
        double elapsed_time;
        size_t data_size;
        AIComputeEngine::Device device;
        std::chrono::steady_clock::time_point timestamp;
    };
    
private:
    std::vector<ProfileEntry> profile_data_;
    mutable std::mutex profile_mutex_;
    bool enabled_ = false;
    
public:
    static ComputeProfiler& instance();
    
    void enable(bool enable) { enabled_ = enable; }
    bool is_enabled() const { return enabled_; }
    
    void record_operation(const std::string& name, double elapsed_time, 
                         size_t data_size, AIComputeEngine::Device device);
    
    void clear_data();
    std::string generate_report() const;
    void export_to_csv(const std::string& filename) const;
    
    // Analysis
    double get_average_time(const std::string& operation) const;
    double get_throughput(const std::string& operation) const; // MB/s
    std::vector<ProfileEntry> get_entries_by_operation(const std::string& operation) const;
};

// Utility macros for profiling
#define PROFILE_COMPUTE_OPERATION(name, device, size, code) \
    do { \
        auto start = std::chrono::steady_clock::now(); \
        { code } \
        auto end = std::chrono::steady_clock::now(); \
        auto duration = std::chrono::duration<double>(end - start).count(); \
        ComputeProfiler::instance().record_operation(name, duration, size, device); \
    } while(0)

} // namespace ai
} // namespace asekioml
