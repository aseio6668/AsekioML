#include "../../include/ai/compute_engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <immintrin.h> // For AVX/SSE

#ifdef CLMODEL_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace clmodel {
namespace ai {

// ================================================================================================
// ThreadPool Implementation
// ================================================================================================

ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    
                    if (stop_ && tasks_.empty()) return;
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return res;
}

void ThreadPool::wait_for_all() {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (tasks_.empty()) break;
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// ================================================================================================
// AIComputeEngine Implementation
// ================================================================================================

AIComputeEngine& AIComputeEngine::instance() {
    static AIComputeEngine instance;
    return instance;
}

AIComputeEngine::AIComputeEngine() 
    : num_threads_(std::thread::hardware_concurrency()) {
    thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
}

AIComputeEngine::~AIComputeEngine() {
    cleanup_gpu_contexts();
}

void AIComputeEngine::set_num_threads(size_t num_threads) {
    num_threads_ = num_threads;
    thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
}

template<typename T>
void AIComputeEngine::parallel_for(size_t start, size_t end, std::function<void(size_t)> func) {
    if (end <= start) return;
    
    size_t range = end - start;
    size_t chunk_size = get_optimal_chunk_size(range);
    size_t num_chunks = (range + chunk_size - 1) / chunk_size;
    
    std::vector<std::future<void>> futures;
    
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t chunk_start = start + i * chunk_size;
        size_t chunk_end = std::min(start + (i + 1) * chunk_size, end);
        
        futures.push_back(thread_pool_->enqueue([func, chunk_start, chunk_end]() {
            for (size_t j = chunk_start; j < chunk_end; ++j) {
                func(j);
            }
        }));
    }
    
    for (auto& future : futures) {
        future.get();
    }
}

template<typename T>
void AIComputeEngine::parallel_transform(const T* input, T* output, size_t size,
                                        std::function<T(T)> func) {
    parallel_for<T>(0, size, [&](size_t i) {
        output[i] = func(input[i]);
    });
}

void AIComputeEngine::vector_add(const double* a, const double* b, double* result, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (use_simd_ && size >= 256) {
        vector_add_simd(a, b, result, size);
    } else if (size >= gpu_threshold_ && use_gpu_ && is_gpu_available(preferred_device_)) {
        vector_add_cuda(a, b, result, size);
    } else {
        // CPU fallback
        parallel_for<size_t>(0, size, [&](size_t i) {
            result[i] = a[i] + b[i];
        });
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    update_stats(false, duration);
}

void AIComputeEngine::vector_sub(const double* a, const double* b, double* result, size_t size) {
    parallel_for<size_t>(0, size, [&](size_t i) {
        result[i] = a[i] - b[i];
    });
}

void AIComputeEngine::vector_mul(const double* a, const double* b, double* result, size_t size) {
    if (use_simd_ && size >= 256) {
        vector_mul_simd(a, b, result, size);
    } else {
        parallel_for<size_t>(0, size, [&](size_t i) {
            result[i] = a[i] * b[i];
        });
    }
}

void AIComputeEngine::vector_div(const double* a, const double* b, double* result, size_t size) {
    parallel_for<size_t>(0, size, [&](size_t i) {
        result[i] = a[i] / b[i];
    });
}

void AIComputeEngine::vector_scale(const double* a, double scalar, double* result, size_t size) {
    parallel_for<size_t>(0, size, [&](size_t i) {
        result[i] = a[i] * scalar;
    });
}

double AIComputeEngine::vector_sum(const double* data, size_t size) {
    const size_t chunk_size = get_optimal_chunk_size(size);
    const size_t num_chunks = (size + chunk_size - 1) / chunk_size;
    
    std::vector<std::future<double>> futures;
    
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min((i + 1) * chunk_size, size);
        
        futures.push_back(thread_pool_->enqueue([data, start, end]() {
            double sum = 0.0;
            for (size_t j = start; j < end; ++j) {
                sum += data[j];
            }
            return sum;
        }));
    }
    
    double total_sum = 0.0;
    for (auto& future : futures) {
        total_sum += future.get();
    }
    
    return total_sum;
}

double AIComputeEngine::vector_max(const double* data, size_t size) {
    if (size == 0) return 0.0;
    
    const size_t chunk_size = get_optimal_chunk_size(size);
    const size_t num_chunks = (size + chunk_size - 1) / chunk_size;
    
    std::vector<std::future<double>> futures;
    
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min((i + 1) * chunk_size, size);
        
        futures.push_back(thread_pool_->enqueue([data, start, end]() {
            double max_val = data[start];
            for (size_t j = start + 1; j < end; ++j) {
                max_val = std::max(max_val, data[j]);
            }
            return max_val;
        }));
    }
    
    double global_max = futures[0].get();
    for (size_t i = 1; i < futures.size(); ++i) {
        global_max = std::max(global_max, futures[i].get());
    }
    
    return global_max;
}

double AIComputeEngine::vector_min(const double* data, size_t size) {
    if (size == 0) return 0.0;
    
    const size_t chunk_size = get_optimal_chunk_size(size);
    const size_t num_chunks = (size + chunk_size - 1) / chunk_size;
    
    std::vector<std::future<double>> futures;
    
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min((i + 1) * chunk_size, size);
        
        futures.push_back(thread_pool_->enqueue([data, start, end]() {
            double min_val = data[start];
            for (size_t j = start + 1; j < end; ++j) {
                min_val = std::min(min_val, data[j]);
            }
            return min_val;
        }));
    }
    
    double global_min = futures[0].get();
    for (size_t i = 1; i < futures.size(); ++i) {
        global_min = std::min(global_min, futures[i].get());
    }
    
    return global_min;
}

double AIComputeEngine::vector_dot(const double* a, const double* b, size_t size) {
    const size_t chunk_size = get_optimal_chunk_size(size);
    const size_t num_chunks = (size + chunk_size - 1) / chunk_size;
    
    std::vector<std::future<double>> futures;
    
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min((i + 1) * chunk_size, size);
        
        futures.push_back(thread_pool_->enqueue([a, b, start, end]() {
            double dot = 0.0;
            for (size_t j = start; j < end; ++j) {
                dot += a[j] * b[j];
            }
            return dot;
        }));
    }
    
    double total_dot = 0.0;
    for (auto& future : futures) {
        total_dot += future.get();
    }
    
    return total_dot;
}

void AIComputeEngine::matrix_multiply(const double* a, const double* b, double* c,
                                     size_t m, size_t n, size_t k) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (m * n * k >= gpu_threshold_ && use_gpu_ && is_gpu_available(preferred_device_)) {
        matrix_multiply_cuda(a, b, c, m, n, k);
    } else {
        // CPU implementation with cache-friendly blocking
        const size_t block_size = 64;
        
        parallel_for<size_t>(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; j += block_size) {
                for (size_t kk = 0; kk < k; kk += block_size) {
                    size_t j_end = std::min(j + block_size, n);
                    size_t k_end = std::min(kk + block_size, k);
                    
                    for (size_t jj = j; jj < j_end; ++jj) {
                        double sum = 0.0;
                        for (size_t kkk = kk; kkk < k_end; ++kkk) {
                            sum += a[i * k + kkk] * b[kkk * n + jj];
                        }
                        if (kk == 0) {
                            c[i * n + jj] = sum;
                        } else {
                            c[i * n + jj] += sum;
                        }
                    }
                }
            }
        });
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    update_stats(false, duration);
}

void AIComputeEngine::matrix_transpose(const double* input, double* output,
                                      size_t rows, size_t cols) {
    parallel_for<size_t>(0, rows, [&](size_t i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    });
}

void AIComputeEngine::relu(const double* input, double* output, size_t size) {
    parallel_for<size_t>(0, size, [&](size_t i) {
        output[i] = std::max(0.0, input[i]);
    });
}

void AIComputeEngine::relu_derivative(const double* input, double* output, size_t size) {
    parallel_for<size_t>(0, size, [&](size_t i) {
        output[i] = input[i] > 0.0 ? 1.0 : 0.0;
    });
}

void AIComputeEngine::sigmoid(const double* input, double* output, size_t size) {
    parallel_for<size_t>(0, size, [&](size_t i) {
        output[i] = 1.0 / (1.0 + std::exp(-input[i]));
    });
}

void AIComputeEngine::tanh_activation(const double* input, double* output, size_t size) {
    parallel_for<size_t>(0, size, [&](size_t i) {
        output[i] = std::tanh(input[i]);
    });
}

void AIComputeEngine::softmax(const double* input, double* output, size_t size) {
    // Find max for numerical stability
    double max_val = vector_max(input, size);
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    vector_scale(output, 1.0 / sum, output, size);
}

void AIComputeEngine::vector_add_simd(const double* a, const double* b, double* result, size_t size) {
#ifdef __AVX__
    const size_t simd_size = 4; // 4 doubles per AVX register
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
#else
    // Fallback to regular implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
#endif
}

void AIComputeEngine::vector_mul_simd(const double* a, const double* b, double* result, size_t size) {
#ifdef __AVX__
    const size_t simd_size = 4;
    const size_t simd_end = (size / simd_size) * simd_size;
    
    for (size_t i = 0; i < simd_end; i += simd_size) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    for (size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
#endif
}

void AIComputeEngine::vector_add_cuda(const double* a, const double* b, double* result, size_t size) {
#ifdef CLMODEL_CUDA_SUPPORT
    // Simplified CUDA implementation - in reality you'd need proper kernel launching
    (void)a; (void)b; (void)result; (void)size;
    // cudaMemcpy, kernel launch, etc.
#else
    // Fallback to CPU
    vector_add(a, b, result, size);
#endif
}

void AIComputeEngine::matrix_multiply_cuda(const double* a, const double* b, double* c,
                                          size_t m, size_t n, size_t k) {
#ifdef CLMODEL_CUDA_SUPPORT
    // Simplified - would use cuBLAS for optimal performance
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
#else
    // Fallback to CPU
    matrix_multiply(a, b, c, m, n, k);
#endif
}

size_t AIComputeEngine::get_optimal_chunk_size(size_t total_size) const {
    size_t chunk_size = total_size / num_threads_;
    const size_t min_chunk_size = 1024;
    const size_t max_chunk_size = 65536;
    
    return std::max(min_chunk_size, std::min(max_chunk_size, chunk_size));
}

void AIComputeEngine::update_stats(bool is_gpu_operation, double elapsed_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (is_gpu_operation) {
        stats_.gpu_operations++;
        stats_.gpu_time += elapsed_time;
    } else {
        stats_.cpu_operations++;
        stats_.cpu_time += elapsed_time;
    }
}

bool AIComputeEngine::initialize_cuda() {
#ifdef CLMODEL_CUDA_SUPPORT
    int device_count;
    cudaError_t result = cudaGetDeviceCount(&device_count);
    cuda_initialized_ = (result == cudaSuccess && device_count > 0);
    return cuda_initialized_;
#else
    return false;
#endif
}

bool AIComputeEngine::initialize_opencl() {
    // OpenCL initialization would go here
    return false;
}

bool AIComputeEngine::initialize_rocm() {
#ifdef CLMODEL_ROCM_SUPPORT
    // ROCm initialization would go here
    return false;
#else
    return false;
#endif
}

void AIComputeEngine::cleanup_gpu_contexts() {
    // Cleanup GPU contexts if needed
}

AIComputeEngine::Device AIComputeEngine::select_optimal_device(size_t operation_size) const {
    if (!use_gpu_ || operation_size < gpu_threshold_) {
        return Device::CPU;
    }
    
    switch (preferred_device_) {
        case Device::CUDA:
            return cuda_initialized_ ? Device::CUDA : Device::CPU;
        case Device::OPENCL:
            return opencl_initialized_ ? Device::OPENCL : Device::CPU;
        case Device::ROCm:
            return rocm_initialized_ ? Device::ROCm : Device::CPU;
        default:
            return Device::CPU;
    }
}

bool AIComputeEngine::is_gpu_available(Device device) const {
    switch (device) {
        case Device::CUDA: return cuda_initialized_;
        case Device::OPENCL: return opencl_initialized_;
        case Device::ROCm: return rocm_initialized_;
        default: return true; // CPU always available
    }
}

void AIComputeEngine::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = ComputeStats{};
}

std::string AIComputeEngine::get_performance_report() const {
    std::ostringstream oss;
    oss << "=== AI Compute Engine Report ===\n";
    oss << "CPU Operations: " << stats_.cpu_operations << "\n";
    oss << "GPU Operations: " << stats_.gpu_operations << "\n";
    oss << "CPU Time: " << std::fixed << std::setprecision(3) << stats_.cpu_time << "s\n";
    oss << "GPU Time: " << stats_.gpu_time << "s\n";
    oss << "Threads: " << num_threads_ << "\n";
    oss << "SIMD Enabled: " << (use_simd_ ? "Yes" : "No") << "\n";
    oss << "GPU Enabled: " << (use_gpu_ ? "Yes" : "No") << "\n";
    oss << "GPU Threshold: " << gpu_threshold_ << " elements\n";
    
    return oss.str();
}

// ================================================================================================
// ComputeProfiler Implementation
// ================================================================================================

ComputeProfiler& ComputeProfiler::instance() {
    static ComputeProfiler instance;
    return instance;
}

void ComputeProfiler::record_operation(const std::string& name, double elapsed_time,
                                      size_t data_size, AIComputeEngine::Device device) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(profile_mutex_);
    profile_data_.push_back({
        name, elapsed_time, data_size, device,
        std::chrono::steady_clock::now()
    });
}

void ComputeProfiler::clear_data() {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    profile_data_.clear();
}

std::string ComputeProfiler::generate_report() const {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    std::ostringstream oss;
    oss << "=== Compute Profiler Report ===\n";
    oss << "Total Operations: " << profile_data_.size() << "\n\n";
    
    // Group by operation name
    std::unordered_map<std::string, std::vector<ProfileEntry>> grouped;
    for (const auto& entry : profile_data_) {
        grouped[entry.operation_name].push_back(entry);
    }
    
    for (const auto& [name, entries] : grouped) {
        double total_time = 0.0;
        size_t total_size = 0;
        for (const auto& entry : entries) {
            total_time += entry.elapsed_time;
            total_size += entry.data_size;
        }
        
        double avg_time = total_time / entries.size();
        double throughput = (total_size / 1024.0 / 1024.0) / total_time; // MB/s
        
        oss << name << ":\n";
        oss << "  Count: " << entries.size() << "\n";
        oss << "  Avg Time: " << std::fixed << std::setprecision(6) << avg_time << "s\n";
        oss << "  Total Time: " << total_time << "s\n";
        oss << "  Throughput: " << std::setprecision(2) << throughput << " MB/s\n\n";
    }
    
    return oss.str();
}

double ComputeProfiler::get_average_time(const std::string& operation) const {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    double total_time = 0.0;
    size_t count = 0;
    
    for (const auto& entry : profile_data_) {
        if (entry.operation_name == operation) {
            total_time += entry.elapsed_time;
            count++;
        }
    }
    
    return count > 0 ? total_time / count : 0.0;
}

double ComputeProfiler::get_throughput(const std::string& operation) const {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    
    double total_time = 0.0;
    size_t total_size = 0;
    
    for (const auto& entry : profile_data_) {
        if (entry.operation_name == operation) {
            total_time += entry.elapsed_time;
            total_size += entry.data_size;
        }
    }
    
    return total_time > 0.0 ? (total_size / 1024.0 / 1024.0) / total_time : 0.0;
}

} // namespace ai
} // namespace clmodel
