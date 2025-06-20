#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstddef>

namespace clmodel {
namespace ai {

/**
 * @brief Advanced memory manager for efficient tensor allocation and reuse
 * 
 * Provides memory pooling, allocation tracking, and GPU memory management
 * to optimize performance for large tensor operations.
 */
class AIMemoryManager {
public:
    struct MemoryStats {
        size_t total_allocated = 0;
        size_t peak_usage = 0;
        size_t current_usage = 0;
        size_t pool_hits = 0;
        size_t pool_misses = 0;
        size_t total_allocations = 0;
        size_t total_deallocations = 0;
    };

    enum class MemoryType {
        CPU,
        CUDA,
        OPENCL,
        ROCm
    };

private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        MemoryType type;
        
        MemoryBlock(void* p, size_t s, MemoryType t) 
            : ptr(p), size(s), in_use(false), type(t) {}
    };

    // Memory pools for different sizes (power of 2 sizes)
    std::unordered_map<size_t, std::vector<MemoryBlock>> memory_pools_;
    std::mutex pool_mutex_;
    MemoryStats stats_;
    
    // Configuration
    size_t max_pool_size_ = 1024 * 1024 * 1024; // 1GB default
    size_t min_block_size_ = 1024; // 1KB minimum
    bool enable_pooling_ = true;
    
    // GPU memory management
    bool cuda_available_ = false;
    bool opencl_available_ = false;
    bool rocm_available_ = false;
    
public:
    static AIMemoryManager& instance();
    
    // Configuration
    void set_max_pool_size(size_t size) { max_pool_size_ = size; }
    void set_min_block_size(size_t size) { min_block_size_ = size; }
    void enable_memory_pooling(bool enable) { enable_pooling_ = enable; }
    
    // Memory allocation
    void* allocate(size_t size, size_t alignment = 32, MemoryType type = MemoryType::CPU);
    void deallocate(void* ptr, size_t size, MemoryType type = MemoryType::CPU);
    
    // Typed allocation helpers
    template<typename T>
    T* allocate_array(size_t count, MemoryType type = MemoryType::CPU) {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T), type));
    }
    
    template<typename T>
    void deallocate_array(T* ptr, size_t count, MemoryType type = MemoryType::CPU) {
        deallocate(static_cast<void*>(ptr), count * sizeof(T), type);
    }
    
    // Memory management
    void clear_pools();
    void compact_pools();
    size_t get_pool_size() const;
    
    // Statistics
    const MemoryStats& get_stats() const { return stats_; }
    void reset_stats();
    std::string get_memory_report() const;
    
    // GPU memory support
    bool is_cuda_available() const { return cuda_available_; }
    bool is_opencl_available() const { return opencl_available_; }
    bool is_rocm_available() const { return rocm_available_; }
    
    void initialize_gpu_support();
    void synchronize_gpu(MemoryType type = MemoryType::CUDA);
    
    // Memory copying between devices
    void copy_to_gpu(void* dst, const void* src, size_t size, MemoryType gpu_type = MemoryType::CUDA);
    void copy_from_gpu(void* dst, const void* src, size_t size, MemoryType gpu_type = MemoryType::CUDA);
    void copy_gpu_to_gpu(void* dst, const void* src, size_t size, 
                        MemoryType dst_type, MemoryType src_type);
    
private:
    AIMemoryManager() = default;
    ~AIMemoryManager();
    
    // Disable copy/move
    AIMemoryManager(const AIMemoryManager&) = delete;
    AIMemoryManager& operator=(const AIMemoryManager&) = delete;
    
    // Internal allocation methods
    void* allocate_cpu(size_t size, size_t alignment);
    void* allocate_cuda(size_t size);
    void* allocate_opencl(size_t size);
    void* allocate_rocm(size_t size);
    
    void deallocate_cpu(void* ptr, size_t size);
    void deallocate_cuda(void* ptr, size_t size);
    void deallocate_opencl(void* ptr, size_t size);
    void deallocate_rocm(void* ptr, size_t size);
    
    // Pool management
    size_t get_pool_key(size_t size) const;
    MemoryBlock* find_free_block(size_t size, MemoryType type);
    void return_to_pool(void* ptr, size_t size, MemoryType type);
    
    // Platform detection
    void detect_cuda_support();
    void detect_opencl_support();
    void detect_rocm_support();
};

/**
 * @brief RAII wrapper for automatic memory management
 */
template<typename T>
class ManagedArray {
private:
    T* data_;
    size_t size_;
    AIMemoryManager::MemoryType type_;
    
public:
    ManagedArray(size_t size, AIMemoryManager::MemoryType type = AIMemoryManager::MemoryType::CPU)
        : size_(size), type_(type) {
        data_ = AIMemoryManager::instance().allocate_array<T>(size, type);
    }
    
    ~ManagedArray() {
        if (data_) {
            AIMemoryManager::instance().deallocate_array(data_, size_, type_);
        }
    }
    
    // Move constructor
    ManagedArray(ManagedArray&& other) noexcept
        : data_(other.data_), size_(other.size_), type_(other.type_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // Move assignment
    ManagedArray& operator=(ManagedArray&& other) noexcept {
        if (this != &other) {
            if (data_) {
                AIMemoryManager::instance().deallocate_array(data_, size_, type_);
            }
            data_ = other.data_;
            size_ = other.size_;
            type_ = other.type_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Disable copy
    ManagedArray(const ManagedArray&) = delete;
    ManagedArray& operator=(const ManagedArray&) = delete;
    
    // Access
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    // Iterator support
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
};

/**
 * @brief Memory-mapped tensor for efficient large tensor operations
 */
class MemoryMappedTensor {
private:
    std::string filename_;
    void* mapped_data_;
    size_t size_;
    bool read_only_;
    
public:
    MemoryMappedTensor(const std::string& filename, size_t size, bool read_only = false);
    ~MemoryMappedTensor();
    
    // Disable copy/move for safety
    MemoryMappedTensor(const MemoryMappedTensor&) = delete;
    MemoryMappedTensor& operator=(const MemoryMappedTensor&) = delete;
    
    void* data() { return mapped_data_; }
    const void* data() const { return mapped_data_; }
    size_t size() const { return size_; }
    
    void sync();
    void async_sync();
};

} // namespace ai
} // namespace clmodel
