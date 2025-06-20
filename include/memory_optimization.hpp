#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace asekioml {
namespace memory {

// Cache-aligned memory allocator
template<typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    
    T* allocate(size_t n) {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            ptr = nullptr;
        }
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, size_t) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

// Memory pool with different size classes
class MemoryPool {
private:
    struct Block {
        void* data;
        size_t size;
        bool in_use;
        std::chrono::time_point<std::chrono::steady_clock> last_used;
    };
    
    std::vector<std::vector<Block>> size_classes_;
    std::mutex pool_mutex_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_usage_{0};
    std::atomic<size_t> allocation_count_{0};
    
    static constexpr size_t MIN_BLOCK_SIZE = 1024;  // 1KB
    static constexpr size_t MAX_BLOCK_SIZE = 1024 * 1024 * 64;  // 64MB
    static constexpr size_t NUM_SIZE_CLASSES = 16;
    
    size_t get_size_class(size_t size) const;
    
public:
    MemoryPool();
    ~MemoryPool();
    
    // Singleton pattern
    static MemoryPool& get_instance() {
        static MemoryPool instance;
        return instance;
    }
    
    // Delete copy constructor and assignment operator
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);
    
    // Memory statistics
    size_t get_total_allocated() const { return total_allocated_.load(); }
    size_t get_peak_usage() const { return peak_usage_.load(); }
    
    // Additional statistics methods
    size_t get_allocation_count() const { return allocation_count_.load(); }
    size_t get_peak_allocated() const { return peak_usage_.load(); }
    
    // Manual tracking methods for objects not using the pool directly
    void track_allocation(size_t size) {
        total_allocated_ += size;
        allocation_count_ += 1;
        size_t current = total_allocated_.load();
        size_t peak = peak_usage_.load();
        if (current > peak) {
            peak_usage_ = current;
        }
    }
    
    void track_deallocation(size_t size) {
        total_allocated_ -= size;
        allocation_count_ -= 1;
    }
    
    // Cleanup unused blocks
    void cleanup_unused(std::chrono::milliseconds max_age = std::chrono::milliseconds(5000));
    
    static MemoryPool& instance();
};

// Smart pointer with custom deleter for pool allocation
template<typename T>
class PoolPtr {
private:
    T* ptr_;
    size_t size_;
    
public:
    explicit PoolPtr(size_t count = 1) : size_(count * sizeof(T)) {
        ptr_ = static_cast<T*>(MemoryPool::instance().allocate(size_));
        if (ptr_ && std::is_default_constructible_v<T>) {
            for (size_t i = 0; i < count; ++i) {
                new (ptr_ + i) T();
            }
        }
    }
    
    ~PoolPtr() {
        if (ptr_) {
            if constexpr (std::is_destructible_v<T>) {
                size_t count = size_ / sizeof(T);
                for (size_t i = 0; i < count; ++i) {
                    (ptr_ + i)->~T();
                }
            }
            MemoryPool::instance().deallocate(ptr_, size_);
        }
    }
    
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T& operator[](size_t index) const { return ptr_[index]; }
    
    // Move semantics
    PoolPtr(PoolPtr&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            this->~PoolPtr();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Disable copy
    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;
};

// Cache-friendly matrix storage with data locality optimization
template<typename T>
class CacheFriendlyMatrix {
private:
    PoolPtr<T> data_;
    size_t rows_;
    size_t cols_;
    size_t row_stride_;  // For memory alignment
    
public:
    CacheFriendlyMatrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols) {
        // Align rows to cache line boundaries
        row_stride_ = ((cols * sizeof(T) + 63) / 64) * 64 / sizeof(T);
        data_ = PoolPtr<T>(rows_ * row_stride_);
    }
    
    T& operator()(size_t row, size_t col) {
        return data_[row * row_stride_ + col];
    }
    
    const T& operator()(size_t row, size_t col) const {
        return data_[row * row_stride_ + col];
    }
    
    // Cache-friendly iteration
    class RowIterator {
        T* ptr_;
        size_t stride_;
    public:
        RowIterator(T* ptr, size_t stride) : ptr_(ptr), stride_(stride) {}
        T* begin() { return ptr_; }
        T* end() { return ptr_ + stride_; }
    };
    
    RowIterator row(size_t r) { return RowIterator(&data_[r * row_stride_], cols_); }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t stride() const { return row_stride_; }
};

} // namespace memory
} // namespace asekioml
