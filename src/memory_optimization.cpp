#include "memory_optimization.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace clmodel {
namespace memory {

// Memory pool implementation
MemoryPool::MemoryPool() {
    size_classes_.resize(NUM_SIZE_CLASSES);
    
    // Pre-allocate some blocks for common sizes
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        size_t block_size = MIN_BLOCK_SIZE << i;
        if (block_size <= MAX_BLOCK_SIZE) {
            // Pre-allocate 4 blocks of each size
            for (int j = 0; j < 4; ++j) {                Block block;
                block.size = block_size;
                block.in_use = false;
#ifdef _WIN32
                block.data = _aligned_malloc(block_size, 64);
#else
                block.data = std::aligned_alloc(64, block_size);
#endif
                block.last_used = std::chrono::steady_clock::now();
                
                if (block.data) {
                    size_classes_[i].push_back(std::move(block));
                }
            }
        }
    }
}

MemoryPool::~MemoryPool() {
    for (auto& size_class : size_classes_) {
        for (auto& block : size_class) {
            if (block.data) {
#ifdef _WIN32
                _aligned_free(block.data);
#else
                free(block.data);
#endif
            }
        }
    }
}

size_t MemoryPool::get_size_class(size_t size) const {
    if (size <= MIN_BLOCK_SIZE) return 0;
    
    // Find the appropriate size class (power of 2)
    size_t size_class = 0;
    size_t class_size = MIN_BLOCK_SIZE;
    
    while (class_size < size && size_class < NUM_SIZE_CLASSES - 1) {
        class_size <<= 1;
        size_class++;
    }
    
    return std::min(size_class, NUM_SIZE_CLASSES - 1);
}

void* MemoryPool::allocate(size_t size) {
    if (size == 0) return nullptr;
    if (size > MAX_BLOCK_SIZE) {
        // Fall back to system allocator for very large allocations
#ifdef _WIN32
        return _aligned_malloc(size, 64);
#else
        return std::aligned_alloc(64, size);
#endif
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t size_class = get_size_class(size);
    auto& blocks = size_classes_[size_class];
    
    // Find an available block
    for (auto& block : blocks) {
        if (!block.in_use) {
            block.in_use = true;
            block.last_used = std::chrono::steady_clock::now();
            
            total_allocated_ += block.size;
            peak_usage_ = std::max(peak_usage_.load(), total_allocated_.load());
            
            return block.data;
        }
    }
    
    // Allocate a new block
    Block new_block;
    new_block.size = MIN_BLOCK_SIZE << size_class;
    new_block.in_use = true;
    new_block.last_used = std::chrono::steady_clock::now();
    
#ifdef _WIN32
    new_block.data = _aligned_malloc(new_block.size, 64);
#else
    new_block.data = std::aligned_alloc(64, new_block.size);
#endif
    
    if (!new_block.data) {
        throw std::bad_alloc();
    }
    
    total_allocated_ += new_block.size;
    peak_usage_ = std::max(peak_usage_.load(), total_allocated_.load());
    
    void* result = new_block.data;
    blocks.push_back(std::move(new_block));
    
    return result;
}

void MemoryPool::deallocate(void* ptr, size_t size) {
    if (!ptr) return;
    
    if (size > MAX_BLOCK_SIZE) {
        // Was allocated by system allocator
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
        return;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t size_class = get_size_class(size);
    auto& blocks = size_classes_[size_class];
    
    // Find the block and mark it as available
    for (auto& block : blocks) {
        if (block.data == ptr) {
            block.in_use = false;
            block.last_used = std::chrono::steady_clock::now();
            total_allocated_ -= block.size;
            return;
        }
    }
    
    // If not found in expected size class, search all classes
    for (auto& size_class_blocks : size_classes_) {
        for (auto& block : size_class_blocks) {
            if (block.data == ptr) {
                block.in_use = false;
                block.last_used = std::chrono::steady_clock::now();
                total_allocated_ -= block.size;
                return;
            }
        }
    }
}

void MemoryPool::cleanup_unused(std::chrono::milliseconds max_age) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    for (auto& size_class : size_classes_) {
        auto it = std::remove_if(size_class.begin(), size_class.end(),
            [&](const Block& block) {
                if (!block.in_use && 
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - block.last_used) > max_age) {
#ifdef _WIN32
                    _aligned_free(block.data);
#else
                    free(block.data);
#endif
                    return true;
                }
                return false;
            });
        
        size_class.erase(it, size_class.end());
    }
}

MemoryPool& MemoryPool::instance() {
    static MemoryPool pool;
    return pool;
}

} // namespace memory
} // namespace clmodel
