#include "../../include/ai/memory_manager.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#else
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

// GPU support headers
#ifdef CLMODEL_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

#ifdef CLMODEL_OPENCL_SUPPORT
#include <CL/cl.h>
#endif

#ifdef CLMODEL_ROCM_SUPPORT
#include <hip/hip_runtime.h>
#endif

namespace clmodel {
namespace ai {

AIMemoryManager& AIMemoryManager::instance() {
    static AIMemoryManager instance;
    return instance;
}

AIMemoryManager::~AIMemoryManager() {
    clear_pools();
}

void* AIMemoryManager::allocate(size_t size, size_t alignment, MemoryType type) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    stats_.total_allocations++;
    
    void* ptr = nullptr;
    
    // Try to get from pool first if pooling is enabled
    if (enable_pooling_ && size >= min_block_size_) {
        MemoryBlock* block = find_free_block(size, type);
        if (block) {
            block->in_use = true;
            ptr = block->ptr;
            stats_.pool_hits++;
        }
    }
    
    // Allocate new memory if not found in pool
    if (!ptr) {
        stats_.pool_misses++;
        
        switch (type) {
            case MemoryType::CPU:
                ptr = allocate_cpu(size, alignment);
                break;
            case MemoryType::CUDA:
                ptr = allocate_cuda(size);
                break;
            case MemoryType::OPENCL:
                ptr = allocate_opencl(size);
                break;
            case MemoryType::ROCm:
                ptr = allocate_rocm(size);
                break;
        }
    }
    
    if (ptr) {
        stats_.current_usage += size;
        stats_.total_allocated += size;
        stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
    }
    
    return ptr;
}

void AIMemoryManager::deallocate(void* ptr, size_t size, MemoryType type) {
    if (!ptr || size == 0) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    stats_.total_deallocations++;
    stats_.current_usage -= size;
    
    // Return to pool if pooling is enabled and size is appropriate
    if (enable_pooling_ && size >= min_block_size_ && get_pool_size() < max_pool_size_) {
        return_to_pool(ptr, size, type);
    } else {
        // Direct deallocation
        switch (type) {
            case MemoryType::CPU:
                deallocate_cpu(ptr, size);
                break;
            case MemoryType::CUDA:
                deallocate_cuda(ptr, size);
                break;
            case MemoryType::OPENCL:
                deallocate_opencl(ptr, size);
                break;
            case MemoryType::ROCm:
                deallocate_rocm(ptr, size);
                break;
        }
    }
}

void* AIMemoryManager::allocate_cpu(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void* AIMemoryManager::allocate_cuda(size_t size) {
#ifdef CLMODEL_CUDA_SUPPORT
    if (!cuda_available_) return nullptr;
    
    void* ptr = nullptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    return (result == cudaSuccess) ? ptr : nullptr;
#else
    (void)size; // Suppress unused parameter warning
    return nullptr;
#endif
}

void* AIMemoryManager::allocate_opencl(size_t size) {
#ifdef CLMODEL_OPENCL_SUPPORT
    if (!opencl_available_) return nullptr;
    
    // OpenCL allocation would require context - simplified implementation
    // In a real implementation, you'd need to maintain OpenCL context and command queue
    (void)size;
    return nullptr;
#else
    (void)size;
    return nullptr;
#endif
}

void* AIMemoryManager::allocate_rocm(size_t size) {
#ifdef CLMODEL_ROCM_SUPPORT
    if (!rocm_available_) return nullptr;
    
    void* ptr = nullptr;
    hipError_t result = hipMalloc(&ptr, size);
    return (result == hipSuccess) ? ptr : nullptr;
#else
    (void)size;
    return nullptr;
#endif
}

void AIMemoryManager::deallocate_cpu(void* ptr, size_t size) {
    (void)size; // Size not needed for CPU deallocation
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void AIMemoryManager::deallocate_cuda(void* ptr, size_t size) {
    (void)size;
#ifdef CLMODEL_CUDA_SUPPORT
    if (cuda_available_) {
        cudaFree(ptr);
    }
#endif
}

void AIMemoryManager::deallocate_opencl(void* ptr, size_t size) {
    (void)ptr;
    (void)size;
#ifdef CLMODEL_OPENCL_SUPPORT
    // OpenCL deallocation would require context
    // clReleaseMemObject((cl_mem)ptr);
#endif
}

void AIMemoryManager::deallocate_rocm(void* ptr, size_t size) {
    (void)size;
#ifdef CLMODEL_ROCM_SUPPORT
    if (rocm_available_) {
        hipFree(ptr);
    }
#endif
}

size_t AIMemoryManager::get_pool_key(size_t size) const {
    // Round up to next power of 2
    size_t key = min_block_size_;
    while (key < size) {
        key *= 2;
    }
    return key;
}

AIMemoryManager::MemoryBlock* AIMemoryManager::find_free_block(size_t size, MemoryType type) {
    size_t key = get_pool_key(size);
    
    auto& pool = memory_pools_[key];
    for (auto& block : pool) {
        if (!block.in_use && block.type == type && block.size >= size) {
            return &block;
        }
    }
    
    return nullptr;
}

void AIMemoryManager::return_to_pool(void* ptr, size_t size, MemoryType type) {
    size_t key = get_pool_key(size);
    
    // Check if we already have this block in the pool
    auto& pool = memory_pools_[key];
    for (auto& block : pool) {
        if (block.ptr == ptr) {
            block.in_use = false;
            return;
        }
    }
    
    // Add new block to pool
    pool.emplace_back(ptr, size, type);
}

void AIMemoryManager::clear_pools() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    for (auto& [key, pool] : memory_pools_) {
        for (auto& block : pool) {
            if (!block.in_use) {
                switch (block.type) {
                    case MemoryType::CPU:
                        deallocate_cpu(block.ptr, block.size);
                        break;
                    case MemoryType::CUDA:
                        deallocate_cuda(block.ptr, block.size);
                        break;
                    case MemoryType::OPENCL:
                        deallocate_opencl(block.ptr, block.size);
                        break;
                    case MemoryType::ROCm:
                        deallocate_rocm(block.ptr, block.size);
                        break;
                }
            }
        }
    }
    
    memory_pools_.clear();
}

void AIMemoryManager::compact_pools() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    for (auto& [key, pool] : memory_pools_) {
        // Remove unused blocks that exceed a certain threshold
        auto it = std::remove_if(pool.begin(), pool.end(), 
            [this](const MemoryBlock& block) {
                if (!block.in_use && get_pool_size() > max_pool_size_ / 2) {
                    switch (block.type) {
                        case MemoryType::CPU:
                            deallocate_cpu(block.ptr, block.size);
                            break;
                        case MemoryType::CUDA:
                            deallocate_cuda(block.ptr, block.size);
                            break;
                        case MemoryType::OPENCL:
                            deallocate_opencl(block.ptr, block.size);
                            break;
                        case MemoryType::ROCm:
                            deallocate_rocm(block.ptr, block.size);
                            break;
                    }
                    return true;
                }
                return false;
            });
        
        pool.erase(it, pool.end());
    }
}

size_t AIMemoryManager::get_pool_size() const {
    size_t total_size = 0;
    for (const auto& [key, pool] : memory_pools_) {
        for (const auto& block : pool) {
            if (!block.in_use) {
                total_size += block.size;
            }
        }
    }
    return total_size;
}

void AIMemoryManager::reset_stats() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    stats_ = MemoryStats{};
}

std::string AIMemoryManager::get_memory_report() const {
    std::ostringstream oss;
    oss << "=== AI Memory Manager Report ===\n";
    oss << "Total Allocated: " << std::fixed << std::setprecision(2) 
        << (stats_.total_allocated / 1024.0 / 1024.0) << " MB\n";
    oss << "Peak Usage: " << (stats_.peak_usage / 1024.0 / 1024.0) << " MB\n";
    oss << "Current Usage: " << (stats_.current_usage / 1024.0 / 1024.0) << " MB\n";
    oss << "Pool Size: " << (get_pool_size() / 1024.0 / 1024.0) << " MB\n";
    oss << "Pool Hit Rate: " << std::setprecision(1)
        << (100.0 * stats_.pool_hits / std::max(static_cast<size_t>(1), stats_.pool_hits + stats_.pool_misses)) << "%\n";
    oss << "Total Allocations: " << stats_.total_allocations << "\n";
    oss << "Total Deallocations: " << stats_.total_deallocations << "\n";
    
    oss << "\nGPU Support:\n";
    oss << "CUDA: " << (cuda_available_ ? "Available" : "Not Available") << "\n";
    oss << "OpenCL: " << (opencl_available_ ? "Available" : "Not Available") << "\n";
    oss << "ROCm: " << (rocm_available_ ? "Available" : "Not Available") << "\n";
    
    return oss.str();
}

void AIMemoryManager::initialize_gpu_support() {
    detect_cuda_support();
    detect_opencl_support();
    detect_rocm_support();
}

void AIMemoryManager::detect_cuda_support() {
#ifdef CLMODEL_CUDA_SUPPORT
    int device_count = 0;
    cudaError_t result = cudaGetDeviceCount(&device_count);
    cuda_available_ = (result == cudaSuccess && device_count > 0);
#else
    cuda_available_ = false;
#endif
}

void AIMemoryManager::detect_opencl_support() {
#ifdef CLMODEL_OPENCL_SUPPORT
    cl_uint platform_count = 0;
    cl_int result = clGetPlatformIDs(0, nullptr, &platform_count);
    opencl_available_ = (result == CL_SUCCESS && platform_count > 0);
#else
    opencl_available_ = false;
#endif
}

void AIMemoryManager::detect_rocm_support() {
#ifdef CLMODEL_ROCM_SUPPORT
    int device_count = 0;
    hipError_t result = hipGetDeviceCount(&device_count);
    rocm_available_ = (result == hipSuccess && device_count > 0);
#else
    rocm_available_ = false;
#endif
}

void AIMemoryManager::synchronize_gpu(MemoryType type) {
    switch (type) {
        case MemoryType::CUDA:
#ifdef CLMODEL_CUDA_SUPPORT
            if (cuda_available_) {
                cudaDeviceSynchronize();
            }
#endif
            break;
        case MemoryType::ROCm:
#ifdef CLMODEL_ROCM_SUPPORT
            if (rocm_available_) {
                hipDeviceSynchronize();
            }
#endif
            break;
        default:
            // No synchronization needed for CPU or OpenCL
            break;
    }
}

void AIMemoryManager::copy_to_gpu(void* dst, const void* src, size_t size, MemoryType gpu_type) {
    switch (gpu_type) {
        case MemoryType::CUDA:
#ifdef CLMODEL_CUDA_SUPPORT
            if (cuda_available_) {
                cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
            }
#endif
            break;
        case MemoryType::ROCm:
#ifdef CLMODEL_ROCM_SUPPORT
            if (rocm_available_) {
                hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
            }
#endif
            break;
        default:
            // For CPU or unsupported types, use regular memcpy
            std::memcpy(dst, src, size);
            break;
    }
}

void AIMemoryManager::copy_from_gpu(void* dst, const void* src, size_t size, MemoryType gpu_type) {
    switch (gpu_type) {
        case MemoryType::CUDA:
#ifdef CLMODEL_CUDA_SUPPORT
            if (cuda_available_) {
                cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
            }
#endif
            break;
        case MemoryType::ROCm:
#ifdef CLMODEL_ROCM_SUPPORT
            if (rocm_available_) {
                hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
            }
#endif
            break;
        default:
            std::memcpy(dst, src, size);
            break;
    }
}

void AIMemoryManager::copy_gpu_to_gpu(void* dst, const void* src, size_t size, 
                                     MemoryType dst_type, MemoryType src_type) {
    if (dst_type == src_type) {
        switch (dst_type) {
            case MemoryType::CUDA:
#ifdef CLMODEL_CUDA_SUPPORT
                if (cuda_available_) {
                    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
                }
#endif
                break;
            case MemoryType::ROCm:
#ifdef CLMODEL_ROCM_SUPPORT
                if (rocm_available_) {
                    hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
                }
#endif
                break;
            default:
                std::memcpy(dst, src, size);
                break;
        }
    } else {
        // Cross-platform copy requires temporary CPU buffer
        auto temp_buffer = std::make_unique<char[]>(size);
        copy_from_gpu(temp_buffer.get(), src, size, src_type);
        copy_to_gpu(dst, temp_buffer.get(), size, dst_type);
    }
}

// ================================================================================================
// MemoryMappedTensor Implementation
// ================================================================================================

MemoryMappedTensor::MemoryMappedTensor(const std::string& filename, size_t size, bool read_only)
    : filename_(filename), mapped_data_(nullptr), size_(size), read_only_(read_only) {
    
#ifdef _WIN32
    HANDLE file_handle = CreateFileA(
        filename.c_str(),
        read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE),
        FILE_SHARE_READ,
        nullptr,
        read_only ? OPEN_EXISTING : CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    
    if (file_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file for memory mapping: " + filename);
    }
    
    HANDLE mapping_handle = CreateFileMappingA(
        file_handle,
        nullptr,
        read_only ? PAGE_READONLY : PAGE_READWRITE,
        static_cast<DWORD>(size >> 32),
        static_cast<DWORD>(size & 0xFFFFFFFF),
        nullptr
    );
    
    CloseHandle(file_handle);
    
    if (!mapping_handle) {
        throw std::runtime_error("Failed to create file mapping: " + filename);
    }
    
    mapped_data_ = MapViewOfFile(
        mapping_handle,
        read_only ? FILE_MAP_READ : FILE_MAP_WRITE,
        0, 0, size
    );
    
    CloseHandle(mapping_handle);
    
    if (!mapped_data_) {
        throw std::runtime_error("Failed to map view of file: " + filename);
    }
    
#else
    int fd = open(filename.c_str(), read_only ? O_RDONLY : (O_RDWR | O_CREAT), 0644);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file for memory mapping: " + filename);
    }
    
    if (!read_only) {
        // Extend file to required size
        if (ftruncate(fd, size) == -1) {
            close(fd);
            throw std::runtime_error("Failed to resize file: " + filename);
        }
    }
    
    mapped_data_ = mmap(
        nullptr, size,
        read_only ? PROT_READ : (PROT_READ | PROT_WRITE),
        MAP_SHARED,
        fd, 0
    );
    
    close(fd);
    
    if (mapped_data_ == MAP_FAILED) {
        throw std::runtime_error("Failed to memory map file: " + filename);
    }
#endif
}

MemoryMappedTensor::~MemoryMappedTensor() {
    if (mapped_data_) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_data_);
#else
        munmap(mapped_data_, size_);
#endif
    }
}

void MemoryMappedTensor::sync() {
    if (mapped_data_ && !read_only_) {
#ifdef _WIN32
        FlushViewOfFile(mapped_data_, size_);
#else
        msync(mapped_data_, size_, MS_SYNC);
#endif
    }
}

void MemoryMappedTensor::async_sync() {
    if (mapped_data_ && !read_only_) {
#ifdef _WIN32
        FlushViewOfFile(mapped_data_, size_);
#else
        msync(mapped_data_, size_, MS_ASYNC);
#endif
    }
}

} // namespace ai
} // namespace clmodel
