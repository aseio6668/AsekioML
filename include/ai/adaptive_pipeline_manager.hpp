#pragma once

namespace asekioml {
namespace ai {

/**
 * @brief Adaptive pipeline manager for cross-modal guidance systems
 */
class AdaptivePipelineManager {
public:
    AdaptivePipelineManager();
    ~AdaptivePipelineManager();
    
    bool initialize();
    void shutdown();
    bool adaptPipeline();
    
private:
    bool is_initialized_ = false;
};

} // namespace ai
} // namespace asekioml
