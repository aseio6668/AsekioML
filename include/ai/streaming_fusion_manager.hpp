#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>

namespace asekioml {
namespace ai {

// Forward declarations and types needed by StreamingFusionManager
struct StreamingChunk {
    std::vector<float> data;
    std::string type;
    double timestamp;
    size_t size;
};

struct MultiModalContent {
    std::vector<float> video_data;
    std::vector<float> audio_data;
    std::string text_data;
    std::map<std::string, double> metadata;
};

struct StreamingConfig {
    size_t buffer_size;
    float quality_threshold;
    bool enable_gpu;
};

class StreamingFusionManager {
public:
    StreamingFusionManager(size_t buffer_size = 1024);
    ~StreamingFusionManager();

    bool initialize();
    void shutdown();
    
    bool processVideoFrame(const std::vector<float>& frame_data);
    bool processAudioSample(const std::vector<float>& audio_data);
    bool processTextInput(const std::string& text);
    
    std::vector<float> getFusedOutput();
    bool isInitialized() const;
    
    void setConfig(const StreamingConfig& config);
    StreamingConfig getConfig() const;
    
    // Additional methods needed by demos
    bool addStreamingChunk(const StreamingChunk& chunk);
    std::vector<MultiModalContent> processAvailableChunks();
    void flushBuffer();
    double getCurrentLatency() const;
    std::map<std::string, double> getStreamingStatistics() const;

private:
    size_t buffer_size_;
    bool is_initialized_;
    StreamingConfig config_;
    
    std::vector<float> video_buffer_;
    std::vector<float> audio_buffer_;
    std::string text_buffer_;
};

} // namespace ai
} // namespace asekioml
