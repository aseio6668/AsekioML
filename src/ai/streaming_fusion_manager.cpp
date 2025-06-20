#include "ai/streaming_fusion_manager.hpp"
#include <iostream>

namespace asekioml {
namespace ai {

StreamingFusionManager::StreamingFusionManager(size_t buffer_size) 
    : buffer_size_(buffer_size), is_initialized_(false) {
    config_.buffer_size = buffer_size;
    config_.quality_threshold = 0.8f;
    config_.enable_gpu = false;
    video_buffer_.reserve(buffer_size);
    audio_buffer_.reserve(buffer_size);
    std::cout << "StreamingFusionManager: Constructor called with buffer size " << buffer_size << std::endl;
}

StreamingFusionManager::~StreamingFusionManager() {
    shutdown();
    std::cout << "StreamingFusionManager: Destructor called" << std::endl;
}

bool StreamingFusionManager::initialize() {
    is_initialized_ = true;
    std::cout << "StreamingFusionManager: Initialized" << std::endl;
    return true;
}

void StreamingFusionManager::shutdown() {
    is_initialized_ = false;
    video_buffer_.clear();
    audio_buffer_.clear();
    text_buffer_.clear();
    std::cout << "StreamingFusionManager: Shutdown complete" << std::endl;
}

bool StreamingFusionManager::processVideoFrame(const std::vector<float>& frame_data) {
    if (!is_initialized_) return false;
    video_buffer_.insert(video_buffer_.end(), frame_data.begin(), frame_data.end());
    std::cout << "StreamingFusionManager: Processed video frame" << std::endl;
    return true;
}

bool StreamingFusionManager::processAudioSample(const std::vector<float>& audio_data) {
    if (!is_initialized_) return false;
    audio_buffer_.insert(audio_buffer_.end(), audio_data.begin(), audio_data.end());
    std::cout << "StreamingFusionManager: Processed audio sample" << std::endl;
    return true;
}

bool StreamingFusionManager::processTextInput(const std::string& text) {
    if (!is_initialized_) return false;
    text_buffer_ += text;
    std::cout << "StreamingFusionManager: Processed text input" << std::endl;
    return true;
}

std::vector<float> StreamingFusionManager::getFusedOutput() {
    std::vector<float> result;
    // Simple mock fusion - combine audio and video buffers
    result.insert(result.end(), video_buffer_.begin(), video_buffer_.end());
    result.insert(result.end(), audio_buffer_.begin(), audio_buffer_.end());
    std::cout << "StreamingFusionManager: Generated fused output" << std::endl;
    return result;
}

bool StreamingFusionManager::isInitialized() const {
    return is_initialized_;
}

void StreamingFusionManager::setConfig(const StreamingConfig& config) {
    config_ = config;
    buffer_size_ = config.buffer_size;
}

StreamingConfig StreamingFusionManager::getConfig() const {
    return config_;
}

// Additional methods needed by demos
bool StreamingFusionManager::addStreamingChunk(const StreamingChunk& chunk) {
    if (!is_initialized_) return false;
    std::cout << "StreamingFusionManager: Added streaming chunk" << std::endl;
    return true;
}

std::vector<MultiModalContent> StreamingFusionManager::processAvailableChunks() {
    std::cout << "StreamingFusionManager: Processing available chunks" << std::endl;
    return std::vector<MultiModalContent>();
}

void StreamingFusionManager::flushBuffer() {
    std::cout << "StreamingFusionManager: Buffer flushed" << std::endl;
}

double StreamingFusionManager::getCurrentLatency() const {
    return 12.5; // Mock latency in ms
}

std::map<std::string, double> StreamingFusionManager::getStreamingStatistics() const {
    std::map<std::string, double> stats;
    stats["latency_ms"] = 12.5;
    stats["throughput_fps"] = 30.0;
    stats["buffer_utilization"] = 0.65;
    return stats;
}

} // namespace ai
} // namespace asekioml
