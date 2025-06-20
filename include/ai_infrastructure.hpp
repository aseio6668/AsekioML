#pragma once

#include "matrix.hpp"
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace clmodel {
namespace ai {

/**
 * @brief High-performance tensor operations for AI workloads
 */
class Tensor {
private:
    std::vector<double> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t total_size_;
    
public:
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& data);
    
    // Multi-dimensional indexing
    double& operator()(const std::vector<size_t>& indices);
    const double& operator()(const std::vector<size_t>& indices) const;
    
    // Shape operations
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor transpose(const std::vector<size_t>& axes) const;
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(int axis) const;
    
    // Broadcasting and element-wise operations
    Tensor broadcast_to(const std::vector<size_t>& target_shape) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // Reduction operations
    Tensor sum(int axis = -1, bool keepdim = false) const;
    Tensor mean(int axis = -1, bool keepdim = false) const;
    Tensor max(int axis = -1, bool keepdim = false) const;
    Tensor min(int axis = -1, bool keepdim = false) const;
    
    // Advanced operations for AI
    Tensor conv2d(const Tensor& kernel, int stride = 1, int padding = 0) const;
    Tensor conv_transpose2d(const Tensor& kernel, int stride = 1, int padding = 0) const;
    Tensor batch_norm(const Tensor& mean, const Tensor& var, const Tensor& gamma, const Tensor& beta) const;
    Tensor layer_norm(const Tensor& gamma, const Tensor& beta, double eps = 1e-5) const;
    
    // FFT operations for audio processing
    Tensor fft() const;
    Tensor ifft() const;
    Tensor spectrogram(int window_size, int hop_length) const;
    
    // Attention mechanisms
    Tensor scaled_dot_product_attention(const Tensor& key, const Tensor& value, const Tensor& mask = Tensor({0})) const;
    Tensor multi_head_attention(const Tensor& key, const Tensor& value, int num_heads) const;
    
    // Utility methods
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return total_size_; }
    size_t ndim() const { return shape_.size(); }
    std::vector<double>& data() { return data_; }
    const std::vector<double>& data() const { return data_; }
    
    // Conversion to/from Matrix
    Matrix to_matrix() const;
    static Tensor from_matrix(const Matrix& matrix);
    
private:
    void calculate_strides();
    size_t get_index(const std::vector<size_t>& indices) const;
};

/**
 * @brief Memory management for large AI models
 */
class AIMemoryManager {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_access;
    };
    
    std::vector<MemoryBlock> memory_pool_;
    std::mutex memory_mutex_;
    size_t total_allocated_;
    size_t memory_limit_;
    
public:
    AIMemoryManager(size_t memory_limit_gb = 8);
    ~AIMemoryManager();
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void garbage_collect();
    void set_memory_limit(size_t limit_gb);
    
    // Memory statistics
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_available_memory() const { return memory_limit_ - total_allocated_; }
    double get_memory_usage_percent() const { return (double)total_allocated_ / memory_limit_ * 100.0; }
};

/**
 * @brief Parallel processing infrastructure for AI workloads
 */
class AIComputeEngine {
private:
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_flag_;
    size_t num_threads_;
    
public:
    AIComputeEngine(size_t num_threads = std::thread::hardware_concurrency());
    ~AIComputeEngine();
    
    // Task submission
    template<typename F, typename... Args>
    auto submit_task(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Parallel operations
    void parallel_for(size_t start, size_t end, std::function<void(size_t)> func);
    void parallel_reduce(const std::vector<double>& data, std::function<double(double, double)> reducer, double& result);
    
    // Batch processing for AI inference
    std::vector<Tensor> process_batch(const std::vector<Tensor>& inputs, 
                                    std::function<Tensor(const Tensor&)> processor);
    
    void shutdown();
    
private:
    void worker_function();
};

/**
 * @brief Data preprocessing pipeline for AI applications
 */
class AIDataProcessor {
public:
    // Text processing
    static std::vector<int> tokenize_text(const std::string& text, const std::map<std::string, int>& vocab);
    static std::string detokenize_text(const std::vector<int>& tokens, const std::map<int, std::string>& vocab);
    static Tensor create_text_embeddings(const std::vector<int>& tokens, const Tensor& embedding_matrix);
    
    // Image processing
    static Tensor load_image(const std::string& filepath, size_t target_height = 0, size_t target_width = 0);
    static void save_image(const Tensor& image_tensor, const std::string& filepath);
    static Tensor resize_image(const Tensor& image, size_t new_height, size_t new_width);
    static Tensor normalize_image(const Tensor& image, const std::vector<double>& mean, const std::vector<double>& std);
    static Tensor augment_image(const Tensor& image, bool random_flip = true, double rotation_range = 15.0);
    
    // Audio processing
    static Tensor load_audio(const std::string& filepath, size_t target_sample_rate = 22050);
    static void save_audio(const Tensor& audio_tensor, const std::string& filepath, size_t sample_rate = 22050);
    static Tensor resample_audio(const Tensor& audio, size_t original_rate, size_t target_rate);
    static Tensor audio_to_spectrogram(const Tensor& audio, size_t window_size = 1024, size_t hop_length = 512);
    static Tensor spectrogram_to_audio(const Tensor& spectrogram, size_t hop_length = 512);
    
    // Video processing
    static std::vector<Tensor> load_video(const std::string& filepath, size_t max_frames = 0);
    static void save_video(const std::vector<Tensor>& frames, const std::string& filepath, double fps = 24.0);
    static std::vector<Tensor> resize_video(const std::vector<Tensor>& frames, size_t new_height, size_t new_width);
    static std::vector<Tensor> extract_video_features(const std::vector<Tensor>& frames);
    
    // Multi-modal synchronization
    static void synchronize_audio_video(Tensor& audio, std::vector<Tensor>& video_frames, double target_fps = 24.0);
    static Tensor create_subtitle_embeddings(const std::string& text, const std::vector<double>& timestamps);
    
private:
    static Tensor apply_fft_window(const Tensor& audio, const std::string& window_type = "hann");
    static std::vector<double> generate_hann_window(size_t window_size);
    static std::vector<double> generate_hamming_window(size_t window_size);
};

/**
 * @brief Real-time streaming infrastructure for AI applications
 */
class AIStreamProcessor {
private:
    struct StreamBuffer {
        std::queue<Tensor> buffer;
        std::mutex mutex;
        std::condition_variable condition;
        size_t max_size;
        bool is_active;
    };
    
    std::map<std::string, StreamBuffer> input_streams_;
    std::map<std::string, StreamBuffer> output_streams_;
    std::unique_ptr<AIComputeEngine> compute_engine_;
    bool is_processing_;
    
public:
    AIStreamProcessor(size_t buffer_size = 10, size_t num_workers = 4);
    ~AIStreamProcessor();
    
    // Stream management
    void create_input_stream(const std::string& stream_name, size_t buffer_size = 10);
    void create_output_stream(const std::string& stream_name, size_t buffer_size = 10);
    void close_stream(const std::string& stream_name);
    
    // Data streaming
    bool push_input(const std::string& stream_name, const Tensor& data);
    bool pop_output(const std::string& stream_name, Tensor& data);
    
    // Processing pipeline
    void set_processor(std::function<std::map<std::string, Tensor>(const std::map<std::string, Tensor>&)> processor);
    void start_processing();
    void stop_processing();
    
    // Real-time content generation
    void start_live_generation(std::function<Tensor(const Tensor&)> generator);
    void update_generation_prompt(const Tensor& new_prompt);
    
    // Performance monitoring
    double get_processing_latency() const;
    double get_throughput() const;
    size_t get_queue_size(const std::string& stream_name) const;
    
private:
    void processing_loop();
    void generation_loop(std::function<Tensor(const Tensor&)> generator);
};

/**
 * @brief Model quantization and optimization for deployment
 */
class AIModelOptimizer {
public:
    enum class QuantizationType {
        INT8,
        INT16,
        FLOAT16,
        DYNAMIC
    };
    
    enum class PruningType {
        MAGNITUDE,
        STRUCTURED,
        UNSTRUCTURED
    };
    
    // Model compression
    static void quantize_model(NeuralNetwork& model, QuantizationType type);
    static void prune_model(NeuralNetwork& model, PruningType type, double sparsity_ratio);
    static void compress_weights(NeuralNetwork& model, double compression_ratio);
    
    // Knowledge distillation
    static void distill_model(const NeuralNetwork& teacher, NeuralNetwork& student, 
                            const std::vector<Tensor>& training_data, double temperature = 3.0);
    
    // Architecture search
    static std::unique_ptr<NeuralNetwork> neural_architecture_search(
        const std::vector<Tensor>& training_data,
        const std::vector<Tensor>& validation_data,
        size_t max_iterations = 100);
    
    // Performance optimization
    static void optimize_for_inference(NeuralNetwork& model);
    static void optimize_for_mobile(NeuralNetwork& model);
    static void optimize_for_edge_device(NeuralNetwork& model, size_t memory_limit_mb);
    
    // Benchmarking
    struct PerformanceMetrics {
        double inference_time_ms;
        size_t memory_usage_mb;
        double flops;
        double accuracy;
        double power_consumption_watts;
    };
    
    static PerformanceMetrics benchmark_model(const NeuralNetwork& model, 
                                            const std::vector<Tensor>& test_data);
};

} // namespace ai
} // namespace clmodel
