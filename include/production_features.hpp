#pragma once

#include "network.hpp"
#include "dataset.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <future>
#include <unordered_map>
#include <string>
#include <vector>

namespace asekioml {
namespace production {

// Model versioning and registry
class ModelRegistry {
private:
    struct ModelVersion {
        std::string name;
        std::string version;
        std::unique_ptr<NeuralNetwork> model;
        std::chrono::time_point<std::chrono::system_clock> created_at;
        std::unordered_map<std::string, std::string> metadata;
        double validation_score;
    };
      std::unordered_map<std::string, std::vector<ModelVersion>> models_;
    mutable std::mutex registry_mutex_;
    
public:
    // Model registration
    void register_model(const std::string& name, const std::string& version,
                       std::unique_ptr<NeuralNetwork> model,
                       double validation_score = 0.0,
                       const std::unordered_map<std::string, std::string>& metadata = {});
    
    // Model retrieval
    std::unique_ptr<NeuralNetwork> get_model(const std::string& name, 
                                            const std::string& version = "latest");
    
    // Model management
    std::vector<std::string> list_models() const;
    std::vector<std::string> list_versions(const std::string& name) const;
    void delete_model(const std::string& name, const std::string& version);
    
    // Model comparison
    std::vector<std::pair<std::string, double>> get_model_scores(const std::string& name) const;
    std::string get_best_version(const std::string& name) const;
    
    // Persistence
    void save_registry(const std::string& path) const;
    void load_registry(const std::string& path);
    
    static ModelRegistry& instance();
};

// High-performance inference server
class InferenceServer {
private:
    struct PredictionRequest {
        size_t request_id;
        Matrix input;
        std::promise<Matrix> result;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    };
    
    std::unique_ptr<NeuralNetwork> model_;
    std::queue<PredictionRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    // Performance monitoring
    std::atomic<size_t> total_requests_{0};
    std::atomic<size_t> successful_requests_{0};
    std::atomic<double> total_inference_time_{0.0};
    mutable std::mutex metrics_mutex_;
    std::vector<double> latency_history_;
    
    // Configuration
    size_t max_batch_size_;
    size_t num_workers_;
    std::chrono::milliseconds batch_timeout_;
    
public:
    struct ServerConfig {
        size_t max_batch_size = 32;
        size_t num_workers = std::thread::hardware_concurrency();
        std::chrono::milliseconds batch_timeout{100};
        bool enable_metrics = true;
    };
    
    InferenceServer(std::unique_ptr<NeuralNetwork> model, const ServerConfig& config = {});
    ~InferenceServer();
    
    // Server control
    void start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Synchronous prediction
    Matrix predict(const Matrix& input);
    
    // Asynchronous prediction
    std::future<Matrix> predict_async(const Matrix& input);
    
    // Batch prediction
    std::vector<Matrix> predict_batch(const std::vector<Matrix>& inputs);
    
    // Performance metrics
    struct PerformanceMetrics {
        size_t total_requests;
        size_t successful_requests;
        double average_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        double throughput_rps;
        double error_rate;
    };
    
    PerformanceMetrics get_metrics() const;
    void reset_metrics();
    
    // Health check
    bool health_check() const;

private:
    void worker_loop();
    void process_batch(std::vector<PredictionRequest>& batch);
    double calculate_percentile(const std::vector<double>& values, double percentile) const;
};

// Model monitoring and drift detection
class ModelMonitor {
public:
    struct DataDriftMetrics {
        double psi_score;           // Population Stability Index
        double wasserstein_distance; 
        double ks_statistic;        // Kolmogorov-Smirnov
        double chi_square_statistic; // Chi-square test for categorical data
        std::vector<double> feature_drift_scores;
        bool drift_detected;
    };
    
    struct PerformanceMetrics {
        double accuracy;
        double precision;
        double recall;
        double f1_score;
        std::chrono::time_point<std::chrono::system_clock> timestamp;
    };
    
private:
    std::string model_name_;
    Dataset reference_data_;
    std::vector<PerformanceMetrics> performance_history_;
    mutable std::mutex monitor_mutex_;
    
    // Thresholds
    double drift_threshold_;
    double performance_threshold_;
    
public:
    ModelMonitor(const std::string& model_name, const Dataset& reference_data,
                double drift_threshold = 0.2, double performance_threshold = 0.05);
    
    // Singleton pattern
    static ModelMonitor& get_instance() {
        static ModelMonitor instance("default_model", Dataset(), 0.2, 0.05);
        return instance;
    }
    
    // Delete copy constructor and assignment operator
    ModelMonitor(const ModelMonitor&) = delete;
    ModelMonitor& operator=(const ModelMonitor&) = delete;
    
    // Data drift detection
    DataDriftMetrics detect_data_drift(const Dataset& new_data) const;
    
    // Performance monitoring
    void log_performance(const PerformanceMetrics& metrics);
    bool detect_performance_degradation() const;
    
    // Alerts
    void set_alert_callback(std::function<void(const std::string&)> callback);
    
    // Reports
    void generate_monitoring_report(const std::string& output_path) const;

private:
    double calculate_psi(const Matrix& reference, const Matrix& current) const;
    double calculate_wasserstein_distance(const Matrix& reference, const Matrix& current) const;
    double calculate_ks_statistic(const Matrix& reference, const Matrix& current) const;
    double calculate_chi_square_statistic(const Matrix& reference, const Matrix& current) const;
    
    std::function<void(const std::string&)> alert_callback_;
};

// A/B testing framework for models
class ABTestFramework {
public:
    struct TestConfig {
        std::string test_name;
        std::vector<std::string> model_variants;
        std::vector<double> traffic_allocation;  // Should sum to 1.0
        size_t min_samples_per_variant = 1000;
        double confidence_level = 0.95;
        std::string metric = "accuracy";
    };
    
    struct TestResult {
        std::string winning_variant;
        double confidence;
        std::unordered_map<std::string, double> variant_metrics;
        bool test_complete;
        size_t total_samples;
    };
    
private:
    std::unordered_map<std::string, TestConfig> active_tests_;
    std::unordered_map<std::string, std::vector<double>> test_metrics_;
    std::mutex test_mutex_;
    
public:
    // Test management
    void start_test(const TestConfig& config);
    void stop_test(const std::string& test_name);
    
    // Traffic routing
    std::string route_request(const std::string& test_name, const std::string& user_id);
    
    // Metric logging
    void log_metric(const std::string& test_name, const std::string& variant, double value);
    
    // Analysis
    TestResult analyze_test(const std::string& test_name) const;
    bool is_test_significant(const std::string& test_name) const;
    
    // Reporting
    void generate_test_report(const std::string& test_name, const std::string& output_path) const;

private:
    double calculate_statistical_significance(const std::vector<double>& variant_a,
                                            const std::vector<double>& variant_b) const;
};

// Model deployment utilities
class ModelDeployment {
public:
    enum class DeploymentStrategy {
        BlueGreen,
        Canary,
        RollingUpdate
    };
    
    struct DeploymentConfig {
        DeploymentStrategy strategy = DeploymentStrategy::BlueGreen;
        double canary_percentage = 0.1;  // For canary deployments
        std::chrono::minutes rollout_duration{30};  // For rolling updates
        bool auto_rollback = true;
        double error_threshold = 0.05;
    };
    
    struct DeploymentStatus {
        std::string deployment_id;
        std::string status;  // "deploying", "deployed", "failed", "rolled_back"
        double progress;     // 0.0 to 1.0
        std::chrono::time_point<std::chrono::system_clock> started_at;
        std::string error_message;
    };
    
private:
    std::unordered_map<std::string, DeploymentStatus> deployments_;
    std::mutex deployment_mutex_;
    
public:
    // Deployment operations
    std::string deploy_model(const std::string& model_name, const std::string& version,
                           const DeploymentConfig& config = {});
    
    void rollback_deployment(const std::string& deployment_id);
    
    // Status tracking
    DeploymentStatus get_deployment_status(const std::string& deployment_id) const;
    std::vector<DeploymentStatus> list_deployments() const;
    
    // Health monitoring during deployment
    void monitor_deployment_health(const std::string& deployment_id);

private:
    void execute_blue_green_deployment(const std::string& model_name, const std::string& version);
    void execute_canary_deployment(const std::string& model_name, const std::string& version, double percentage);
    void execute_rolling_update(const std::string& model_name, const std::string& version, std::chrono::minutes duration);
};

// Configuration management
class ConfigManager {
private:
    std::unordered_map<std::string, std::string> config_;
    std::mutex config_mutex_;
    std::string config_file_path_;
    
public:
    ConfigManager(const std::string& config_file = "clmodel.conf");
    
    // Configuration access
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const;
    
    void set(const std::string& key, const std::string& value);
    
    // File operations
    void load_from_file(const std::string& path);
    void save_to_file(const std::string& path) const;
    
    // Environment variables
    void load_from_env(const std::string& prefix = "ASEKIOML_");
    
    // Watch for changes
    void watch_file_changes(std::function<void()> callback);
};

// Distributed training coordinator
class DistributedTraining {
public:
    enum class Strategy {
        DataParallel,
        ModelParallel,
        PipelineParallel
    };
    
    struct DistributedConfig {
        Strategy strategy = Strategy::DataParallel;
        std::vector<std::string> worker_addresses;
        int rank = 0;  // Current worker rank
        int world_size = 1;  // Total number of workers
        std::string backend = "nccl";  // Communication backend
    };
    
private:
    DistributedConfig config_;
    std::vector<std::unique_ptr<NeuralNetwork>> worker_models_;
    
public:
    DistributedTraining(const DistributedConfig& config);
    
    // Distributed training operations
    void initialize_workers();
    void synchronize_models();
    void all_reduce_gradients();
    
    // Training coordination
    TrainingHistory distributed_fit(const Dataset& dataset, int epochs, int batch_size);
    
    // Fault tolerance
    void handle_worker_failure(int failed_rank);
    void checkpoint_training_state(const std::string& path);
    void restore_training_state(const std::string& path);
};

} // namespace production
} // namespace asekioml
