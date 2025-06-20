#include "production_features.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <future>
#include <chrono>
#include <cmath>
#include <numeric>
#include <map>

namespace clmodel {
namespace production {

// ModelRegistry implementation
void ModelRegistry::register_model(const std::string& name, const std::string& version,
                                  std::unique_ptr<NeuralNetwork> model,
                                  double validation_score,
                                  const std::unordered_map<std::string, std::string>& metadata) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    ModelVersion model_version;
    model_version.name = name;
    model_version.version = version;
    model_version.model = std::move(model);
    model_version.created_at = std::chrono::system_clock::now();
    model_version.metadata = metadata;
    model_version.validation_score = validation_score;
    
    models_[name].push_back(std::move(model_version));
    
    // Sort by validation score (better scores first)
    std::sort(models_[name].begin(), models_[name].end(),
        [](const ModelVersion& a, const ModelVersion& b) {
            return a.validation_score < b.validation_score; // Assuming lower is better
        });
}

std::unique_ptr<NeuralNetwork> ModelRegistry::get_model(const std::string& name, 
                                                       const std::string& version) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        throw std::runtime_error("Model '" + name + "' not found");
    }
    
    if (version == "latest") {
        if (it->second.empty()) {
            throw std::runtime_error("No versions available for model '" + name + "'");
        }
        // Return a copy of the best model
        return std::make_unique<NeuralNetwork>(*it->second[0].model);
    }
    
    for (const auto& model_version : it->second) {
        if (model_version.version == version) {
            return std::make_unique<NeuralNetwork>(*model_version.model);
        }
    }
    
    throw std::runtime_error("Version '" + version + "' not found for model '" + name + "'");
}

std::vector<std::string> ModelRegistry::list_models() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<std::string> model_names;
    for (const auto& pair : models_) {
        model_names.push_back(pair.first);
    }
    return model_names;
}

std::vector<std::string> ModelRegistry::list_versions(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        return {};
    }
    
    std::vector<std::string> versions;
    for (const auto& model_version : it->second) {
        versions.push_back(model_version.version);
    }
    return versions;
}

void ModelRegistry::delete_model(const std::string& name, const std::string& version) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        return;
    }
    
    auto& versions = it->second;
    versions.erase(std::remove_if(versions.begin(), versions.end(),
        [&version](const ModelVersion& mv) {
            return mv.version == version;
        }), versions.end());
    
    if (versions.empty()) {
        models_.erase(it);
    }
}

std::vector<std::pair<std::string, double>> ModelRegistry::get_model_scores(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        return {};
    }
    
    std::vector<std::pair<std::string, double>> scores;
    for (const auto& model_version : it->second) {
        scores.emplace_back(model_version.version, model_version.validation_score);
    }
    return scores;
}

std::string ModelRegistry::get_best_version(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end() || it->second.empty()) {
        return "";
    }
    
    return it->second[0].version; // Already sorted by score
}

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry registry;
    return registry;
}

// InferenceServer implementation
InferenceServer::InferenceServer(std::unique_ptr<NeuralNetwork> model, const ServerConfig& config)
    : model_(std::move(model)), max_batch_size_(config.max_batch_size),
      num_workers_(config.num_workers), batch_timeout_(config.batch_timeout) {
}

InferenceServer::~InferenceServer() {
    stop();
}

void InferenceServer::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    
    // Start worker threads
    for (size_t i = 0; i < num_workers_; ++i) {
        worker_threads_.emplace_back([this]() { worker_loop(); });
    }
}

void InferenceServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    queue_cv_.notify_all();
    
    // Wait for all workers to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
}

Matrix InferenceServer::predict(const Matrix& input) {
    auto future = predict_async(input);
    return future.get();
}

std::future<Matrix> InferenceServer::predict_async(const Matrix& input) {
    PredictionRequest request;
    request.request_id = total_requests_.fetch_add(1);
    request.input = input;
    request.timestamp = std::chrono::high_resolution_clock::now();
    
    auto future = request.result.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(request));
    }
    
    queue_cv_.notify_one();
    return future;
}

std::vector<Matrix> InferenceServer::predict_batch(const std::vector<Matrix>& inputs) {
    std::vector<std::future<Matrix>> futures;
    futures.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        futures.push_back(predict_async(input));
    }
    
    std::vector<Matrix> results;
    results.reserve(inputs.size());
    
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

InferenceServer::PerformanceMetrics InferenceServer::get_metrics() const {
    PerformanceMetrics metrics;
    metrics.total_requests = total_requests_.load();
    metrics.successful_requests = successful_requests_.load();
    
    double total_time = total_inference_time_.load();
    if (metrics.successful_requests > 0) {
        metrics.average_latency_ms = total_time / metrics.successful_requests;
    }
    
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        if (!latency_history_.empty()) {
            auto sorted_latencies = latency_history_;
            std::sort(sorted_latencies.begin(), sorted_latencies.end());
            
            metrics.p95_latency_ms = calculate_percentile(sorted_latencies, 0.95);
            metrics.p99_latency_ms = calculate_percentile(sorted_latencies, 0.99);
        }
    }
    
    if (total_time > 0) {
        metrics.throughput_rps = metrics.successful_requests / (total_time / 1000.0);
    }
    
    if (metrics.total_requests > 0) {
        metrics.error_rate = 1.0 - (static_cast<double>(metrics.successful_requests) / metrics.total_requests);
    }
    
    return metrics;
}

void InferenceServer::reset_metrics() {
    total_requests_.store(0);
    successful_requests_.store(0);
    total_inference_time_.store(0.0);
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    latency_history_.clear();
}

bool InferenceServer::health_check() const {
    return running_.load() && model_ != nullptr;
}

void InferenceServer::worker_loop() {
    std::vector<PredictionRequest> batch;
    batch.reserve(max_batch_size_);
    
    while (running_.load()) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for requests or timeout
            queue_cv_.wait_for(lock, batch_timeout_, [this, &batch]() {
                return !running_.load() || !request_queue_.empty() || batch.size() >= max_batch_size_;
            });
            
            if (!running_.load()) {
                break;
            }
            
            // Collect batch
            while (!request_queue_.empty() && batch.size() < max_batch_size_) {
                batch.push_back(std::move(request_queue_.front()));
                request_queue_.pop();
            }
        }
        
        if (!batch.empty()) {
            process_batch(batch);
            batch.clear();
        }
    }
}

void InferenceServer::process_batch(std::vector<PredictionRequest>& batch) {
    for (auto& request : batch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Perform inference
            Matrix result = model_->predict(request.input);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double latency_ms = duration.count() / 1000.0;
              // Update metrics
            successful_requests_.fetch_add(1);
            // For atomic double, we need to use compare_exchange or store/load pattern
            double current_time = total_inference_time_.load();
            while (!total_inference_time_.compare_exchange_weak(current_time, current_time + latency_ms)) {
                // Retry if another thread modified the value
            }
            
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                latency_history_.push_back(latency_ms);
                
                // Keep only recent history (last 1000 requests)
                if (latency_history_.size() > 1000) {
                    latency_history_.erase(latency_history_.begin(), 
                                         latency_history_.begin() + (latency_history_.size() - 1000));
                }
            }
            
            // Return result
            request.result.set_value(std::move(result));
              } catch (const std::exception& /* e */) {
            // Return error
            request.result.set_exception(std::current_exception());
        }
    }
}

double InferenceServer::calculate_percentile(const std::vector<double>& values, double percentile) const {
    if (values.empty()) {
        return 0.0;
    }
    
    size_t index = static_cast<size_t>(percentile * (values.size() - 1));
    return values[index];
}

// ModelMonitor implementation
ModelMonitor::ModelMonitor(const std::string& model_name, const Dataset& reference_data,
                          double drift_threshold, double performance_threshold)
    : model_name_(model_name), reference_data_(reference_data),
      drift_threshold_(drift_threshold), performance_threshold_(performance_threshold) {
}

ModelMonitor::DataDriftMetrics ModelMonitor::detect_data_drift(const Dataset& new_data) const {
    DataDriftMetrics metrics;
    metrics.psi_score = calculate_psi(reference_data_.features(), new_data.features());
    metrics.wasserstein_distance = calculate_wasserstein_distance(reference_data_.features(), new_data.features());
    metrics.ks_statistic = calculate_ks_statistic(reference_data_.features(), new_data.features());
    metrics.chi_square_statistic = calculate_chi_square_statistic(reference_data_.features(), new_data.features());
    metrics.drift_detected = metrics.psi_score > drift_threshold_;
    return metrics;
}

void ModelMonitor::log_performance(const PerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    performance_history_.push_back(metrics);
}

bool ModelMonitor::detect_performance_degradation() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    if (performance_history_.size() < 2) {
        return false;
    }
    
    const auto& latest = performance_history_.back();
    const auto& baseline = performance_history_[0];
    
    return (baseline.accuracy - latest.accuracy) > performance_threshold_;
}

void ModelMonitor::set_alert_callback(std::function<void(const std::string&)> callback) {
    alert_callback_ = callback;
}

void ModelMonitor::generate_monitoring_report(const std::string& output_path) const {
    // Stub implementation
    (void)output_path;
}

double ModelMonitor::calculate_psi(const Matrix& reference, const Matrix& current) const {
    // Calculate Population Stability Index
    // PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
    
    if (reference.rows() == 0 || current.rows() == 0 || 
        reference.cols() != current.cols()) {
        return 1.0; // High PSI indicates significant drift
    }
    
    double psi = 0.0;
    const size_t num_features = reference.cols();
    const size_t num_bins = 10; // Standard binning for PSI
    
    for (size_t feature = 0; feature < num_features; ++feature) {
        // Extract feature values
        std::vector<double> ref_values, cur_values;
        for (size_t i = 0; i < reference.rows(); ++i) {
            ref_values.push_back(reference[i][feature]);
        }
        for (size_t i = 0; i < current.rows(); ++i) {
            cur_values.push_back(current[i][feature]);
        }
        
        // Calculate percentiles for binning (use reference distribution)
        std::sort(ref_values.begin(), ref_values.end());
        std::vector<double> bin_edges;
        for (size_t i = 0; i <= num_bins; ++i) {
            double percentile = static_cast<double>(i) / num_bins;
            size_t index = static_cast<size_t>(percentile * (ref_values.size() - 1));
            bin_edges.push_back(ref_values[index]);
        }
        
        // Count occurrences in each bin for both distributions
        std::vector<size_t> ref_counts(num_bins, 0);
        std::vector<size_t> cur_counts(num_bins, 0);
        
        // Count reference distribution
        for (double val : ref_values) {
            for (size_t bin = 0; bin < num_bins; ++bin) {
                if (val <= bin_edges[bin + 1] || bin == num_bins - 1) {
                    ref_counts[bin]++;
                    break;
                }
            }
        }
        
        // Count current distribution
        std::sort(cur_values.begin(), cur_values.end());
        for (double val : cur_values) {
            for (size_t bin = 0; bin < num_bins; ++bin) {
                if (val <= bin_edges[bin + 1] || bin == num_bins - 1) {
                    cur_counts[bin]++;
                    break;
                }
            }
        }
        
        // Calculate PSI for this feature
        double feature_psi = 0.0;
        const double epsilon = 1e-10; // Prevent division by zero
        
        for (size_t bin = 0; bin < num_bins; ++bin) {
            double ref_pct = static_cast<double>(ref_counts[bin]) / ref_values.size();
            double cur_pct = static_cast<double>(cur_counts[bin]) / cur_values.size();
            
            // Add small epsilon to prevent log(0)
            ref_pct = std::max(ref_pct, epsilon);
            cur_pct = std::max(cur_pct, epsilon);
            
            feature_psi += (cur_pct - ref_pct) * std::log(cur_pct / ref_pct);
        }
        
        psi += feature_psi;
    }
    
    return psi / num_features; // Average PSI across features
}

double ModelMonitor::calculate_wasserstein_distance(const Matrix& reference, const Matrix& current) const {
    // Calculate 1-Wasserstein distance (Earth Mover's Distance)
    // For multivariate data, we compute the average distance across features
    
    if (reference.rows() == 0 || current.rows() == 0 || 
        reference.cols() != current.cols()) {
        return 1.0; // High distance indicates significant drift
    }
    
    double total_distance = 0.0;
    const size_t num_features = reference.cols();
    
    for (size_t feature = 0; feature < num_features; ++feature) {
        // Extract and sort feature values
        std::vector<double> ref_values, cur_values;
        for (size_t i = 0; i < reference.rows(); ++i) {
            ref_values.push_back(reference[i][feature]);
        }
        for (size_t i = 0; i < current.rows(); ++i) {
            cur_values.push_back(current[i][feature]);
        }
        
        std::sort(ref_values.begin(), ref_values.end());
        std::sort(cur_values.begin(), cur_values.end());
        
        // Calculate 1-Wasserstein distance for this feature
        // W1(P,Q) = integral |F_P^{-1}(u) - F_Q^{-1}(u)| du from 0 to 1
        // Approximated using quantiles
        
        double feature_distance = 0.0;
        const size_t num_quantiles = 100; // Use 100 quantiles for approximation
        
        for (size_t q = 1; q < num_quantiles; ++q) {
            double quantile = static_cast<double>(q) / num_quantiles;
            
            // Find quantile values (inverse CDF)
            size_t ref_idx = static_cast<size_t>(quantile * (ref_values.size() - 1));
            size_t cur_idx = static_cast<size_t>(quantile * (cur_values.size() - 1));
            
            double ref_quantile = ref_values[ref_idx];
            double cur_quantile = cur_values[cur_idx];
            
            feature_distance += std::abs(ref_quantile - cur_quantile);
        }
        
        feature_distance /= (num_quantiles - 1); // Average over quantiles
        total_distance += feature_distance;
    }
    
    return total_distance / num_features; // Average distance across features
}

double ModelMonitor::calculate_ks_statistic(const Matrix& reference, const Matrix& current) const {
    // Calculate Kolmogorov-Smirnov statistic
    // KS = max|F_ref(x) - F_cur(x)| where F(x) is the cumulative distribution function
    
    if (reference.rows() == 0 || current.rows() == 0 || 
        reference.cols() != current.cols()) {
        return 1.0; // High KS indicates significant difference
    }
    
    double max_ks = 0.0;
    const size_t num_features = reference.cols();
    
    for (size_t feature = 0; feature < num_features; ++feature) {
        // Extract feature values
        std::vector<double> ref_values, cur_values;
        for (size_t i = 0; i < reference.rows(); ++i) {
            ref_values.push_back(reference[i][feature]);
        }
        for (size_t i = 0; i < current.rows(); ++i) {
            cur_values.push_back(current[i][feature]);
        }
        
        // Sort both datasets
        std::sort(ref_values.begin(), ref_values.end());
        std::sort(cur_values.begin(), cur_values.end());
        
        // Calculate KS statistic for this feature
        double feature_ks = 0.0;
        size_t ref_idx = 0, cur_idx = 0;
        
        // Merge the two sorted arrays and compute CDFs
        while (ref_idx < ref_values.size() || cur_idx < cur_values.size()) {
            double ref_cdf = static_cast<double>(ref_idx) / ref_values.size();
            double cur_cdf = static_cast<double>(cur_idx) / cur_values.size();
            
            // Update KS statistic with maximum difference
            feature_ks = std::max(feature_ks, std::abs(ref_cdf - cur_cdf));
            
            // Advance the pointer with smaller value
            if (ref_idx >= ref_values.size()) {
                cur_idx++;
            } else if (cur_idx >= cur_values.size()) {
                ref_idx++;
            } else if (ref_values[ref_idx] <= cur_values[cur_idx]) {
                ref_idx++;
            } else {
                cur_idx++;
            }
        }
        
        max_ks = std::max(max_ks, feature_ks);
    }
    
    return max_ks; // Return maximum KS statistic across all features
}

double ModelMonitor::calculate_chi_square_statistic(const Matrix& reference, const Matrix& current) const {
    // Calculate Chi-square statistic for categorical data drift detection
    // Chi-square = sum((observed - expected)^2 / expected) for each category
    
    if (reference.rows() == 0 || current.rows() == 0 || 
        reference.cols() != current.cols()) {
        return 1000.0; // High Chi-square indicates significant difference
    }
    
    double total_chi_square = 0.0;
    const size_t num_features = reference.cols();
    const double min_expected = 1.0; // Minimum expected count to ensure reasonable Chi-square values
    
    for (size_t feature = 0; feature < num_features; ++feature) {
        // Extract feature values and discretize them into categories
        std::vector<double> ref_values, cur_values;
        for (size_t i = 0; i < reference.rows(); ++i) {
            ref_values.push_back(reference[i][feature]);
        }
        for (size_t i = 0; i < current.rows(); ++i) {
            cur_values.push_back(current[i][feature]);
        }
        
        // Find the range for binning
        auto ref_minmax = std::minmax_element(ref_values.begin(), ref_values.end());
        auto cur_minmax = std::minmax_element(cur_values.begin(), cur_values.end());
        
        double min_val = std::min(*ref_minmax.first, *cur_minmax.first);
        double max_val = std::max(*ref_minmax.second, *cur_minmax.second);
          // Create bins for categorical analysis (discretization)
        const size_t num_bins = 10;
        double bin_width = (max_val - min_val) / num_bins;
        if (bin_width < 1e-6) {
            // Handle case where all values are the same - return very low Chi-square
            continue; // Skip this feature as it has no variance
        }
        
        // Count observations in each bin
        std::vector<size_t> ref_counts(num_bins, 0);
        std::vector<size_t> cur_counts(num_bins, 0);
        
        for (double val : ref_values) {
            size_t bin = static_cast<size_t>((val - min_val) / bin_width);
            bin = std::min(bin, num_bins - 1); // Ensure we don't go out of bounds
            ref_counts[bin]++;
        }
        
        for (double val : cur_values) {
            size_t bin = static_cast<size_t>((val - min_val) / bin_width);
            bin = std::min(bin, num_bins - 1);
            cur_counts[bin]++;
        }
        
        // Calculate expected frequencies based on reference distribution
        double total_ref = static_cast<double>(ref_values.size());
        double total_cur = static_cast<double>(cur_values.size());
        
        double feature_chi_square = 0.0;
          for (size_t bin = 0; bin < num_bins; ++bin) {
            double expected_proportion = static_cast<double>(ref_counts[bin]) / total_ref;
            double expected_count = expected_proportion * total_cur;
            double observed_count = static_cast<double>(cur_counts[bin]);
            
            // Skip bins with very low expected counts to avoid numerical issues
            if (expected_count < min_expected) {
                continue;
            }
            
            // Chi-square contribution: (observed - expected)^2 / expected
            double diff = observed_count - expected_count;
            feature_chi_square += (diff * diff) / expected_count;
        }
        
        total_chi_square += feature_chi_square;
    }
    
    // Return average Chi-square statistic across features
    return total_chi_square / num_features;
}

} // namespace production
} // namespace clmodel
