#include "ai/dynamic_model_dispatcher.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace clmodel {
namespace ai {

DynamicModelDispatcher::DynamicModelDispatcher(const LoadBalancingConfig& config)
    : config_(config), routing_strategy_(RoutingStrategy::ADAPTIVE) {
    
    std::cout << "DynamicModelDispatcher: Initializing with max load " 
              << config_.max_model_load << ", " << config_.max_concurrent_tasks 
              << " max tasks per model" << std::endl;
}

DynamicModelDispatcher::~DynamicModelDispatcher() {
    shutdown();
}

bool DynamicModelDispatcher::initialize() {
    std::cout << "DynamicModelDispatcher: Starting initialization..." << std::endl;
    
    if (is_running_.load()) {
        std::cout << "DynamicModelDispatcher: Already running" << std::endl;
        return true;
    }
    
    try {
        // Start monitoring threads
        is_running_.store(true);
        
        // For demo purposes, disable background threads to avoid stability issues
        // load_balancer_thread_ = std::thread(&DynamicModelDispatcher::loadBalancerLoop, this);
        // performance_monitor_thread_ = std::thread(&DynamicModelDispatcher::performanceMonitorLoop, this);
        
        std::cout << "DynamicModelDispatcher: Initialization complete (background threads disabled for demo)" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "DynamicModelDispatcher: Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void DynamicModelDispatcher::shutdown() {
    if (!is_running_.load()) {
        return;
    }
    
    std::cout << "DynamicModelDispatcher: Shutting down..." << std::endl;
    
    // Signal shutdown
    is_running_.store(false);
    
    // Wait for threads (disabled for demo)
    // if (load_balancer_thread_.joinable()) {
    //     load_balancer_thread_.join();
    // }
    // if (performance_monitor_thread_.joinable()) {
    //     performance_monitor_thread_.join();
    // }
    
    std::cout << "DynamicModelDispatcher: Shutdown complete" << std::endl;
}

DynamicModelDispatcher::DispatchResult DynamicModelDispatcher::dispatchTask(
    const OrchestralAIDirector::OrchestralTask& task) {
    
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    DispatchResult result;
    
    // Analyze content
    ContentAnalysis analysis = analyzeContent(task.input_content);
    
    // Get compatible models
    auto compatible_models = getCompatibleModels(task);
    if (compatible_models.empty()) {
        result.success = false;
        result.reason = "No compatible models available";
        return result;
    }
    
    // Select optimal model
    std::string selected_model = selectByStrategy(compatible_models, task, analysis);
    
    if (selected_model.empty()) {
        result.success = false;
        result.reason = "No suitable model found";
        return result;
    }
    
    // Check if model can accept task
    if (!canAcceptTask(selected_model)) {
        result.success = false;
        result.reason = "Selected model is overloaded";
        return result;
    }
    
    // Update model load
    auto& performance = model_performance_[selected_model];
    performance.active_tasks++;
    performance.current_load = std::min(1.0, 
        static_cast<double>(performance.active_tasks) / config_.max_concurrent_tasks);
    
    // Prepare result
    result.selected_model_id = selected_model;
    result.success = true;
    result.confidence_score = calculateModelScore(selected_model, task, analysis);
    result.estimated_processing_time_ms = performance.avg_processing_time_ms;
    result.estimated_quality = performance.quality_score;
    result.reason = "Optimal model selected based on " + 
                   std::to_string(static_cast<int>(routing_strategy_)) + " strategy";
    
    // Update dispatch statistics
    dispatch_counts_[selected_model]++;
    
    std::cout << "DynamicModelDispatcher: Dispatched task to " << selected_model 
              << " (confidence: " << std::fixed << std::setprecision(2) 
              << result.confidence_score << ")" << std::endl;
    
    return result;
}

DynamicModelDispatcher::ContentAnalysis DynamicModelDispatcher::analyzeContent(
    const MultiModalContent& content) {
    
    ContentAnalysis analysis;
    
    // Detect modalities
    if (content.has_video()) {
        analysis.detected_modalities.push_back("video");
        analysis.complexity_scores["video"] = calculateContentComplexity(content) * 1.2; // Video is complex
    }
    if (content.has_audio()) {
        analysis.detected_modalities.push_back("audio");
        analysis.complexity_scores["audio"] = calculateContentComplexity(content) * 0.8; // Audio is simpler
    }
    if (content.has_text()) {
        analysis.detected_modalities.push_back("text");
        analysis.complexity_scores["text"] = calculateContentComplexity(content) * 0.6; // Text is simplest
    }
    
    // Calculate overall complexity
    if (!analysis.complexity_scores.empty()) {
        double total_complexity = 0.0;
        for (const auto& pair : analysis.complexity_scores) {
            total_complexity += pair.second;
        }
        analysis.overall_complexity = total_complexity / analysis.complexity_scores.size();
    }
    
    // Set quality requirements based on complexity
    for (const auto& modality : analysis.detected_modalities) {
        analysis.quality_requirements[modality] = 0.7 + (analysis.overall_complexity * 0.2);
    }
    
    // Processing priority based on modality count and complexity
    analysis.processing_priority = 0.5 + (analysis.detected_modalities.size() * 0.15) + 
                                  (analysis.overall_complexity * 0.2);
    analysis.processing_priority = std::min(1.0, analysis.processing_priority);
    
    return analysis;
}

std::string DynamicModelDispatcher::selectOptimalModel(
    const OrchestralAIDirector::OrchestralTask& task, 
    const ContentAnalysis& analysis) {
    
    auto compatible_models = getCompatibleModels(task);
    return selectByStrategy(compatible_models, task, analysis);
}

bool DynamicModelDispatcher::registerModel(const OrchestralAIDirector::ModelInfo& model_info) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    registered_models_[model_info.model_id] = model_info;
    
    // Initialize performance metrics
    ModelPerformance& performance = model_performance_[model_info.model_id];
    performance.model_id = model_info.model_id;
    performance.avg_processing_time_ms = model_info.avg_processing_time_ms;
    performance.quality_score = model_info.quality_score;
    performance.last_update = std::chrono::steady_clock::now();
    
    std::cout << "DynamicModelDispatcher: Registered model " << model_info.model_id 
              << " (type: " << model_info.model_type << ")" << std::endl;
    
    return true;
}

bool DynamicModelDispatcher::unregisterModel(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    auto model_it = registered_models_.find(model_id);
    auto perf_it = model_performance_.find(model_id);
    
    if (model_it != registered_models_.end()) {
        registered_models_.erase(model_it);
    }
    if (perf_it != model_performance_.end()) {
        model_performance_.erase(perf_it);
    }
    
    std::cout << "DynamicModelDispatcher: Unregistered model " << model_id << std::endl;
    return true;
}

void DynamicModelDispatcher::updateModelPerformance(const std::string& model_id, 
                                                   double processing_time_ms, 
                                                   double quality_score, 
                                                   bool task_success) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    auto it = model_performance_.find(model_id);
    if (it != model_performance_.end()) {
        ModelPerformance& perf = it->second;
        
        // Update metrics with exponential moving average
        const double alpha = 0.2; // Smoothing factor
        perf.avg_processing_time_ms = (1.0 - alpha) * perf.avg_processing_time_ms + 
                                     alpha * processing_time_ms;
        perf.quality_score = (1.0 - alpha) * perf.quality_score + alpha * quality_score;
        
        // Update counters
        perf.completed_tasks++;
        if (perf.active_tasks > 0) {
            perf.active_tasks--;
        }
        
        // Update success rate
        perf.success_rate = (perf.success_rate * (perf.completed_tasks - 1) + 
                           (task_success ? 1.0 : 0.0)) / perf.completed_tasks;
        
        // Update load
        perf.current_load = std::min(1.0, 
            static_cast<double>(perf.active_tasks) / config_.max_concurrent_tasks);
        
        perf.last_update = std::chrono::steady_clock::now();
        
        // Update statistics
        updateDispatchStatistics(model_id, processing_time_ms, quality_score);
    }
}

DynamicModelDispatcher::ModelPerformance DynamicModelDispatcher::getModelPerformance(
    const std::string& model_id) const {
    
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    auto it = model_performance_.find(model_id);
    if (it != model_performance_.end()) {
        return it->second;
    }
    
    return ModelPerformance{};
}

std::vector<std::string> DynamicModelDispatcher::getRegisteredModels() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::vector<std::string> models;
    for (const auto& pair : registered_models_) {
        models.push_back(pair.first);
    }
    
    return models;
}

bool DynamicModelDispatcher::canAcceptTask(const std::string& model_id) const {
    auto it = model_performance_.find(model_id);
    if (it == model_performance_.end()) {
        return false;
    }
    
    const ModelPerformance& perf = it->second;
    return perf.current_load < config_.max_model_load && 
           perf.active_tasks < config_.max_concurrent_tasks;
}

double DynamicModelDispatcher::getSystemLoad() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (model_performance_.empty()) {
        return 0.0;
    }
    
    double total_load = 0.0;
    for (const auto& pair : model_performance_) {
        total_load += pair.second.current_load;
    }
    
    return total_load / model_performance_.size();
}

void DynamicModelDispatcher::balanceLoad() {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    double system_load = getSystemLoad();
    if (system_load > config_.load_balance_threshold) {
        std::cout << "DynamicModelDispatcher: Triggering load balancing (system load: " 
                  << std::fixed << std::setprecision(2) << system_load << ")" << std::endl;
        
        // Simple load balancing: log the event for now
        // In a full implementation, this would redistribute tasks
    }
}

void DynamicModelDispatcher::scaleResources() {
    if (!config_.enable_predictive_scaling) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    // Simple predictive scaling logic
    double avg_load = getSystemLoad();
    if (avg_load > 0.8) {
        std::cout << "DynamicModelDispatcher: High load detected (" 
                  << std::fixed << std::setprecision(2) << avg_load 
                  << "), scaling may be needed" << std::endl;
    }
}

void DynamicModelDispatcher::setRoutingStrategy(RoutingStrategy strategy) {
    routing_strategy_ = strategy;
    std::cout << "DynamicModelDispatcher: Routing strategy set to " 
              << static_cast<int>(strategy) << std::endl;
}

DynamicModelDispatcher::RoutingStrategy DynamicModelDispatcher::getRoutingStrategy() const {
    return routing_strategy_;
}

void DynamicModelDispatcher::updateConfig(const LoadBalancingConfig& config) {
    config_ = config;
    std::cout << "DynamicModelDispatcher: Configuration updated" << std::endl;
}

DynamicModelDispatcher::LoadBalancingConfig DynamicModelDispatcher::getConfig() const {
    return config_;
}

std::map<std::string, size_t> DynamicModelDispatcher::getDispatchStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return dispatch_counts_;
}

std::map<std::string, double> DynamicModelDispatcher::getPerformanceAnalytics() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::map<std::string, double> analytics;
    for (const auto& pair : model_performance_) {
        analytics[pair.first + "_avg_time"] = pair.second.avg_processing_time_ms;
        analytics[pair.first + "_quality"] = pair.second.quality_score;
        analytics[pair.first + "_load"] = pair.second.current_load;
        analytics[pair.first + "_success_rate"] = pair.second.success_rate;
    }
    
    return analytics;
}

std::map<std::string, double> DynamicModelDispatcher::getLoadBalancingMetrics() const {
    std::map<std::string, double> metrics;
    metrics["system_load"] = getSystemLoad();
    metrics["registered_models"] = static_cast<double>(registered_models_.size());
    metrics["load_balance_threshold"] = config_.load_balance_threshold;
    metrics["max_model_load"] = config_.max_model_load;
    
    return metrics;
}

std::string DynamicModelDispatcher::generatePerformanceReport() const {
    std::ostringstream report;
    report << "=== Dynamic Model Dispatcher Performance Report ===\n";
    report << "System Load: " << std::fixed << std::setprecision(2) << getSystemLoad() << "\n";
    report << "Registered Models: " << registered_models_.size() << "\n";
    report << "Routing Strategy: " << static_cast<int>(routing_strategy_) << "\n\n";
    
    report << "Model Performance:\n";
    for (const auto& pair : model_performance_) {
        const auto& perf = pair.second;
        report << "  " << pair.first << ":\n";
        report << "    Avg Time: " << std::fixed << std::setprecision(1) 
               << perf.avg_processing_time_ms << "ms\n";
        report << "    Quality: " << std::fixed << std::setprecision(3) 
               << perf.quality_score << "\n";
        report << "    Load: " << std::fixed << std::setprecision(2) 
               << perf.current_load << "\n";
        report << "    Success Rate: " << std::fixed << std::setprecision(2) 
               << perf.success_rate << "\n";
    }
    
    return report.str();
}

// Private methods

void DynamicModelDispatcher::loadBalancerLoop() {
    while (is_running_.load()) {
        balanceLoad();
        scaleResources();
        
        std::this_thread::sleep_for(config_.update_interval);
    }
}

void DynamicModelDispatcher::performanceMonitorLoop() {
    while (is_running_.load()) {
        // Update performance metrics
        std::lock_guard<std::mutex> lock(models_mutex_);
        
        for (auto& pair : model_performance_) {
            ModelPerformance& perf = pair.second;
            
            // Simulate resource usage updates
            perf.cpu_usage = 0.3 + (perf.current_load * 0.4);
            perf.memory_usage = 0.2 + (perf.current_load * 0.3);
            perf.gpu_usage = 0.1 + (perf.current_load * 0.5);
        }
        
        std::this_thread::sleep_for(config_.update_interval);
    }
}

double DynamicModelDispatcher::calculateModelScore(const std::string& model_id, 
                                                  const OrchestralAIDirector::OrchestralTask& task,
                                                  const ContentAnalysis& analysis) const {
    
    auto it = model_performance_.find(model_id);
    if (it == model_performance_.end()) {
        return 0.0;
    }
    
    const ModelPerformance& perf = it->second;
    
    // Weighted scoring based on multiple factors
    double quality_score = perf.quality_score * 0.3;
    double speed_score = (100.0 - std::min(100.0, perf.avg_processing_time_ms)) / 100.0 * 0.25;
    double load_score = (1.0 - perf.current_load) * 0.25;
    double success_score = perf.success_rate * 0.2;
    
    return quality_score + speed_score + load_score + success_score;
}

std::vector<std::string> DynamicModelDispatcher::getCompatibleModels(
    const OrchestralAIDirector::OrchestralTask& task) const {
    
    std::vector<std::string> compatible;
    
    for (const auto& pair : registered_models_) {
        const auto& model = pair.second;
        
        // Check if model supports required modalities
        bool is_compatible = true;
        for (const auto& required_modality : task.required_modalities) {
            bool supports_modality = false;
            for (const auto& supported_modality : model.supported_modalities) {
                if (supported_modality == required_modality) {
                    supports_modality = true;
                    break;
                }
            }
            if (!supports_modality) {
                is_compatible = false;
                break;
            }
        }
        
        if (is_compatible && model.is_available) {
            compatible.push_back(model.model_id);
        }
    }
    
    return compatible;
}

double DynamicModelDispatcher::calculateContentComplexity(const MultiModalContent& content) const {
    double complexity = 0.5; // Base complexity
    
    try {
        // Video complexity based on dimensions
        if (content.has_video()) {
            size_t video_size = content.video_features.size();
            complexity += std::min(0.3, static_cast<double>(video_size) / 100000.0); // Normalize by size
        }
        
        // Audio complexity
        if (content.has_audio()) {
            size_t audio_size = content.audio_features.size();
            complexity += std::min(0.2, static_cast<double>(audio_size) / 50000.0);
        }
        
        // Text complexity
        if (content.has_text()) {
            size_t text_size = content.text_features.size();
            complexity += std::min(0.1, static_cast<double>(text_size) / 10000.0);
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Error calculating content complexity: " << e.what() << std::endl;
        // Return safe default complexity
        return 0.5;
    }
    
    return std::min(1.0, complexity);
}

std::string DynamicModelDispatcher::selectByStrategy(const std::vector<std::string>& candidates,
                                                    const OrchestralAIDirector::OrchestralTask& task,
                                                    const ContentAnalysis& analysis) const {
    
    if (candidates.empty()) {
        return "";
    }
    
    switch (routing_strategy_) {
        case RoutingStrategy::QUALITY_OPTIMIZED: {
            std::string best_model = candidates[0];
            double best_quality = 0.0;
            
            for (const auto& model_id : candidates) {
                auto it = model_performance_.find(model_id);
                if (it != model_performance_.end() && it->second.quality_score > best_quality) {
                    best_quality = it->second.quality_score;
                    best_model = model_id;
                }
            }
            return best_model;
        }
        
        case RoutingStrategy::LATENCY_OPTIMIZED: {
            std::string fastest_model = candidates[0];
            double best_time = std::numeric_limits<double>::max();
            
            for (const auto& model_id : candidates) {
                auto it = model_performance_.find(model_id);
                if (it != model_performance_.end() && it->second.avg_processing_time_ms < best_time) {
                    best_time = it->second.avg_processing_time_ms;
                    fastest_model = model_id;
                }
            }
            return fastest_model;
        }
        
        case RoutingStrategy::LOAD_BALANCED: {
            std::string best_model = candidates[0];
            double lowest_load = 1.0;
            
            for (const auto& model_id : candidates) {
                auto it = model_performance_.find(model_id);
                if (it != model_performance_.end() && it->second.current_load < lowest_load) {
                    lowest_load = it->second.current_load;
                    best_model = model_id;
                }
            }
            return best_model;
        }
        
        case RoutingStrategy::ADAPTIVE: {
            // Use composite score
            std::string best_model = candidates[0];
            double best_score = -1.0;
            
            for (const auto& model_id : candidates) {
                double score = calculateModelScore(model_id, task, analysis);
                if (score > best_score) {
                    best_score = score;
                    best_model = model_id;
                }
            }
            return best_model;
        }
        
        case RoutingStrategy::ROUND_ROBIN: {
            // Simple round robin based on dispatch count
            std::string selected_model = candidates[0];
            size_t min_dispatches = std::numeric_limits<size_t>::max();
            
            for (const auto& model_id : candidates) {
                auto it = dispatch_counts_.find(model_id);
                size_t dispatches = (it != dispatch_counts_.end()) ? it->second : 0;
                if (dispatches < min_dispatches) {
                    min_dispatches = dispatches;
                    selected_model = model_id;
                }
            }
            return selected_model;
        }
        
        case RoutingStrategy::CONTENT_AWARE:
        default: {
            // Content-aware selection based on complexity and requirements
            return selectByStrategy(candidates, task, analysis); // Fallback to adaptive
        }
    }
}

void DynamicModelDispatcher::updateDispatchStatistics(const std::string& model_id, 
                                                     double processing_time, 
                                                     double quality) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    total_processing_times_[model_id] += processing_time;
    total_quality_scores_[model_id] += quality;
}

} // namespace ai
} // namespace clmodel
