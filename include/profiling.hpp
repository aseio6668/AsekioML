#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

namespace clmodel {
namespace profiling {

// High-resolution timer for precise measurements
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }
    
    void reset() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
};

// Detailed operation metrics
struct OperationMetrics {
    std::string name;
    double total_time_ms;
    size_t call_count;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    size_t memory_allocated;
    size_t memory_peak;
    
    OperationMetrics() : total_time_ms(0), call_count(0), avg_time_ms(0), 
                        min_time_ms(std::numeric_limits<double>::max()),
                        max_time_ms(0), memory_allocated(0), memory_peak(0) {}
};

// System-wide profiler for transparency
class Profiler {
private:
    std::map<std::string, OperationMetrics> metrics_;
    std::map<std::string, Timer> active_timers_;
    mutable std::mutex mutex_;  // Make mutex mutable for const methods
    bool enabled_;
    
    static std::unique_ptr<Profiler> instance_;
    
public:
    static Profiler& get_instance() {
        if (!instance_) {
            instance_ = std::make_unique<Profiler>();
        }
        return *instance_;
    }
    
    Profiler() : enabled_(true) {}
    
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }
    
    // Start timing an operation
    void start_operation(const std::string& name) {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        active_timers_[name] = Timer();
    }
    
    // End timing an operation
    void end_operation(const std::string& name, size_t memory_used = 0) {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto timer_it = active_timers_.find(name);
        if (timer_it == active_timers_.end()) {
            std::cerr << "Warning: Ending operation '" << name << "' that wasn't started" << std::endl;
            return;
        }
        
        double elapsed = timer_it->second.elapsed_ms();
        active_timers_.erase(timer_it);
        
        auto& metrics = metrics_[name];
        metrics.name = name;
        metrics.total_time_ms += elapsed;
        metrics.call_count++;
        metrics.avg_time_ms = metrics.total_time_ms / metrics.call_count;
        metrics.min_time_ms = std::min(metrics.min_time_ms, elapsed);
        metrics.max_time_ms = std::max(metrics.max_time_ms, elapsed);
        metrics.memory_allocated += memory_used;
        metrics.memory_peak = std::max(metrics.memory_peak, memory_used);
    }
    
    // Get metrics for a specific operation
    OperationMetrics get_metrics(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            return it->second;
        }
        return OperationMetrics();
    }
    
    // Get all metrics
    std::map<std::string, OperationMetrics> get_all_metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return metrics_;
    }
    
    // Generate detailed report
    void print_report() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n=== CLModel Performance Report ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Operation"
                  << std::setw(10) << "Calls"
                  << std::setw(12) << "Total (ms)"
                  << std::setw(12) << "Avg (ms)"
                  << std::setw(12) << "Min (ms)"
                  << std::setw(12) << "Max (ms)"
                  << std::setw(15) << "Memory (KB)" << std::endl;
        std::cout << std::string(110, '-') << std::endl;
        
        // Sort by total time
        std::vector<std::pair<std::string, OperationMetrics>> sorted_metrics;
        for (const auto& pair : metrics_) {
            sorted_metrics.push_back(pair);
        }
        
        std::sort(sorted_metrics.begin(), sorted_metrics.end(),
                 [](const auto& a, const auto& b) {
                     return a.second.total_time_ms > b.second.total_time_ms;
                 });
        
        for (const auto& pair : sorted_metrics) {
            const auto& metrics = pair.second;
            std::cout << std::left << std::setw(25) << metrics.name
                      << std::setw(10) << metrics.call_count
                      << std::setw(12) << std::fixed << std::setprecision(3) << metrics.total_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << metrics.avg_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << metrics.min_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << metrics.max_time_ms
                      << std::setw(15) << (metrics.memory_peak / 1024) << std::endl;
        }
        
        // Summary statistics
        double total_time = 0;
        size_t total_calls = 0;
        for (const auto& pair : metrics_) {
            total_time += pair.second.total_time_ms;
            total_calls += pair.second.call_count;
        }
        
        std::cout << std::string(110, '-') << std::endl;
        std::cout << "Total operations: " << metrics_.size() 
                  << ", Total calls: " << total_calls
                  << ", Total time: " << std::fixed << std::setprecision(3) << total_time << " ms" << std::endl;
    }
    
    // Clear all metrics
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.clear();
        active_timers_.clear();
    }
    
    // Get summary statistics
    struct Summary {
        size_t total_operations;
        size_t total_calls;
        double total_time_ms;
        std::string hottest_operation;
        double hottest_operation_time;
    };
    
    Summary get_summary() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        Summary summary;
        summary.total_operations = metrics_.size();
        summary.total_calls = 0;
        summary.total_time_ms = 0;
        summary.hottest_operation_time = 0;
        
        for (const auto& pair : metrics_) {
            const auto& metrics = pair.second;
            summary.total_calls += metrics.call_count;
            summary.total_time_ms += metrics.total_time_ms;
            
            if (metrics.total_time_ms > summary.hottest_operation_time) {
                summary.hottest_operation = metrics.name;
                summary.hottest_operation_time = metrics.total_time_ms;
            }
        }
        
        return summary;
    }
};

// RAII profiling scope for automatic timing
class ProfileScope {
private:
    std::string operation_name_;
    size_t memory_at_start_;
    
public:
    ProfileScope(const std::string& name) : operation_name_(name) {
        memory_at_start_ = 0; // TODO: Get actual memory usage
        Profiler::get_instance().start_operation(operation_name_);
    }
    
    ~ProfileScope() {
        size_t memory_used = 0; // TODO: Calculate memory diff
        Profiler::get_instance().end_operation(operation_name_, memory_used);
    }
};

// Convenience macro for profiling
#define CLMODEL_PROFILE(name) clmodel::profiling::ProfileScope _prof_scope(name)

} // namespace profiling
} // namespace clmodel
