#pragma once

#include <thread>
#include <future>
#include <queue>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <type_traits>
#include <memory>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <type_traits>

namespace clmodel {
namespace threading {

// Thread pool for parallel inference (no GIL limitations!)
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_tasks_;
    std::atomic<size_t> completed_tasks_;
    size_t max_queue_size_;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point start_time_;
    std::atomic<double> total_execution_time_;
    
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency(), size_t max_queue_size = 1000)
        : stop_(false), active_tasks_(0), completed_tasks_(0), max_queue_size_(max_queue_size),
          start_time_(std::chrono::high_resolution_clock::now()), total_execution_time_(0.0) {
        
        std::cout << "Creating ThreadPool with " << num_threads << " threads" << std::endl;
        
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }
    
    ~ThreadPool() {
        shutdown();
    }    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            if (stop_) {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool");
            }
            
            if (tasks_.size() >= max_queue_size_) {
                throw std::runtime_error("ThreadPool queue is full");
            }
            
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }
    
    // Parallel inference for batches
    template<typename InputType, typename OutputType, typename ModelType>
    std::vector<OutputType> parallel_inference(const std::vector<InputType>& inputs, 
                                              ModelType& model) {
        std::vector<std::future<OutputType>> futures;
        futures.reserve(inputs.size());
        
        // Submit all inference tasks
        for (const auto& input : inputs) {
            futures.push_back(submit([&model, input]() {
                return model.predict(input);
            }));
        }
        
        // Collect results
        std::vector<OutputType> results;
        results.reserve(inputs.size());
        
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        return results;
    }
    
    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        
        condition_.notify_all();
        
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        workers_.clear();
    }
    
    // Performance monitoring
    struct ThreadPoolStats {
        size_t num_threads;
        size_t active_tasks;
        size_t completed_tasks;
        size_t queue_size;
        double average_execution_time_ms;
        double throughput_tasks_per_second;
        double cpu_utilization;
    };
      ThreadPoolStats get_stats() const {
        ThreadPoolStats stats;
        stats.num_threads = workers_.size();
        stats.active_tasks = active_tasks_.load();
        stats.completed_tasks = completed_tasks_.load();
        
        {
            std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queue_mutex_));
            stats.queue_size = tasks_.size();
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
          stats.average_execution_time_ms = completed_tasks_ > 0 ? 
            (total_execution_time_.load() / completed_tasks_) : 0.0;
        stats.throughput_tasks_per_second = elapsed > 0 ? (completed_tasks_ / elapsed) : 0.0;
        stats.cpu_utilization = (double)active_tasks_ / workers_.size() * 100.0;
        
        return stats;
    }
    
    void print_stats() const {
        auto stats = get_stats();
        std::cout << "=== ThreadPool Statistics ===" << std::endl;
        std::cout << "Threads: " << stats.num_threads << std::endl;
        std::cout << "Active tasks: " << stats.active_tasks << std::endl;
        std::cout << "Completed tasks: " << stats.completed_tasks << std::endl;
        std::cout << "Queue size: " << stats.queue_size << std::endl;
        std::cout << "Average execution time: " << std::fixed << std::setprecision(3) 
                  << stats.average_execution_time_ms << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
                  << stats.throughput_tasks_per_second << " tasks/sec" << std::endl;
        std::cout << "CPU utilization: " << std::fixed << std::setprecision(1) 
                  << stats.cpu_utilization << "%" << std::endl;
    }

private:
    void worker_loop(size_t worker_id) {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                
                if (stop_ && tasks_.empty()) {
                    break;
                }
                
                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
            }
            
            if (task) {
                active_tasks_++;
                auto start = std::chrono::high_resolution_clock::now();
                
                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "Exception in worker " << worker_id << ": " << e.what() << std::endl;
                }
                  auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double, std::milli>(end - start).count();
                
                // Use proper atomic operations for floating point
                double current_total = total_execution_time_.load();
                while (!total_execution_time_.compare_exchange_weak(current_total, current_total + duration)) {
                    // Retry if another thread modified total_execution_time_
                }
                
                active_tasks_--;
                completed_tasks_++;
            }
        }
    }
};

// Async GPU scheduler (no GIL blocking!)
class GPUScheduler {
private:
    struct GPUTask {
        std::function<void()> task;
        int device_id;
        std::chrono::high_resolution_clock::time_point submit_time;
        std::string name;
    };
      std::vector<std::queue<GPUTask>> device_queues_;
    std::vector<std::thread> gpu_workers_;
    std::vector<std::unique_ptr<std::mutex>> device_mutexes_;
    std::vector<std::unique_ptr<std::condition_variable>> device_conditions_;
    std::atomic<bool> stop_;
    std::atomic<size_t> total_gpu_tasks_;
    
public:    GPUScheduler(int num_devices = 1) 
        : stop_(false), total_gpu_tasks_(0) {
        device_queues_.resize(num_devices);
        
        for (int i = 0; i < num_devices; ++i) {
            device_mutexes_.push_back(std::make_unique<std::mutex>());
            device_conditions_.push_back(std::make_unique<std::condition_variable>());
        }
        
        for (int i = 0; i < num_devices; ++i) {
            gpu_workers_.emplace_back([this, i] { gpu_worker_loop(i); });
        }
    }
    
    ~GPUScheduler() {
        shutdown();
    }
      template<typename F>
    auto schedule_gpu_task(F&& task, int device_id = 0, const std::string& name = "gpu_task") 
        -> std::future<std::invoke_result_t<F>> {
        using ReturnType = std::invoke_result_t<F>;
        auto promise = std::make_shared<std::promise<ReturnType>>();
        auto future = promise->get_future();
        
        GPUTask gpu_task;
        gpu_task.task = [promise, task = std::forward<F>(task)]() {
            try {
                if constexpr (std::is_void_v<ReturnType>) {
                    task();
                    promise->set_value();
                } else {
                    auto result = task();
                    promise->set_value(std::move(result));
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        };
        gpu_task.device_id = device_id;
        gpu_task.submit_time = std::chrono::high_resolution_clock::now();
        gpu_task.name = name;
          {
            std::lock_guard<std::mutex> lock(*device_mutexes_[device_id]);
            device_queues_[device_id].push(std::move(gpu_task));
        }
        
        device_conditions_[device_id]->notify_one();
        total_gpu_tasks_++;
        
        return future;
    }
      void shutdown() {
        stop_ = true;
        
        for (auto& condition : device_conditions_) {
            condition->notify_all();
        }
        
        for (auto& worker : gpu_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:    void gpu_worker_loop(int device_id) {
        while (true) {
            GPUTask task;
            
            {
                std::unique_lock<std::mutex> lock(*device_mutexes_[device_id]);
                device_conditions_[device_id]->wait(lock, [this, device_id] {
                    return stop_ || !device_queues_[device_id].empty();
                });
                
                if (stop_ && device_queues_[device_id].empty()) {
                    break;
                }
                
                if (!device_queues_[device_id].empty()) {
                    task = std::move(device_queues_[device_id].front());
                    device_queues_[device_id].pop();
                }
            }
            
            if (task.task) {
                try {
                    // Set GPU device context here if needed
                    task.task();
                } catch (const std::exception& e) {
                    std::cerr << "Exception in GPU worker " << device_id << ": " << e.what() << std::endl;
                }
            }
        }
    }
};

// Real-time inference server for chatbots/agents
class InferenceServer {
private:
    ThreadPool thread_pool_;
    GPUScheduler gpu_scheduler_;
    std::atomic<bool> running_;
    std::atomic<size_t> total_requests_;
    std::atomic<double> total_latency_ms_;
    
public:
    InferenceServer(size_t num_cpu_threads = std::thread::hardware_concurrency(), 
                   int num_gpu_devices = 1)
        : thread_pool_(num_cpu_threads), gpu_scheduler_(num_gpu_devices),
          running_(true), total_requests_(0), total_latency_ms_(0.0) {}
      template<typename ModelType, typename InputType, typename OutputType>
    std::future<OutputType> async_inference(ModelType& model, const InputType& input, 
                                           bool use_gpu = false) {
        total_requests_++;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (use_gpu) {
            return gpu_scheduler_.schedule_gpu_task([&model, input, this, start_time]() -> OutputType {
                OutputType result;
                if constexpr (std::is_invocable_v<ModelType, InputType>) {
                    // Handle function objects and lambdas
                    result = model(input);
                } else {
                    // Handle objects with predict method
                    result = model.predict(input);
                }
                update_latency(start_time);
                return result;
            });
        } else {
            return thread_pool_.submit([&model, input, this, start_time]() -> OutputType {
                OutputType result;
                if constexpr (std::is_invocable_v<ModelType, InputType>) {
                    // Handle function objects and lambdas
                    result = model(input);
                } else {
                    // Handle objects with predict method
                    result = model.predict(input);
                }
                update_latency(start_time);
                return result;
            });
        }
    }
    
    struct ServerStats {
        size_t total_requests;
        double average_latency_ms;
        double requests_per_second;
        ThreadPool::ThreadPoolStats thread_stats;
    };
    
    ServerStats get_stats() const {
        ServerStats stats;
        stats.total_requests = total_requests_.load();
        stats.average_latency_ms = stats.total_requests > 0 ? 
            (total_latency_ms_.load() / stats.total_requests) : 0.0;
        stats.thread_stats = thread_pool_.get_stats();
        stats.requests_per_second = stats.thread_stats.throughput_tasks_per_second;
        
        return stats;
    }
    
    void print_server_stats() const {
        auto stats = get_stats();
        std::cout << "=== Inference Server Statistics ===" << std::endl;        std::cout << "Total requests: " << stats.total_requests << std::endl;
        std::cout << "Average latency: " << std::fixed << std::setprecision(3) 
                  << stats.average_latency_ms << " ms" << std::endl;
        std::cout << "Requests per second: " << std::fixed << std::setprecision(2) 
                  << stats.requests_per_second << std::endl;
        std::cout << std::endl;
        thread_pool_.print_stats();
    }

private:    void update_latency(std::chrono::high_resolution_clock::time_point start_time) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // Use proper atomic operations for floating point
        double current_total = total_latency_ms_.load();
        while (!total_latency_ms_.compare_exchange_weak(current_total, current_total + latency)) {
            // Retry if another thread modified total_latency_ms_
        }
    }
};

} // namespace threading
} // namespace clmodel
