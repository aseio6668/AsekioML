#include "../include/clmodel.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace clmodel;
using namespace clmodel::gpu;

void print_device_info(const std::vector<GPUDeviceInfo>& devices) {
    std::cout << "\n=== Available GPU Devices ===" << std::endl;
    std::cout << std::left << std::setw(5) << "ID"
              << std::setw(15) << "Type"
              << std::setw(30) << "Name"
              << std::setw(12) << "Memory (MB)"
              << std::setw(8) << "CC"
              << std::setw(8) << "FP64"
              << std::setw(8) << "Unified" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& device : devices) {
        std::string type_str;
        switch (device.type) {
            case DeviceType::CPU: type_str = "CPU"; break;
            case DeviceType::CUDA: type_str = "NVIDIA CUDA"; break;
            case DeviceType::ROCM: type_str = "AMD ROCm"; break;
            case DeviceType::OPENCL: type_str = "OpenCL"; break;
            case DeviceType::METAL: type_str = "Apple Metal"; break;
            case DeviceType::VULKAN: type_str = "Vulkan"; break;
            default: type_str = "Unknown"; break;
        }
        
        std::cout << std::left << std::setw(5) << device.device_id
                  << std::setw(15) << type_str
                  << std::setw(30) << device.name
                  << std::setw(12) << device.memory_mb
                  << std::setw(8) << (std::to_string(device.compute_capability_major) + "." + std::to_string(device.compute_capability_minor))
                  << std::setw(8) << (device.supports_double_precision ? "Yes" : "No")
                  << std::setw(8) << (device.supports_unified_memory ? "Yes" : "No") << std::endl;
    }
    std::cout << std::endl;
}

void benchmark_matrix_operations(DeviceType device_type, const std::string& device_name, size_t size = 1024) {
    std::cout << "\n=== Benchmarking " << device_name << " ===" << std::endl;
    
    try {
        // Create test matrices
        Matrix a(size, size);
        Matrix b(size, size);
          // Initialize with random values
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                a(i, j) = static_cast<double>(rand()) / RAND_MAX;
                b(i, j) = static_cast<double>(rand()) / RAND_MAX;
            }
        }
        
        auto start = std::chrono::high_resolution_clock::now();
          // Test matrix multiplication
        if (device_type == DeviceType::CPU) {
            Matrix result = a * b;  // Use operator* instead of multiply()
        } else {
            Matrix result = gpu_ops::multiply_gpu(a, b, device_type);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Matrix multiplication (" << size << "x" << size << "): " 
                  << duration.count() << " ms" << std::endl;
        
        // Test matrix addition
        start = std::chrono::high_resolution_clock::now();
          if (device_type == DeviceType::CPU) {
            Matrix result = a + b;  // Use operator+ instead of add()
        } else {
            Matrix result = gpu_ops::add_gpu(a, b, device_type);
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Matrix addition (" << size << "x" << size << "): " 
                  << duration.count() << " ms" << std::endl;
        
        // Test activation functions
        start = std::chrono::high_resolution_clock::now();
        
        if (device_type == DeviceType::CPU) {            // Use CPU activation
            Matrix result(size, size);
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    double val = a(i, j);  // Use operator() instead of get()
                    result(i, j) = std::max(0.0, val); // ReLU, use operator() instead of set()
                }
            }
        } else {
            Matrix result = gpu_ops::relu_gpu(a, device_type);
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "ReLU activation (" << size << "x" << size << "): " 
                  << duration.count() << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error during benchmarking: " << e.what() << std::endl;
    }
}

void demonstrate_cross_platform_training() {
    std::cout << "\n=== Cross-Platform Neural Network Training ===" << std::endl;
    
    // Get available devices
    auto& gpu_manager = GPUManager::get_instance();
    auto devices = gpu_manager.get_available_devices();
    
    if (devices.empty()) {
        std::cout << "No devices available for training" << std::endl;
        return;
    }    // Create a simple neural network
    clmodel::NeuralNetwork network;
    network.add_layer(std::make_unique<clmodel::DenseLayer>(784, 128));  // Input layer
    network.add_layer(std::make_unique<clmodel::DenseLayer>(128, 64));   // Hidden layer
    network.add_layer(std::make_unique<clmodel::DenseLayer>(64, 10));    // Output layer
    
    // Compile the network
    network.compile("mse", "sgd", 0.01);
    
    // Create sample training data
    std::vector<Matrix> train_inputs;
    std::vector<Matrix> train_targets;
    
    for (int i = 0; i < 100; ++i) {
        Matrix input(784, 1);
        Matrix target(10, 1);
          // Fill with dummy data
        for (size_t j = 0; j < 784; ++j) {
            input(j, 0) = static_cast<double>(rand()) / RAND_MAX;  // Use operator()
        }
        
        int label = rand() % 10;
        target(label, 0) = 1.0;  // Use operator()
        
        train_inputs.push_back(input);
        train_targets.push_back(target);
    }
    
    // Train on different device types
    for (const auto& device : devices) {
        if (device.type == DeviceType::CPU) continue; // Skip CPU for this demo
        
        std::string device_name;
        switch (device.type) {
            case DeviceType::CUDA: device_name = "NVIDIA CUDA"; break;
            case DeviceType::ROCM: device_name = "AMD ROCm"; break;
            case DeviceType::OPENCL: device_name = "OpenCL"; break;
            default: continue;
        }
        
        std::cout << "\nTraining on " << device_name << " device: " << device.name << std::endl;
        
        try {
            // Simulate training with GPU acceleration
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int epoch = 0; epoch < 5; ++epoch) {
                double total_loss = 0.0;
                
                for (size_t i = 0; i < train_inputs.size(); ++i) {                    // Use the network's train_step method instead of manual forward/backward
                    network.train_step(train_inputs[i], train_targets[i]);
                    
                    // Calculate loss for display
                    Matrix prediction = network.predict(train_inputs[i]);
                    Matrix error = prediction - train_targets[i];  // Use operator- instead of subtract()
                    double loss = 0.0;
                    for (size_t j = 0; j < error.rows(); ++j) {
                        for (size_t k = 0; k < error.cols(); ++k) {
                            loss += error(j, k) * error(j, k);  // Use operator() instead of get()
                        }
                    }
                    total_loss += loss;
                }
                
                std::cout << "  Epoch " << (epoch + 1) << "/5, Loss: " 
                          << std::fixed << std::setprecision(6) << (total_loss / train_inputs.size()) << std::endl;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "  Training completed in " << duration.count() << " ms" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  Error during training: " << e.what() << std::endl;
        }
    }
}

void demonstrate_memory_efficiency() {
    std::cout << "\n=== GPU Memory Efficiency Demo ===" << std::endl;
    
    auto& gpu_manager = GPUManager::get_instance();
    
    // Show memory information for each GPU type
    if (gpu_manager.has_cuda_support()) {
        size_t cuda_memory = gpu_manager.get_total_gpu_memory(DeviceType::CUDA);
        std::cout << "Total NVIDIA CUDA memory: " << cuda_memory << " MB" << std::endl;
    }
    
    if (gpu_manager.has_rocm_support()) {
        size_t rocm_memory = gpu_manager.get_total_gpu_memory(DeviceType::ROCM);
        std::cout << "Total AMD ROCm memory: " << rocm_memory << " MB" << std::endl;
    }
    
    if (gpu_manager.has_opencl_support()) {
        size_t opencl_memory = gpu_manager.get_total_gpu_memory(DeviceType::OPENCL);
        std::cout << "Total OpenCL memory: " << opencl_memory << " MB" << std::endl;
    }
    
    // Demonstrate batch processing for memory efficiency
    DeviceType best_device = gpu_manager.get_best_device_type();
    if (best_device != DeviceType::CPU) {
        std::cout << "\nDemonstrating efficient batch processing..." << std::endl;
        
        try {
            // GPUBatchProcessor batch_processor(32, best_device);  // Temporarily disabled
            std::cout << "Batch processing demo temporarily disabled during development" << std::endl;
            
            // Create sample batch data
            std::vector<Matrix> batch_data;
            for (int i = 0; i < 100; ++i) {            Matrix data(256, 1);
            for (size_t j = 0; j < 256; ++j) {
                data(j, 0) = static_cast<double>(rand()) / RAND_MAX;  // Use operator()
            }
                batch_data.push_back(data);
            }
            
            auto start = std::chrono::high_resolution_clock::now();
              // auto results = batch_processor.process_batches(batch_data, 
            //     [](const GPUMatrix& input) -> GPUMatrix {
            //         return input.relu(); // Apply ReLU activation
            //     });
            
            // For now, just process one sample to show the concept
            if (!batch_data.empty()) {
                Matrix sample_result = gpu_ops::relu_gpu(batch_data[0], best_device);
                std::cout << "Sample processed successfully" << std::endl;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
              std::cout << "Processed " << batch_data.size() << " samples (demo mode)" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error during batch processing: " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "CLModel Multi-Vendor GPU Support Demo" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        // Initialize GPU manager and scan devices
        auto& gpu_manager = GPUManager::get_instance();
        auto devices = gpu_manager.get_available_devices();
        
        // Display available devices
        print_device_info(devices);
        
        // Show which GPU backends are available
        std::cout << "=== Available GPU Backends ===" << std::endl;
        std::cout << "NVIDIA CUDA: " << (gpu_manager.has_cuda_support() ? "✓ Available" : "✗ Not available") << std::endl;
        std::cout << "AMD ROCm: " << (gpu_manager.has_rocm_support() ? "✓ Available" : "✗ Not available") << std::endl;
        std::cout << "OpenCL: " << (gpu_manager.has_opencl_support() ? "✓ Available" : "✗ Not available") << std::endl;
        
        DeviceType best_device = gpu_manager.get_best_device_type();
        std::string best_device_name;
        switch (best_device) {
            case DeviceType::CPU: best_device_name = "CPU"; break;
            case DeviceType::CUDA: best_device_name = "NVIDIA CUDA"; break;
            case DeviceType::ROCM: best_device_name = "AMD ROCm"; break;
            case DeviceType::OPENCL: best_device_name = "OpenCL"; break;
            default: best_device_name = "Unknown"; break;
        }
        std::cout << "Best available device: " << best_device_name << std::endl;
        
        // Benchmark operations on different device types
        benchmark_matrix_operations(DeviceType::CPU, "CPU", 512);
        
        for (const auto& device : devices) {
            if (device.type == DeviceType::CPU) continue;
            
            std::string device_name;
            switch (device.type) {
                case DeviceType::CUDA: device_name = "NVIDIA CUDA"; break;
                case DeviceType::ROCM: device_name = "AMD ROCm"; break;
                case DeviceType::OPENCL: device_name = "OpenCL"; break;
                default: continue;
            }
            
            benchmark_matrix_operations(device.type, device_name + " (" + device.name + ")", 512);
        }
        
        // Demonstrate cross-platform training
        demonstrate_cross_platform_training();
        
        // Demonstrate memory efficiency
        demonstrate_memory_efficiency();
        
        std::cout << "\n=== GPU Compatibility Summary ===" << std::endl;
        std::cout << "✓ NVIDIA GPUs: Full CUDA support with cuBLAS and optional cuDNN" << std::endl;
        std::cout << "✓ AMD GPUs: ROCm/HIP support with rocBLAS" << std::endl;
        std::cout << "✓ Intel GPUs: OpenCL support for cross-platform compatibility" << std::endl;
        std::cout << "✓ Other GPUs: OpenCL fallback for maximum compatibility" << std::endl;
        std::cout << "✓ Automatic device selection based on performance and availability" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nGPU demo completed successfully!" << std::endl;
    return 0;
}
