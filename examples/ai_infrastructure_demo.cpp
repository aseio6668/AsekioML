/**
 * @file ai_infrastructure_demo.cpp
 * @brief Comprehensive demo testing all Phase 1 AI infrastructure
 * 
 * This demo exercises:
 * - Tensor operations and advanced features
 * - Memory manager with pooling and statistics  
 * - Parallel compute engine with profiling
 * - CNN layers (Conv2D, MaxPool2D, AvgPool2D, Flatten)
 * - Multi-head attention and transformer components
 * - Layer normalization and feed-forward networks
 * - Integration testing and performance benchmarks
 */

#include "../include/tensor.hpp"
#include "../include/ai/memory_manager.hpp"
#include "../include/ai/compute_engine.hpp"
#include "../include/ai/cnn_layers.hpp"
#include "../include/ai/attention_layers.hpp"
#include "../include/network.hpp"
#include "../include/layer.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace clmodel;
using namespace clmodel::ai;

// Utility functions for timing and formatting
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_result(const std::string& test, bool passed, double time_ms = -1) {
    std::cout << "  " << std::setw(35) << std::left << test << ": ";
    std::cout << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    if (time_ms >= 0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << time_ms << " ms)";
    }
    std::cout << "\n";
}

// Test functions for each component
bool test_memory_manager() {
    try {
        clmodel::ai::AIMemoryManager& manager = clmodel::ai::AIMemoryManager::instance();
        
        // Test basic allocation
        void* ptr1 = manager.allocate(1024);
        void* ptr2 = manager.allocate(2048);
        
        if (!ptr1 || !ptr2) return false;
        
        // Test statistics
        auto stats = manager.get_stats();
        bool has_allocations = stats.total_allocated > 0;
        
        manager.deallocate(ptr1, 1024);
        manager.deallocate(ptr2, 2048);
        
        return has_allocations;
    } catch (const std::exception& e) {
        std::cerr << "Memory manager test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_compute_engine() {
    try {        clmodel::ai::AIComputeEngine& engine = clmodel::ai::AIComputeEngine::instance();
        
        // Test vector operations
        std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
        std::vector<double> b = {2.0, 3.0, 4.0, 5.0};
        std::vector<double> result(4);
        
        engine.vector_add(a.data(), b.data(), result.data(), 4);
        
        bool all_correct = true;
        for (int i = 0; i < 4; ++i) {
            if (std::abs(result[i] - (a[i] + b[i])) > 1e-6) {
                all_correct = false;
            }
        }
        
        // Test statistics
        auto stats = engine.get_stats();
        bool has_operations = stats.cpu_operations > 0;
        
        return all_correct && has_operations;
    } catch (const std::exception& e) {
        std::cerr << "Compute engine test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_tensor_operations() {
    try {
        // Test advanced tensor operations
        Tensor a = Tensor::randn({2, 3, 4}, 0.0, 1.0);
        Tensor b = Tensor::ones({2, 3, 4});
        
        // Test broadcasting
        Tensor c = Tensor::ones({3, 4});
        Tensor result = a + c; // Should broadcast
          // Test reductions
        auto sum_result = a.sum();
        auto mean_result = a.mean();
        auto max_result = a.max();
        
        // Verify reduction results have reasonable shapes
        bool reductions_valid = (sum_result.shape().size() <= 1) && 
                               (mean_result.shape().size() <= 1) &&
                               (max_result.shape().size() <= 1);
        
        // Test reshaping and transposing
        Tensor reshaped = a.reshape({6, 4});
        Tensor transposed = reshaped.transpose({1, 0});
        
        // Test matrix multiplication
        Tensor x = Tensor::randn({3, 4});
        Tensor y = Tensor::randn({4, 5});
        Tensor matmul_result = x.matmul(y);
        
        // Verify shapes
        bool shape_correct = (result.shape() == a.shape()) &&
                           (matmul_result.shape() == std::vector<size_t>{3, 5}) &&
                           (transposed.shape() == std::vector<size_t>{4, 6});
        
        return shape_correct && reductions_valid;
    } catch (const std::exception& e) {
        std::cerr << "Tensor operations test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_cnn_layers() {
    try {
        // Test CNN layer creation and forward pass
        auto conv_layer = std::make_unique<Conv2DLayer>(3, 16, 3, 1, 1); // 3->16 channels, 3x3 kernel
        auto pool_layer = std::make_unique<MaxPool2DLayer>(2, 2); // 2x2 pooling
        auto flatten_layer = std::make_unique<FlattenLayer>();
        
        // Create test input (batch=1, channels=3, height=32, width=32)
        Tensor input = Tensor::randn({1, 3, 32, 32});
        
        // Test forward passes
        Tensor conv_output = conv_layer->forward_tensor(input);
        Tensor pool_output = pool_layer->forward_tensor(conv_output);
        Tensor flatten_output = flatten_layer->forward_tensor(pool_output);
        
        // Verify shapes
        bool conv_shape = conv_output.shape() == std::vector<size_t>{1, 16, 32, 32};
        bool pool_shape = pool_output.shape() == std::vector<size_t>{1, 16, 16, 16};
        bool flatten_shape = flatten_output.ndim() == 2 && flatten_output.size(0) == 1;
        
        // Test backward compatibility with Matrix interface
        Matrix input_matrix(1, 3 * 32 * 32);
        for (size_t i = 0; i < input_matrix.cols(); ++i) {
            input_matrix(0, i) = input.data()[i];
        }
        
        Matrix conv_matrix_output = conv_layer->forward(input_matrix);
        bool matrix_compatible = conv_matrix_output.rows() == 1;
        
        return conv_shape && pool_shape && flatten_shape && matrix_compatible;
    } catch (const std::exception& e) {
        std::cerr << "CNN layers test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_attention_layers() {
    try {
        // Test multi-head attention
        size_t model_dim = 64;
        size_t num_heads = 8;
        size_t seq_len = 10;
        size_t batch_size = 2;
        
        auto attention_layer = std::make_unique<MultiHeadAttentionLayer>(model_dim, num_heads);
        attention_layer->initialize_weights("xavier");
        
        // Create test input
        Tensor input = Tensor::randn({batch_size, seq_len, model_dim});
        
        // Test self-attention forward pass
        Tensor attention_output = attention_layer->forward_tensor_self_attention(input);
        
        // Verify output shape
        bool attention_shape = attention_output.shape() == input.shape();
        
        // Test layer normalization
        auto norm_layer = std::make_unique<LayerNormalizationLayer>(model_dim);
        norm_layer->initialize_weights();
        
        Tensor norm_output = norm_layer->forward_tensor(attention_output);
        bool norm_shape = norm_output.shape() == attention_output.shape();
        
        // Test feed-forward layer
        auto ff_layer = std::make_unique<TransformerFeedForwardLayer>(model_dim, model_dim * 4);
        ff_layer->initialize_weights("xavier");
        
        Tensor ff_output = ff_layer->forward_tensor(norm_output);
        bool ff_shape = ff_output.shape() == norm_output.shape();
        
        return attention_shape && norm_shape && ff_shape;
    } catch (const std::exception& e) {
        std::cerr << "Attention layers test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_transformer_block() {
    try {
        size_t model_dim = 64;
        size_t num_heads = 8;
        size_t ff_hidden_dim = 256;
        size_t seq_len = 10;
        size_t batch_size = 2;
        
        auto transformer_block = std::make_unique<TransformerBlock>(model_dim, num_heads, ff_hidden_dim);
        
        // Create test input
        Tensor input = Tensor::randn({batch_size, seq_len, model_dim});
        
        // Test forward pass
        Tensor output = transformer_block->forward_tensor(input);
        
        // Verify output shape matches input (residual connections)
        bool shape_correct = output.shape() == input.shape();
        
        // Verify gradients can be computed (basic check)
        bool can_update = true;
        try {
            transformer_block->update_weights(0.001);
        } catch (...) {
            can_update = false;
        }
        
        return shape_correct && can_update;
    } catch (const std::exception& e) {
        std::cerr << "Transformer block test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_backward_compatibility() {
    try {
        // Test that new infrastructure works with existing Network class
        NeuralNetwork network;
        
        // Add some traditional layers
        network.add_layer(std::make_unique<DenseLayer>(10, 32));
        network.add_layer(std::make_unique<ActivationLayer>("relu", 32));
        
        // Add new AI layers
        network.add_layer(std::make_unique<Conv2DLayer>(1, 8, 3, 1, 1));
        network.add_layer(std::make_unique<MaxPool2DLayer>(2, 2));
        network.add_layer(std::make_unique<FlattenLayer>());
        network.add_layer(std::make_unique<DenseLayer>(200, 10)); // Adjust based on flatten output
        
        // Test that network can be created and configured
        Matrix test_input(1, 10);
        for (size_t i = 0; i < 10; ++i) {
            test_input(0, i) = 0.1 * i;
        }
        
        // This should work without throwing
        Matrix output = network.forward(test_input);
        
        return output.rows() == 1 && output.cols() > 0;
    } catch (const std::exception& e) {
        std::cerr << "Backward compatibility test failed: " << e.what() << std::endl;
        return false;
    }
}

// Performance benchmarks
void run_performance_benchmarks() {
    print_section("PERFORMANCE BENCHMARKS");
    Timer timer;
    
    // Tensor operations benchmark
    timer.start();
    for (int i = 0; i < 100; ++i) {
        Tensor a = Tensor::randn({100, 100});
        Tensor b = Tensor::randn({100, 100});
        Tensor c = a.matmul(b);
    }
    double tensor_time = timer.elapsed_ms();
    print_result("100x Matrix Multiplication (100x100)", true, tensor_time);
    
    // CNN layers benchmark
    timer.start();
    auto conv_layer = std::make_unique<Conv2DLayer>(3, 32, 3, 1, 1);
    Tensor input = Tensor::randn({1, 3, 64, 64});
    for (int i = 0; i < 10; ++i) {
        Tensor output = conv_layer->forward_tensor(input);
    }
    double cnn_time = timer.elapsed_ms();
    print_result("10x Conv2D Forward (3->32, 64x64)", true, cnn_time);
    
    // Attention benchmark
    timer.start();
    auto attention = std::make_unique<MultiHeadAttentionLayer>(128, 8);
    attention->initialize_weights();
    Tensor attn_input = Tensor::randn({2, 50, 128});
    for (int i = 0; i < 5; ++i) {
        Tensor output = attention->forward_tensor_self_attention(attn_input);
    }
    double attention_time = timer.elapsed_ms();
    print_result("5x Self-Attention (128 dim, 8 heads, 50 seq)", true, attention_time);
}

// Memory usage analysis
void analyze_memory_usage() {
    print_section("MEMORY USAGE ANALYSIS");
    
    clmodel::ai::AIMemoryManager& manager = clmodel::ai::AIMemoryManager::instance();
    auto initial_stats = manager.get_stats();
    
    // Create various objects and track memory
    std::vector<std::unique_ptr<Tensor>> tensors;
    for (int i = 0; i < 10; ++i) {
        tensors.push_back(std::make_unique<Tensor>(Tensor::randn({100, 100})));
    }
    
    auto tensor_stats = manager.get_stats();
    
    // Create attention layers
    std::vector<std::unique_ptr<MultiHeadAttentionLayer>> attention_layers;
    for (int i = 0; i < 3; ++i) {
        auto layer = std::make_unique<MultiHeadAttentionLayer>(128, 8);
        layer->initialize_weights();
        attention_layers.push_back(std::move(layer));
    }
    
    auto final_stats = manager.get_stats();
    
    std::cout << "  Initial Memory: " << initial_stats.total_allocated << " bytes\n";
    std::cout << "  After Tensors:  " << tensor_stats.total_allocated << " bytes (+";
    std::cout << (tensor_stats.total_allocated - initial_stats.total_allocated) << ")\n";
    std::cout << "  After Attention: " << final_stats.total_allocated << " bytes (+";
    std::cout << (final_stats.total_allocated - tensor_stats.total_allocated) << ")\n";    std::cout << "  Peak Memory:    " << final_stats.peak_usage << " bytes\n";
    std::cout << "  Total Allocs:   " << final_stats.total_allocations << "\n";
}

int main() {
    std::cout << "CLModel AI Infrastructure Demo\n";
    std::cout << "Testing Phase 1 components and integration\n";
    
    // Core infrastructure tests
    print_section("CORE INFRASTRUCTURE TESTS");
    Timer timer;
    
    timer.start();
    bool memory_ok = test_memory_manager();
    print_result("Memory Manager", memory_ok, timer.elapsed_ms());
    
    timer.start();
    bool compute_ok = test_compute_engine();
    print_result("Compute Engine", compute_ok, timer.elapsed_ms());
    
    timer.start();
    bool tensor_ok = test_tensor_operations();
    print_result("Tensor Operations", tensor_ok, timer.elapsed_ms());
    
    // AI components tests
    print_section("AI COMPONENTS TESTS");
    
    timer.start();
    bool cnn_ok = test_cnn_layers();
    print_result("CNN Layers", cnn_ok, timer.elapsed_ms());
    
    timer.start();
    bool attention_ok = test_attention_layers();
    print_result("Attention Layers", attention_ok, timer.elapsed_ms());
    
    timer.start();
    bool transformer_ok = test_transformer_block();
    print_result("Transformer Block", transformer_ok, timer.elapsed_ms());
    
    // Integration tests
    print_section("INTEGRATION TESTS");
    
    timer.start();
    bool compat_ok = test_backward_compatibility();
    print_result("Backward Compatibility", compat_ok, timer.elapsed_ms());
    
    // Performance and memory analysis
    run_performance_benchmarks();
    analyze_memory_usage();
    
    // Summary
    print_section("SUMMARY");
    
    int total_tests = 7;
    int passed_tests = memory_ok + compute_ok + tensor_ok + cnn_ok + 
                      attention_ok + transformer_ok + compat_ok;
    
    std::cout << "  Tests Passed: " << passed_tests << "/" << total_tests << "\n";
    std::cout << "  Success Rate: " << std::fixed << std::setprecision(1) 
              << (100.0 * passed_tests / total_tests) << "%\n";
    
    if (passed_tests == total_tests) {
        std::cout << "\n\033[32m✓ All Phase 1 infrastructure tests PASSED!\033[0m\n";
        std::cout << "Ready to proceed with Phase 2 development.\n";
        return 0;
    } else {
        std::cout << "\n\033[31m✗ Some tests FAILED. Review implementation.\033[0m\n";
        return 1;
    }
}
