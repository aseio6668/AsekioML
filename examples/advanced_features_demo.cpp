#include "clmodel.hpp"
#include "modern_api.hpp"
#include "simd_matrix.hpp"
#include "memory_optimization.hpp"
#include "production_features.hpp"
#include <iostream>
#include <chrono>
#include <memory>

using namespace clmodel;

// Demonstration of advanced features
void demonstrate_modern_api() {
    std::cout << "\n=== Modern API Demonstration ===" << std::endl;
    
    // Fluent model building
    auto model = api::ModelBuilder()
        .input(784)
        .dense(128, "relu")
        .dropout(0.3)
        .dense(64, "relu")
        .dropout(0.2)
        .dense(10, "softmax")
        .compile("categorical_crossentropy", "adam", 0.001)
        .build();
    
    std::cout << "✓ Built model with fluent API" << std::endl;
    
    // Quick MLP creation
    auto mlp = api::ModelBuilder::mlp({784, 128, 64, 10}, "relu", "softmax").build();
    std::cout << "✓ Created MLP with shorthand syntax" << std::endl;
    
    // Autoencoder
    auto autoencoder = api::ModelBuilder::autoencoder(784, {256, 128, 64}).build();
    std::cout << "✓ Created autoencoder architecture" << std::endl;
}

void demonstrate_simd_performance() {
    std::cout << "\n=== SIMD Performance Demonstration ===" << std::endl;
    
    const size_t size = 512;
    Matrix A = Matrix::random(size, size, -1.0, 1.0);
    Matrix B = Matrix::random(size, size, -1.0, 1.0);
    
    // Regular matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_regular = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // SIMD matrix multiplication
    simd::SIMDMatrix A_simd = A;
    simd::SIMDMatrix B_simd = B;
    
    start = std::chrono::high_resolution_clock::now();
    Matrix C_simd = A_simd.multiply_simd(B_simd);
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Parallel multiplication
    start = std::chrono::high_resolution_clock::now();
    Matrix C_parallel = A_simd.multiply_parallel(B_simd);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Matrix multiplication (" << size << "x" << size << "):" << std::endl;
    std::cout << "  Regular: " << regular_time.count() << "ms" << std::endl;
    std::cout << "  SIMD: " << simd_time.count() << "ms" << std::endl;
    std::cout << "  Parallel: " << parallel_time.count() << "ms" << std::endl;
    
    double simd_speedup = static_cast<double>(regular_time.count()) / simd_time.count();
    double parallel_speedup = static_cast<double>(regular_time.count()) / parallel_time.count();
    
    std::cout << "  SIMD speedup: " << simd_speedup << "x" << std::endl;
    std::cout << "  Parallel speedup: " << parallel_speedup << "x" << std::endl;
    
    // Verify correctness (check if results are similar)
    double diff = 0.0;
    for (size_t i = 0; i < size && i < 10; ++i) {
        for (size_t j = 0; j < size && j < 10; ++j) {
            diff += std::abs(C_regular(i, j) - C_simd(i, j));
        }
    }
    std::cout << "  Average difference (first 10x10): " << diff / 100.0 << std::endl;
}

void demonstrate_memory_optimization() {
    std::cout << "\n=== Memory Optimization Demonstration ===" << std::endl;
    
    auto& pool = memory::MemoryPool::instance();
    
    std::cout << "Initial memory usage: " << pool.get_total_allocated() << " bytes" << std::endl;
    
    // Allocate matrices using memory pool
    std::vector<memory::PoolPtr<double>> matrices;
    
    for (int i = 0; i < 100; ++i) {
        matrices.emplace_back(1024 * 1024); // 1M doubles each
    }
    
    std::cout << "After allocating 100 matrices: " << pool.get_total_allocated() << " bytes" << std::endl;
    std::cout << "Peak usage: " << pool.get_peak_usage() << " bytes" << std::endl;
    
    // Clear half the matrices
    matrices.erase(matrices.begin(), matrices.begin() + 50);
    
    std::cout << "After freeing 50 matrices: " << pool.get_total_allocated() << " bytes" << std::endl;
    
    // Demonstrate cache-friendly matrix
    memory::CacheFriendlyMatrix<double> cache_matrix(256, 256);
    
    // Fill with data
    for (size_t i = 0; i < 256; ++i) {
        for (size_t j = 0; j < 256; ++j) {
            cache_matrix(i, j) = static_cast<double>(i * 256 + j);
        }
    }
    
    std::cout << "✓ Created cache-friendly matrix with optimized memory layout" << std::endl;
}

void demonstrate_production_features() {
    std::cout << "\n=== Production Features Demonstration ===" << std::endl;
    
    // Create a simple model
    auto model = std::make_unique<NeuralNetwork>();
    model->add_dense_layer(10, 5, "relu");
    model->add_dense_layer(5, 1, "sigmoid");
    model->set_loss_function("binary_crossentropy");
    model->set_optimizer("adam", 0.001);
    
    // Register model in registry
    auto& registry = production::ModelRegistry::instance();
    std::unordered_map<std::string, std::string> metadata;
    metadata["description"] = "Binary classification model";
    metadata["author"] = "CLModel Demo";
    
    registry.register_model("binary_classifier", "v1.0", 
                           std::make_unique<NeuralNetwork>(*model), 
                           0.85, metadata);
    
    std::cout << "✓ Registered model in registry" << std::endl;
    
    // List models
    auto model_names = registry.list_models();
    std::cout << "Available models: ";
    for (const auto& name : model_names) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    
    // Create inference server
    production::InferenceServer::ServerConfig config;
    config.max_batch_size = 16;
    config.num_workers = 4;
    config.batch_timeout = std::chrono::milliseconds(50);
    
    production::InferenceServer server(std::move(model), config);
    server.start();
    
    std::cout << "✓ Started inference server with " << config.num_workers << " workers" << std::endl;
    
    // Simulate predictions
    std::vector<Matrix> test_inputs;
    for (int i = 0; i < 10; ++i) {
        test_inputs.push_back(Matrix::random(1, 10, -1.0, 1.0));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = server.predict_batch(test_inputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Batch prediction of " << test_inputs.size() 
              << " samples took " << duration.count() << "ms" << std::endl;
    
    // Get performance metrics
    auto metrics = server.get_metrics();
    std::cout << "Server metrics:" << std::endl;
    std::cout << "  Total requests: " << metrics.total_requests << std::endl;
    std::cout << "  Successful requests: " << metrics.successful_requests << std::endl;
    std::cout << "  Average latency: " << metrics.average_latency_ms << "ms" << std::endl;
    std::cout << "  Throughput: " << metrics.throughput_rps << " requests/sec" << std::endl;
    
    server.stop();
    std::cout << "✓ Stopped inference server" << std::endl;
}

void demonstrate_training_with_callbacks() {
    std::cout << "\n=== Advanced Training with Callbacks ===" << std::endl;
      // Generate synthetic dataset
    Dataset dataset = datasets::make_classification(1000, 20, 2);
    
    // Build model with modern API
    auto model = api::ModelBuilder()
        .input(20)
        .dense(64, "relu")
        .dropout(0.3)
        .dense(32, "relu")
        .dropout(0.2)
        .dense(2, "softmax")
        .compile("categorical_crossentropy", "adam", 0.001)
        .build();
    
    // Setup training configuration
    api::Trainer::TrainingConfig config;
    config.epochs = 50;
    config.batch_size = 32;
    config.validation_split = 0.2;
    config.verbose = true;
    
    // Create callbacks
    std::vector<std::unique_ptr<api::Trainer::Callback>> callbacks;
    callbacks.push_back(std::make_unique<api::Trainer::EarlyStopping>(10, 1e-4));
    callbacks.push_back(std::make_unique<api::Trainer::ModelCheckpoint>("best_model.clm", true));
    
    std::cout << "Starting training with early stopping and model checkpointing..." << std::endl;
    
    // Train model
    auto history = api::Trainer::fit(*model, dataset, config, callbacks);
    
    std::cout << "✓ Training completed!" << std::endl;
    std::cout << "Final training loss: " << history.training_loss.back() << std::endl;
    if (!history.validation_loss.empty()) {
        std::cout << "Final validation loss: " << history.validation_loss.back() << std::endl;
    }
}

void demonstrate_hyperparameter_tuning() {
    std::cout << "\n=== Hyperparameter Tuning Demonstration ===" << std::endl;
    
    // Generate dataset
    Dataset dataset = datasets::make_classification(500, 10, 2);
    
    // Define hyperparameter space
    api::HyperparameterTuner::HyperparameterSpace space;
    space.learning_rate = std::make_pair(1e-4, 1e-1);
    space.batch_size = std::vector<int>{16, 32, 64};
    space.optimizer = std::vector<std::string>{"adam", "sgd", "rmsprop"};
    space.dropout_rate = std::make_pair(0.1, 0.5);
    space.hidden_layers = std::vector<std::vector<int>>{{32}, {64}, {32, 16}, {64, 32}};
    
    // Define objective function
    auto objective = [&dataset](const api::HyperparameterTuner::HyperparameterConfig& config) -> double {
        try {
            // Build model with hyperparameters
            auto model = api::ModelBuilder()
                .input(10)
                .dense(config.hidden_layers[0], "relu")
                .dropout(config.dropout_rate)
                .dense(2, "softmax")
                .compile("categorical_crossentropy", config.optimizer, config.learning_rate)
                .build();
            
            // Quick training
            model->fit(dataset.features(), dataset.targets(), 10, config.batch_size, 0.2, false);
            
            // Return validation loss
            auto [train, val] = dataset.train_test_split(0.8);
            return model->evaluate(val.features(), val.targets());
            
        } catch (...) {
            return std::numeric_limits<double>::max();
        }
    };
    
    std::cout << "Starting hyperparameter search..." << std::endl;
    
    // Run random search
    api::HyperparameterTuner tuner;
    auto result = tuner.random_search(objective, space, 20);
    
    std::cout << "✓ Hyperparameter tuning completed!" << std::endl;
    std::cout << "Best score: " << result.best_score << std::endl;
    std::cout << "Best config:" << std::endl;
    std::cout << "  Learning rate: " << result.best_config.learning_rate << std::endl;
    std::cout << "  Batch size: " << result.best_config.batch_size << std::endl;
    std::cout << "  Optimizer: " << result.best_config.optimizer << std::endl;
    std::cout << "  Dropout rate: " << result.best_config.dropout_rate << std::endl;
}

int main() {
    std::cout << "CLModel Advanced Features Demonstration" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        demonstrate_modern_api();
        demonstrate_simd_performance();
        demonstrate_memory_optimization();
        demonstrate_production_features();
        demonstrate_training_with_callbacks();
        demonstrate_hyperparameter_tuning();
        
        std::cout << "\n🎉 All demonstrations completed successfully!" << std::endl;
        std::cout << "\nKey advantages of CLModel framework:" << std::endl;
        std::cout << "✓ High-performance SIMD operations" << std::endl;
        std::cout << "✓ Memory-efficient allocation strategies" << std::endl;
        std::cout << "✓ Modern, fluent API design" << std::endl;
        std::cout << "✓ Production-ready inference server" << std::endl;
        std::cout << "✓ Advanced training features with callbacks" << std::endl;
        std::cout << "✓ Automated hyperparameter tuning" << std::endl;
        std::cout << "✓ Model versioning and registry" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
