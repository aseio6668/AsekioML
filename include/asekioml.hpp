#pragma once

// Main header file that includes all AsekioML components

// Core components
#include "matrix.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "network.hpp"
#include "dataset.hpp"

// Advanced features
#include "regularization.hpp"
#include "modern_api.hpp"
#include "simd_matrix.hpp"
#include "memory_optimization.hpp"
#include "production_features.hpp"
#include "profiling.hpp"
#include "tokenizer.hpp"
#include "threading.hpp"

// Advanced training and optimization
#include "schedulers.hpp"
#include "advanced_optimizers.hpp"
#include "callbacks.hpp"
#include "gradient_clipping.hpp"
#include "advanced_losses.hpp"
#include "advanced_trainer.hpp"

// Optional advanced components (header-only for now)
#include "gpu_acceleration.hpp"
#include "advanced_layers.hpp"
#include "model_zoo.hpp"

namespace asekioml {

// Version information
const char* version() { return "2.0.0"; } // Bumped version for advanced features

// Framework capabilities
struct FrameworkInfo {
    static constexpr bool has_simd_support = true;
    static constexpr bool has_parallel_support = true;
    static constexpr bool has_memory_optimization = true;
    static constexpr bool has_modern_api = true;
    static constexpr bool has_production_features = true;
    static constexpr bool has_tokenization = true;
    static constexpr bool has_advanced_threading = true;
    static constexpr bool has_transparency_tools = true;
    
#ifdef ASEKIOML_CUDA_SUPPORT
    static constexpr bool has_gpu_support = true;
#else
    static constexpr bool has_gpu_support = false;
#endif

#ifdef ASEKIOML_OPENMP_SUPPORT
    static constexpr bool has_openmp = true;
#else
    static constexpr bool has_openmp = false;
#endif
};

// Print framework capabilities
inline void print_info() {
    std::cout << "CLModel v" << version() << " - C++ Machine Learning Framework" << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  ✓ Core ML components (layers, optimizers, loss functions)" << std::endl;
    std::cout << "  ✓ Modern fluent API with method chaining" << std::endl;
    std::cout << "  ✓ SIMD-optimized matrix operations" << std::endl;
    std::cout << "  ✓ Memory pool allocation for efficiency" << std::endl;
    std::cout << "  ✓ Production-ready inference server" << std::endl;
    std::cout << "  ✓ Advanced training with callbacks" << std::endl;
    std::cout << "  ✓ Automated hyperparameter tuning" << std::endl;
    std::cout << "  ✓ Model versioning and registry" << std::endl;
    
    if (FrameworkInfo::has_gpu_support) {
        std::cout << "  ✓ GPU acceleration (CUDA)" << std::endl;
    }
    if (FrameworkInfo::has_openmp) {
        std::cout << "  ✓ OpenMP parallel processing" << std::endl;
    }
}

// Convenience functions for quick model creation (legacy API)
inline std::unique_ptr<NeuralNetwork> create_mlp(
    const std::vector<size_t>& layer_sizes,
    const std::vector<std::string>& activations,
    const std::string& loss_function = "mse",
    const std::string& optimizer = "adam",
    double learning_rate = 0.001) {
    
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Must have at least input and output layer");
    }
    
    if (activations.size() != layer_sizes.size() - 1) {
        throw std::invalid_argument("Number of activations must equal number of layer transitions");
    }
    
    auto network = std::make_unique<NeuralNetwork>();
    
    // Add layers
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        network->add_dense_layer(layer_sizes[i], layer_sizes[i + 1], activations[i]);
    }
    
    // Compile network
    network->set_loss_function(loss_function);
    network->set_optimizer(optimizer, learning_rate);
    
    return network;
}

// Modern API shortcuts (recommended)
namespace quick {
    // Quick MLP creation using modern API
    inline std::unique_ptr<NeuralNetwork> mlp(const std::vector<size_t>& sizes) {
        return api::ModelBuilder::mlp(sizes).compile().build();
    }
    
    // Quick classification model
    inline std::unique_ptr<NeuralNetwork> classifier(size_t input_dim, size_t num_classes) {
        return api::ModelBuilder()
            .input(input_dim)
            .dense(128, "relu")
            .dropout(0.3)
            .dense(64, "relu")
            .dropout(0.2)
            .dense(num_classes, "softmax")
            .compile("categorical_crossentropy", "adam")
            .build();
    }
    
    // Quick regression model
    inline std::unique_ptr<NeuralNetwork> regressor(size_t input_dim, size_t output_dim = 1) {
        return api::ModelBuilder()
            .input(input_dim)
            .dense(64, "relu")
            .dense(32, "relu")
            .dense(output_dim, "linear")
            .compile("mse", "adam")
            .build();
    }
}

// Quick training function
inline TrainingHistory quick_train(
    NeuralNetwork& network,
    const Dataset& dataset,
    int epochs = 100,
    int batch_size = 32,
    double validation_split = 0.2,
    bool verbose = true) {
    
    network.fit(dataset.features(), dataset.targets(), epochs, batch_size, validation_split, verbose);
    return network.get_history();
}

} // namespace asekioml
