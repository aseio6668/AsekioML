# AsekioML - High-Performance C++ Machine Learning Framework

A comprehensive, from-scratch machine learning framework implemented in modern C++17. AsekioML provides all the essential components needed to build, train, and deploy neural networks with advanced performance optimizations and production-ready features.

## 🚀 Key Advantages

**Performance & Efficiency:**
- **SIMD-Optimized Operations**: AVX2-accelerated matrix operations for 2-4x speedup
- **Parallel Processing**: OpenMP-based parallel matrix multiplication and training
- **Memory Pool Allocation**: Custom memory management for reduced allocation overhead
- **Cache-Friendly Data Structures**: Optimized memory layouts for better performance

**Modern API Design:**
- **Fluent Interface**: Method chaining for intuitive model building
- **Automatic Best Practices**: Smart defaults and auto-configuration
- **Type Safety**: Modern C++17 features for compile-time safety
- **Zero Dependencies**: Self-contained framework with no external requirements

**Production Ready:**
- **High-Performance Inference Server**: Multi-threaded server with batching
- **Model Registry & Versioning**: Enterprise-grade model management
- **Advanced Training**: Callbacks, early stopping, automatic hyperparameter tuning
- **Statistical Monitoring**: Complete suite with PSI, Wasserstein, KS, and Chi-square tests for drift detection

## Features

### Core Components
- **Matrix Operations**: Efficient matrix class with SIMD-optimized linear algebra
- **Neural Network Layers**: Dense, Activation, and Dropout layers with advanced variants
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy, MAE, Huber Loss
- **Optimizers**: SGD (with momentum), Adam, RMSprop, AdaGrad
- **Dataset Management**: Data loading, preprocessing, and synthetic data generation

### Advanced Features ⭐
- **SIMD Matrix Operations**: 2-4x faster matrix multiplication with AVX2
- **Memory Optimization**: Custom allocators and memory pools for efficiency
- **Modern Fluent API**: Intuitive model building with method chaining
- **Production Features**: Inference server, model registry, monitoring
- **Statistical Monitoring**: Complete drift detection suite with PSI, Wasserstein, KS, and Chi-square tests
- **Advanced Training**: Callbacks, early stopping, hyperparameter tuning
- **Parallel Processing**: OpenMP-accelerated operations
- **Multi-Vendor GPU Support**: NVIDIA CUDA, AMD ROCm, and OpenCL acceleration

### GPU Acceleration 🚀
- **NVIDIA GPUs**: Full CUDA support with cuBLAS and cuDNN optimization
- **AMD GPUs**: Native ROCm/HIP support with rocBLAS acceleration
- **Intel GPUs**: OpenCL support for Intel Arc and integrated graphics
- **Cross-Platform**: OpenCL fallback for maximum hardware compatibility
- **Automatic Detection**: Framework automatically selects the best available GPU

## Quick Start

### Building the Framework

```bash
mkdir build
cd build
cmake ..
make

# Run performance demonstrations
./advanced_demo
```

### Modern API Usage (Recommended)

```cpp
#include "asekioml.hpp"

int main() {
    // Print framework capabilities
    asekioml::print_info();
    
    // Create dataset
    auto dataset = asekioml::Dataset::make_classification(1000, 20, 3);
    
    // Build model with fluent API
    auto model = asekioml::api::ModelBuilder()
        .input(20)
        .dense(128, "relu")
        .dropout(0.3)
        .dense(64, "relu")
        .dropout(0.2)
        .dense(3, "softmax")
        .compile("categorical_crossentropy", "adam", 0.001)
        .build();
    
    // Advanced training with callbacks
    asekioml::api::Trainer::TrainingConfig config;
    config.epochs = 100;
    config.batch_size = 32;
    config.validation_split = 0.2;
    
    std::vector<std::unique_ptr<asekioml::api::Trainer::Callback>> callbacks;
    callbacks.push_back(std::make_unique<asekioml::api::Trainer::EarlyStopping>(10));
    callbacks.push_back(std::make_unique<asekioml::api::Trainer::ModelCheckpoint>("best_model.clm"));
    
    auto history = asekioml::api::Trainer::fit(*model, dataset, config, callbacks);
      return 0;
}
```

### GPU Acceleration Demo

```cpp
#include "asekioml.hpp"
using namespace asekioml::gpu;

int main() {
    // Detect available GPU devices
    auto devices = GPUMatrix::get_available_devices();
    for (const auto& device : devices) {
        std::cout << "Found " << device.name 
                  << " with " << device.memory_mb << " MB" << std::endl;
    }
    
    // Create matrices for GPU computation
    Matrix a(1024, 1024);  // Initialize with data
    Matrix b(1024, 1024);
    
    // GPU-accelerated operations with automatic device selection
    Matrix result = gpu_ops::multiply_gpu(a, b);           // Auto-detect best GPU
    Matrix cuda_result = gpu_ops::multiply_gpu(a, b, DeviceType::CUDA);   // Force NVIDIA
    Matrix rocm_result = gpu_ops::multiply_gpu(a, b, DeviceType::ROCM);   // Force AMD
    
    // Train neural networks on GPU
    GPUDenseLayer layer(784, 128, DeviceType::AUTO);
    // ... training code ...
    
    return 0;
}
```

## Detailed Examples

### 1. Regression Problem

```cpp
// Create synthetic regression data
auto dataset = asekioml::datasets::make_regression(500, 2, 0.1, 42);
dataset.normalize_features();

// Split data
auto [train_data, test_data] = dataset.train_test_split(0.2);

// Build network manually
asekioml::NeuralNetwork network;
network.add_dense_layer(2, 32, "relu");
network.add_dropout_layer(32, 0.2);
network.add_dense_layer(32, 16, "relu");
network.add_dense_layer(16, 1, "linear");
network.compile("mse", "adam", 0.001);

// Train
network.fit(train_data.features(), train_data.targets(), 150, 32, 0.2, true);
```

### 2. Classification Problem

```cpp
// Create synthetic classification data
auto dataset = asekioml::datasets::make_classification(800, 4, 3, 0.1, 42);
dataset.normalize_features();

// Create network for 3-class classification
auto network = asekioml::create_mlp(
    {4, 64, 32, 3},
    {"relu", "relu", "softmax"},
    "cross_entropy",
    "adam",
    0.001
);

// Train and evaluate
auto [train_data, test_data] = dataset.train_test_split(0.2);
network->fit(train_data.features(), train_data.targets(), 100, 32, 0.2);

asekioml::Matrix predictions = network->predict(test_data.features());
double accuracy = network->calculate_accuracy(predictions, test_data.targets());
```

### 3. XOR Problem (Non-linear)

```cpp
// Classic XOR problem
auto dataset = asekioml::datasets::make_xor(400, 0.05);

asekioml::NeuralNetwork network;
network.add_dense_layer(2, 8, "relu");
network.add_dense_layer(8, 4, "relu");
network.add_dense_layer(4, 1, "sigmoid");
network.compile("binary_cross_entropy", "adam", 0.01);

auto [train_data, test_data] = dataset.train_test_split(0.3);
network.fit(train_data.features(), train_data.targets(), 200, 16);
```

## Matrix Operations

The framework includes a comprehensive Matrix class:

```cpp
// Create matrices
asekioml::Matrix A = {{1, 2, 3}, {4, 5, 6}};
asekioml::Matrix B = asekioml::Matrix::random(3, 2, -1.0, 1.0);

// Operations
asekioml::Matrix C = A * B;                    // Matrix multiplication
asekioml::Matrix D = A + A;                    // Element-wise addition
asekioml::Matrix E = A.transpose();            // Transpose
asekioml::Matrix F = A.hadamard(A);           // Element-wise multiplication

// Apply functions
asekioml::Matrix G = A.apply([](double x) { 
    return std::tanh(x); 
});

// Statistics
double mean = A.mean();
double std_dev = A.std_dev();
```

## Dataset Management

```cpp
// Load from CSV
asekioml::Dataset dataset = asekioml::Dataset::load_csv("data.csv", true, ',', 1);

// Data preprocessing
dataset.normalize_features();
dataset.shuffle();

// Create synthetic datasets
auto regression_data = asekioml::datasets::make_regression(1000, 5, 0.1);
auto classification_data = asekioml::datasets::make_classification(500, 3, 2);
auto xor_data = asekioml::datasets::make_xor(200, 0.1);
auto circle_data = asekioml::datasets::make_circles(300, 0.1, 0.7);

// Save processed data
dataset.save_csv("processed_data.csv");
```

## Activation Functions

Available activation functions:
- **ReLU**: `f(x) = max(0, x)`
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **Tanh**: `f(x) = tanh(x)`
- **Softmax**: For multi-class classification
- **LeakyReLU**: `f(x) = x if x > 0 else α*x`
- **Linear**: Identity function

```cpp
// Create activation layers
auto relu_layer = std::make_unique<asekioml::ActivationLayer>("relu", 64);
auto sigmoid_layer = std::make_unique<asekioml::ActivationLayer>("sigmoid", 32);

// Or use in network building
network.add_dense_layer(64, 32, "relu");
network.add_activation_layer("softmax", 10);
```

## Loss Functions

- **Mean Squared Error**: For regression
- **Cross-Entropy**: For multi-class classification
- **Binary Cross-Entropy**: For binary classification
- **Mean Absolute Error**: Robust to outliers
- **Huber Loss**: Combination of MSE and MAE

## Optimizers

### SGD (Stochastic Gradient Descent)
```cpp
auto sgd = std::make_unique<asekioml::SGD>(0.01, 0.9);  // lr=0.01, momentum=0.9
```

### Adam (Adaptive Moment Estimation)
```cpp
auto adam = std::make_unique<asekioml::Adam>(0.001, 0.9, 0.999, 1e-8);
```

### RMSprop
```cpp
auto rmsprop = std::make_unique<asekioml::RMSprop>(0.001, 0.9, 1e-8);
```

### AdaGrad
```cpp
auto adagrad = std::make_unique<asekioml::AdaGrad>(0.01, 1e-8);
```

## Model Training and Evaluation

```cpp
// Compile the network
network.compile("mse", "adam", 0.001);

// Train with validation split
network.fit(X_train, y_train, 
           epochs=100, 
           batch_size=32, 
           validation_split=0.2, 
           verbose=true);

// Or train with separate validation set
network.fit(X_train, y_train, X_val, y_val, 100, 32, true);

// Evaluate performance
double test_loss = network.evaluate(X_test, y_test);
double accuracy = network.calculate_accuracy(predictions, y_test);

// Access training history
const auto& history = network.get_history();
for (size_t i = 0; i < history.training_loss.size(); ++i) {
    std::cout << "Epoch " << i+1 << ": Loss = " << history.training_loss[i] << std::endl;
}
```

## Model Information

```cpp
// Display network architecture
network.summary();

// Count parameters
size_t total_params = network.count_parameters();

// Check if compiled
bool is_ready = network.is_compiled();
```

## Advanced Features

### Custom Training Loops
```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (int batch = 0; batch < num_batches; ++batch) {
        // Get batch data
        asekioml::Matrix X_batch = get_batch_features(batch);
        asekioml::Matrix y_batch = get_batch_targets(batch);
        
        // Single training step
        network.train_step(X_batch, y_batch);
    }
}
```

### Dropout for Regularization
```cpp
network.add_dense_layer(128, 64, "relu");
network.add_dropout_layer(64, 0.5);  // 50% dropout rate
network.add_dense_layer(64, 32, "relu");
```

### Weight Initialization
```cpp
asekioml::DenseLayer layer(128, 64);
layer.initialize_xavier();  // Xavier/Glorot initialization
// or
layer.initialize_he();      // He initialization
// or
layer.initialize_random(-0.1, 0.1);  // Random uniform
```

## Building from Source

### Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16+

### Optional GPU Requirements
For GPU acceleration, install one or more of the following:

**NVIDIA GPUs (CUDA)**:
- CUDA Toolkit 11.0+ ([Download](https://developer.nvidia.com/cuda-downloads))
- cuBLAS (included with CUDA)
- cuDNN 8.0+ (optional, for advanced optimizations)

**AMD GPUs (ROCm)**:
- ROCm 4.0+ ([Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html))
- rocBLAS (included with ROCm)
- Compatible with RX 5000/6000 series and RDNA/RDNA2 architecture

**Intel/Cross-Platform (OpenCL)**:
- OpenCL 2.0+ drivers
- Intel Arc GPU drivers for discrete GPUs
- Integrated graphics drivers for Intel CPUs

### Build Instructions
- ROCm 4.0+ ([Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html))
- rocBLAS (included with ROCm)
- Supported on AMD RX 5000/6000 series and RDNA/RDNA2 architecture

**Intel/Cross-Platform (OpenCL)**:
- OpenCL 2.0+ drivers (usually included with GPU drivers)
- Intel Arc GPUs, Intel integrated graphics, or any OpenCL-compatible device

The framework will automatically detect and use the best available GPU. If no GPU is found, it will run efficiently on CPU with SIMD optimizations.

### Compilation
```bash
git clone <repository>
cd CLModel
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
./tests

# Run basic example
./example

# Run advanced features demo
./advanced_demo

# Run GPU acceleration demo (requires GPU)
./gpu_demo

# Run transparency demo (debugging & monitoring)
./transparency_demo

# Run Python pain points demo
./python_pain_points_demo

# Run statistical monitoring demo (PSI, Wasserstein, KS, Chi-square)
./statistical_monitoring_demo
```

### Windows (Visual Studio)
```cmd
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release

# Run demos
.\Release\example.exe
.\Release\advanced_demo.exe
.\Release\gpu_demo.exe
.\Release\transparency_demo.exe
.\Release\python_pain_points_demo.exe
.\Release\statistical_monitoring_demo.exe
```

## Architecture

The framework is organized into several key components:

```
CLModel/
├── include/           # Header files
│   ├── matrix.hpp     # Matrix operations
│   ├── activation.hpp # Activation functions
│   ├── layer.hpp      # Neural network layers
│   ├── loss.hpp       # Loss functions
│   ├── optimizer.hpp  # Optimization algorithms
│   ├── network.hpp    # Neural network class
│   ├── dataset.hpp    # Data management
│   └── clmodel.hpp    # Main header
├── src/               # Implementation files
├── examples/          # Usage examples
├── tests/             # Unit tests
└── CMakeLists.txt     # Build configuration
```

## Performance Tips

1. **Use appropriate batch sizes**: 32-128 typically work well
2. **Normalize your data**: Use `dataset.normalize_features()`
3. **Choose the right optimizer**: Adam is generally a good default
4. **Experiment with learning rates**: Start with 0.001 for Adam, 0.01 for SGD
5. **Use dropout for regularization**: Helps prevent overfitting
6. **Monitor training**: Set `verbose=true` to watch training progress

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

This framework implements standard machine learning algorithms and techniques from the research community. It's designed for educational purposes and practical applications where you need full control over the ML pipeline.
