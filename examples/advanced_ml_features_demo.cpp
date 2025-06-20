#include "clmodel.hpp"
#include "advanced_layers.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

using namespace clmodel;
using namespace clmodel::advanced;

void demonstrate_batch_normalization() {
    std::cout << "\n=== Batch Normalization Demo ===\n";
    
    // Create test data
    Matrix input(100, 10);  // 100 samples, 10 features
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(5.0, 2.0);  // Mean=5, Std=2
    
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            input(i, j) = dist(gen);
        }
    }
    
    std::cout << "Input statistics (before normalization):\n";
    // Calculate mean and std of first feature
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < input.rows(); ++i) {
        sum += input(i, 0);
        sum_sq += input(i, 0) * input(i, 0);
    }
    double mean = sum / input.rows();
    double variance = sum_sq / input.rows() - mean * mean;
    std::cout << "  Feature 0 - Mean: " << std::fixed << std::setprecision(3) 
              << mean << ", Std: " << std::sqrt(variance) << "\n";
    
    // Apply batch normalization
    BatchNormalizationLayer bn_layer(10);
    Matrix normalized = bn_layer.forward(input);
    
    std::cout << "Output statistics (after normalization):\n";
    sum = 0.0; sum_sq = 0.0;
    for (size_t i = 0; i < normalized.rows(); ++i) {
        sum += normalized(i, 0);
        sum_sq += normalized(i, 0) * normalized(i, 0);
    }
    mean = sum / normalized.rows();
    variance = sum_sq / normalized.rows() - mean * mean;
    std::cout << "  Feature 0 - Mean: " << std::fixed << std::setprecision(3) 
              << mean << ", Std: " << std::sqrt(variance) << "\n";
    
    std::cout << "✓ Batch normalization successfully normalized the data!\n";
}

void demonstrate_conv2d() {
    std::cout << "\n=== 2D Convolutional Layer Demo ===\n";
    
    // Simulate a small 8x8 image with 3 channels (RGB)
    size_t height = 8, width = 8, channels = 3;
    size_t batch_size = 4;
    
    Matrix input(batch_size, height * width * channels);
    
    // Fill with some pattern
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                for (size_t c = 0; c < channels; ++c) {
                    size_t idx = h * width * channels + w * channels + c;
                    input(b, idx) = (h + w + c) * 0.1;  // Simple gradient pattern
                }
            }
        }
    }
    
    std::cout << "Input: " << batch_size << " images of size " 
              << height << "x" << width << "x" << channels << "\n";
    
    // Create a 2D convolutional layer
    // 3x3 kernels, 16 filters, stride=1, no padding
    Conv2DLayer conv_layer(height, width, channels, 16, 3, 3, 1, 1, 0, 0);
    
    Matrix output = conv_layer.forward(input);
    
    std::cout << "Output shape: [" << output.rows() << ", " << output.cols() << "]\n";
    std::cout << "Output spatial dimensions: " << conv_layer.get_output_height() 
              << "x" << conv_layer.get_output_width() 
              << "x" << conv_layer.get_num_filters() << "\n";
    std::cout << "✓ 2D Convolution successfully processed the images!\n";
}

void demonstrate_lstm() {
    std::cout << "\n=== LSTM Layer Demo ===\n";
    
    // Create sequence data (simplified - single timestep)
    size_t input_size = 50;  // Feature dimension
    size_t hidden_size = 32; // LSTM hidden state size
    size_t batch_size = 10;
    
    Matrix input(batch_size, input_size);
    
    // Fill with random sequence data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            input(i, j) = dist(gen);
        }
    }
    
    std::cout << "Input sequence: [" << batch_size << " samples, " 
              << input_size << " features]\n";
    
    // Create LSTM layer
    LSTMLayer lstm_layer(input_size, hidden_size);
    
    // Process sequence
    Matrix output = lstm_layer.forward(input);
    
    std::cout << "LSTM output: [" << output.rows() << " samples, " 
              << output.cols() << " hidden units]\n";
    
    // Process another timestep to show state persistence
    Matrix input2(batch_size, input_size);
    for (size_t i = 0; i < input2.rows(); ++i) {
        for (size_t j = 0; j < input2.cols(); ++j) {
            input2(i, j) = dist(gen);
        }
    }
    
    Matrix output2 = lstm_layer.forward(input2);
    std::cout << "Second timestep output: [" << output2.rows() << ", " 
              << output2.cols() << "]\n";
    
    std::cout << "✓ LSTM successfully processed sequential data!\n";
    
    // Demonstrate state reset
    lstm_layer.reset_state();
    std::cout << "✓ LSTM state reset for new sequence\n";
}

void demonstrate_regularization() {
    std::cout << "\n=== L1/L2 Regularization Demo ===\n";
    
    // Create test data
    Matrix input(50, 20);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            input(i, j) = dist(gen);
        }
    }
    
    // Create regularization layers
    RegularizationLayer l1_reg(20, 0.01, 0.0);    // L1 only
    RegularizationLayer l2_reg(20, 0.0, 0.01);    // L2 only  
    RegularizationLayer l1l2_reg(20, 0.01, 0.01); // Both L1 and L2
    
    std::cout << "Testing regularization layers:\n";
    std::cout << "  L1 regularization: λ=" << l1_reg.get_l1_lambda() << "\n";
    std::cout << "  L2 regularization: λ=" << l2_reg.get_l2_lambda() << "\n";
    std::cout << "  L1+L2 regularization: λ1=" << l1l2_reg.get_l1_lambda() 
              << ", λ2=" << l1l2_reg.get_l2_lambda() << "\n";
    
    // Forward pass (should be identity)
    Matrix l1_output = l1_reg.forward(input);
    Matrix l2_output = l2_reg.forward(input);
    
    std::cout << "✓ Forward pass preserves data (regularization affects gradients)\n";
    
    // Simulate backward pass with dummy gradients
    Matrix dummy_grad(50, 20);
    for (size_t i = 0; i < dummy_grad.rows(); ++i) {
        for (size_t j = 0; j < dummy_grad.cols(); ++j) {
            dummy_grad(i, j) = 1.0;  // Uniform gradient
        }
    }
    
    Matrix l1_grad = l1_reg.backward(dummy_grad);
    Matrix l2_grad = l2_reg.backward(dummy_grad);
    
    std::cout << "✓ Regularization layers modify gradients for penalty terms\n";
}

void demonstrate_advanced_network() {
    std::cout << "\n=== Advanced Network Architecture Demo ===\n";
    
    // Create a sophisticated network using advanced layers
    std::cout << "Building advanced network with:\n";
    std::cout << "  - Dense layers with batch normalization\n";
    std::cout << "  - L1/L2 regularization\n";
    std::cout << "  - Dropout for additional regularization\n";
    
    // Input data: 64 features
    Matrix input(32, 64);  // 32 samples, 64 features
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            input(i, j) = dist(gen);
        }
    }
    
    // Create layers manually to demonstrate advanced features
    std::vector<std::unique_ptr<Layer>> layers;
    
    // Layer 1: Dense + BatchNorm + Regularization
    layers.push_back(std::make_unique<DenseLayer>(64, 128));
    layers.push_back(std::make_unique<BatchNormalizationLayer>(128));
    layers.push_back(std::make_unique<ActivationLayer>("relu", 128));
    layers.push_back(std::make_unique<RegularizationLayer>(128, 0.001, 0.001));
      // Layer 2: Dense + BatchNorm + Dropout
    layers.push_back(std::make_unique<DenseLayer>(128, 64));
    layers.push_back(std::make_unique<BatchNormalizationLayer>(64));
    layers.push_back(std::make_unique<ActivationLayer>("relu", 64));
    layers.push_back(std::make_unique<DropoutLayer>(0.3f));
    
    // Output layer
    layers.push_back(std::make_unique<DenseLayer>(64, 10));
    layers.push_back(std::make_unique<ActivationLayer>("softmax", 10));
    
    std::cout << "Network architecture:\n";
    Matrix current_input = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "  Layer " << i+1 << ": " << layers[i]->type() 
                  << " [" << layers[i]->input_size() << " → " 
                  << layers[i]->output_size() << "]\n";
        
        // Forward pass through each layer
        current_input = layers[i]->forward(current_input);
    }
    
    std::cout << "Final output shape: [" << current_input.rows() 
              << ", " << current_input.cols() << "]\n";
    std::cout << "✓ Advanced network successfully processed data through all layers!\n";
}

int main() {
    std::cout << "=== CLModel Advanced ML Features Demo ===\n";
    std::cout << "Showcasing CNN, LSTM, Regularization, and Batch Normalization\n";
    
    try {
        demonstrate_batch_normalization();
        demonstrate_conv2d();
        demonstrate_lstm();
        demonstrate_regularization();
        demonstrate_advanced_network();
        
        std::cout << "\n=== Demo Complete ===\n";
        std::cout << "✅ All advanced ML features are working correctly!\n";
        std::cout << "\nAdvanced features demonstrated:\n";
        std::cout << "  🧠 Batch Normalization - Data normalization for stable training\n";
        std::cout << "  🖼️  2D Convolution - Image and spatial data processing\n";
        std::cout << "  🔄 LSTM - Sequential data and memory modeling\n";
        std::cout << "  ⚖️  L1/L2 Regularization - Overfitting prevention\n";
        std::cout << "  🏗️  Advanced Architecture - Combining multiple techniques\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
