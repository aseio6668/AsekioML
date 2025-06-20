#include "../include/clmodel.hpp"
#include "../include/tensor.hpp"
#include <iostream>
#include <chrono>

using namespace clmodel;
using namespace clmodel::ai;

int main() {
    std::cout << "========================================\n";
    std::cout << "    CLModel AI Phase 1 Foundation Demo\n";
    std::cout << "========================================\n\n";
    
    try {
        // 1. Test basic Tensor functionality
        std::cout << "=== Testing Tensor Basics ===" << std::endl;
        
        // Create tensors
        Tensor t1 = Tensor::zeros({2, 3});
        std::cout << "Zeros tensor (2x3): " << t1 << std::endl;
        
        Tensor t2 = Tensor::ones({2, 3});
        std::cout << "Ones tensor (2x3): " << t2 << std::endl;
        
        Tensor t3 = Tensor::random({2, 3}, 0.0, 1.0);
        std::cout << "Random tensor (2x3): " << t3 << std::endl;
        
        // Test element access
        t1(0, 0) = 5.0;
        t1(1, 2) = 10.0;
        std::cout << "Modified zeros tensor: " << t1 << std::endl;
        
        // Test initializer list constructor
        Tensor t4 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        std::cout << "Tensor from initializer list: " << t4 << std::endl;
        
        // 2. Test arithmetic operations
        std::cout << "\n=== Testing Tensor Arithmetic ===" << std::endl;
        
        Tensor a = {{1.0, 2.0}, {3.0, 4.0}};
        Tensor b = {{5.0, 6.0}, {7.0, 8.0}};
        
        std::cout << "Tensor A: " << a << std::endl;
        std::cout << "Tensor B: " << b << std::endl;
        
        Tensor sum = a + b;
        std::cout << "A + B: " << sum << std::endl;
        
        Tensor diff = a - b;
        std::cout << "A - B: " << diff << std::endl;
        
        Tensor prod = a * b;
        std::cout << "A * B (element-wise): " << prod << std::endl;
        
        Tensor scalar_mult = a * 2.0;
        std::cout << "A * 2.0: " << scalar_mult << std::endl;
        
        // 3. Test matrix multiplication
        std::cout << "\n=== Testing Matrix Operations ===" << std::endl;
        
        Tensor matmul_result = a.matmul(b);
        std::cout << "A @ B (matrix multiplication): " << matmul_result << std::endl;
        
        // Test identity matrix
        Tensor eye = Tensor::eye(3);
        std::cout << "3x3 Identity matrix: " << eye << std::endl;
        
        // 4. Test shape operations
        std::cout << "\n=== Testing Shape Operations ===" << std::endl;
        
        Tensor original = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
        std::cout << "Original (1x6): " << original << std::endl;
        
        Tensor reshaped = original.reshape({2, 3});
        std::cout << "Reshaped to (2x3): " << reshaped << std::endl;
        
        Tensor transposed = reshaped.transpose();
        std::cout << "Transposed (3x2): " << transposed << std::endl;
        
        // 5. Test conversion with existing Matrix class
        std::cout << "\n=== Testing Matrix Integration ===" << std::endl;
        
        Matrix old_matrix(2, 2);
        old_matrix(0, 0) = 1.0; old_matrix(0, 1) = 2.0;
        old_matrix(1, 0) = 3.0; old_matrix(1, 1) = 4.0;
        
        std::cout << "Original Matrix:\n";
        for (size_t i = 0; i < old_matrix.rows(); ++i) {
            for (size_t j = 0; j < old_matrix.cols(); ++j) {
                std::cout << old_matrix(i, j) << " ";
            }
            std::cout << "\n";
        }
        
        Tensor from_matrix = Tensor::from_matrix(old_matrix);
        std::cout << "Tensor from Matrix: " << from_matrix << std::endl;
        
        Matrix back_to_matrix = from_matrix.to_matrix();
        std::cout << "Back to Matrix:\n";
        for (size_t i = 0; i < back_to_matrix.rows(); ++i) {
            for (size_t j = 0; j < back_to_matrix.cols(); ++j) {
                std::cout << back_to_matrix(i, j) << " ";
            }
            std::cout << "\n";
        }
        
        // 6. Test with existing neural network
        std::cout << "\n=== Testing Neural Network Integration ===" << std::endl;
        
        // Create a simple network using existing functionality
        NeuralNetwork network;
        network.add_dense_layer(4, 8, "relu");
        network.add_dense_layer(8, 4, "relu");
        network.add_dense_layer(4, 2, "sigmoid");
        network.compile("mse", "adam", 0.01);
        
        std::cout << "Created network with " << network.num_layers() << " layers" << std::endl;
        std::cout << "Total parameters: " << network.count_parameters() << std::endl;
        
        // Test forward pass with tensor input (converted to Matrix)
        Tensor input_tensor = Tensor::random({1, 4}, 0.0, 1.0);
        std::cout << "Input tensor: " << input_tensor << std::endl;
        
        Matrix input_matrix = input_tensor.to_matrix();
        Matrix output_matrix = network.predict(input_matrix);
        
        Tensor output_tensor = Tensor::from_matrix(output_matrix);
        std::cout << "Output tensor: " << output_tensor << std::endl;
        
        // 7. Performance comparison
        std::cout << "\n=== Performance Testing ===" << std::endl;
        
        size_t matrix_size = 500;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Test Tensor performance
        Tensor large_a = Tensor::random({matrix_size, matrix_size}, 0.0, 1.0);
        Tensor large_b = Tensor::random({matrix_size, matrix_size}, 0.0, 1.0);
        Tensor result_tensor = large_a.matmul(large_b);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Tensor matrix multiplication (" << matrix_size << "x" << matrix_size << "): " 
                  << duration.count() << " ms" << std::endl;
        
        // Compare with Matrix class
        start_time = std::chrono::high_resolution_clock::now();
        
        Matrix large_m1(matrix_size, matrix_size);
        Matrix large_m2(matrix_size, matrix_size);
        
        // Fill with random values
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                large_m1(i, j) = static_cast<double>(rand()) / RAND_MAX;
                large_m2(i, j) = static_cast<double>(rand()) / RAND_MAX;
            }
        }
        
        Matrix result_matrix = large_m1 * large_m2;
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Matrix multiplication (" << matrix_size << "x" << matrix_size << "): " 
                  << duration.count() << " ms" << std::endl;
        
        // 8. Future capability preview
        std::cout << "\n=== Future AI Capabilities (Phase 1 Foundation) ===" << std::endl;
        std::cout << "✅ Multi-dimensional Tensor operations" << std::endl;
        std::cout << "✅ Backward compatibility with existing Matrix API" << std::endl;
        std::cout << "✅ Foundation for CNN layers (Conv2D, Pooling)" << std::endl;
        std::cout << "✅ Efficient shape operations and broadcasting (basic)" << std::endl;
        std::cout << "✅ Integration with existing neural network architecture" << std::endl;
        
        std::cout << "\n🔜 Next Phase: CNN Implementation" << std::endl;
        std::cout << "   - Conv2D layers for image processing" << std::endl;
        std::cout << "   - MaxPool2D and AvgPool2D layers" << std::endl;
        std::cout << "   - Flatten layer for CNN-to-Dense transitions" << std::endl;
        std::cout << "   - Image preprocessing utilities" << std::endl;
        
        std::cout << "\n🎯 CLModel AI Foundation: Successfully Established!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
