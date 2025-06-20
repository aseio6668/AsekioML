#include "../include/clmodel.hpp"
#include <iostream>

using namespace clmodel;

int main() {
    std::cout << "Simple Dropout Test\n" << std::endl;
    
    try {
        // Test standalone dropout layer
        std::cout << "=== Test 1: Standalone Dropout Layer ===" << std::endl;
        DropoutLayer dropout(0.3f);
        
        Matrix input(2, 4);  // 2 samples, 4 features
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                input(i, j) = (i + 1) * (j + 1) * 0.1;
            }
        }
        
        std::cout << "Input shape: [" << input.rows() << ", " << input.cols() << "]" << std::endl;
        
        // Training mode
        dropout.set_training_mode(true);
        Matrix output_train = dropout.forward(input);
        std::cout << "Training output shape: [" << output_train.rows() << ", " << output_train.cols() << "]" << std::endl;
        std::cout << "Input size: " << dropout.input_size() << ", Output size: " << dropout.output_size() << std::endl;
        
        // Inference mode  
        dropout.set_training_mode(false);
        Matrix output_inference = dropout.forward(input);
        std::cout << "Inference output shape: [" << output_inference.rows() << ", " << output_inference.cols() << "]" << std::endl;
        
        std::cout << "\n=== Test 2: Simple Network with Dropout ===" << std::endl;
        NeuralNetwork network;
        network.add_dense_layer(4, 6, "relu");
        std::cout << "Added dense layer 4->6" << std::endl;
        
        network.add_dropout_layer(0.2f);
        std::cout << "Added dropout layer" << std::endl;
        
        network.add_dense_layer(6, 1, "sigmoid");
        std::cout << "Added dense layer 6->1" << std::endl;
        
        network.compile("mse", "sgd", 0.01);
        std::cout << "Compiled network" << std::endl;
        
        Matrix test_input(1, 4);  // 1 sample, 4 features
        for (size_t j = 0; j < 4; ++j) {
            test_input(0, j) = j * 0.1 + 0.1;
        }
        
        Matrix prediction = network.predict(test_input);
        std::cout << "Prediction: " << prediction(0, 0) << std::endl;
        
        std::cout << "\n✅ Dropout tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
