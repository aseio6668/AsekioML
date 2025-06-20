#include "../include/clmodel.hpp"
#include <iostream>

using namespace clmodel;

int main() {
    std::cout << "Testing Neural Network with Regularization Features\n" << std::endl;
    
    try {
        // Test 1: Basic network without regularization
        std::cout << "=== Test 1: Basic Network ===" << std::endl;
        NeuralNetwork network1;
        network1.add_dense_layer(2, 4, "relu");
        network1.add_dense_layer(4, 1, "sigmoid");
        network1.compile("mse", "sgd", 0.01);
        
        // Create simple input/output data
        Matrix input(1, 2);  // 1 sample, 2 features
        Matrix target(1, 1); // 1 sample, 1 output
        input(0, 0) = 1.0;
        input(0, 1) = 0.5;
        target(0, 0) = 0.8;
        
        Matrix prediction1 = network1.predict(input);
        std::cout << "Basic network prediction: " << prediction1(0, 0) << std::endl;
        
        // Test 2: Network with Dropout        std::cout << "\n=== Test 2: Network with Dropout ===" << std::endl;
        NeuralNetwork network2;
        network2.add_dense_layer(2, 8, "relu");
        std::cout << "After adding dense(2->8) + relu, layers: " << network2.num_layers() << std::endl;
        
        network2.add_dropout_layer(0.3f);  // 30% dropout
        std::cout << "After adding dropout, layers: " << network2.num_layers() << std::endl;
        
        network2.add_dense_layer(8, 4, "relu");
        std::cout << "After adding dense(8->4) + relu, layers: " << network2.num_layers() << std::endl;
        
        network2.add_dropout_layer(0.2f);  // 20% dropout
        std::cout << "After adding dropout, layers: " << network2.num_layers() << std::endl;
        
        network2.add_dense_layer(4, 1, "sigmoid");
        std::cout << "After adding dense(4->1) + sigmoid, layers: " << network2.num_layers() << std::endl;
        std::cout << "About to compile..." << std::endl;
        network2.compile("mse", "adam", 0.01);
        
        // Test in training mode
        network2.set_training_mode(true);
        Matrix prediction2_train = network2.predict(input);
        std::cout << "Dropout network (training mode): " << prediction2_train(0, 0) << std::endl;
        
        // Test in inference mode
        network2.set_training_mode(false);
        Matrix prediction2_inference = network2.predict(input);
        std::cout << "Dropout network (inference mode): " << prediction2_inference(0, 0) << std::endl;
        
        // Test 3: Network with Batch Normalization
        std::cout << "\n=== Test 3: Network with Batch Normalization ===" << std::endl;
        NeuralNetwork network3;
        network3.add_dense_layer(2, 6, "relu");
        network3.add_batch_norm_layer();
        network3.add_dense_layer(6, 3, "relu");
        network3.add_batch_norm_layer();
        network3.add_dense_layer(3, 1, "sigmoid");
        network3.compile("mse", "adam", 0.01);
        
        Matrix prediction3 = network3.predict(input);
        std::cout << "Batch norm network prediction: " << prediction3(0, 0) << std::endl;
        
        // Test 4: Combined regularization
        std::cout << "\n=== Test 4: Combined Regularization ===" << std::endl;
        NeuralNetwork network4;
        network4.add_dense_layer(2, 8, "relu");
        network4.add_batch_norm_layer();
        network4.add_dropout_layer(0.25f);
        network4.add_dense_layer(8, 4, "relu");
        network4.add_batch_norm_layer();
        network4.add_dropout_layer(0.15f);
        network4.add_dense_layer(4, 1, "sigmoid");
        network4.compile("mse", "adam", 0.01);
        
        // Training mode
        network4.set_training_mode(true);
        Matrix prediction4_train = network4.predict(input);
        std::cout << "Combined regularization (training): " << prediction4_train(0, 0) << std::endl;
        
        // Inference mode
        network4.set_training_mode(false);
        Matrix prediction4_inference = network4.predict(input);
        std::cout << "Combined regularization (inference): " << prediction4_inference(0, 0) << std::endl;
        
        // Test 5: Training with regularization
        std::cout << "\n=== Test 5: Training with Regularization ===" << std::endl;
        NeuralNetwork network5;
        network5.add_dense_layer(2, 6, "relu");
        network5.add_dropout_layer(0.2f);
        network5.add_dense_layer(6, 1, "sigmoid");
        network5.compile("mse", "adam", 0.01);
        
        std::cout << "Training network with dropout..." << std::endl;
        network5.set_training_mode(true);
        
        double initial_loss = network5.compute_loss(network5.predict(input), target);
        std::cout << "Initial loss: " << initial_loss << std::endl;
        
        // Train for a few steps
        for (int i = 0; i < 10; ++i) {
            network5.train_step(input, target);
            if (i % 3 == 0) {
                double loss = network5.compute_loss(network5.predict(input), target);
                std::cout << "Step " << i << " loss: " << loss << std::endl;
            }
        }
        
        double final_loss = network5.compute_loss(network5.predict(input), target);
        std::cout << "Final loss: " << final_loss << std::endl;
        
        std::cout << "\n✅ All regularization tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
