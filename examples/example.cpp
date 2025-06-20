#include "clmodel.hpp"
#include <iostream>
#include <chrono>

void demonstrate_regression() {
    std::cout << "=== Regression Example ===" << std::endl;
    
    // Create a synthetic regression dataset
    auto dataset = clmodel::datasets::make_regression(1000, 3, 0.1, 42);
    dataset.normalize_features();
    
    std::cout << "Created regression dataset with " << dataset.size() << " samples" << std::endl;
    
    // Split into train and test
    auto [train_data, test_data] = dataset.train_test_split(0.2);
    
    // Create a neural network for regression
    auto network = clmodel::create_mlp(
        {3, 64, 32, 1},                    // Layer sizes
        {"relu", "relu", "linear"},        // Activations
        "mse",                             // Loss function
        "adam",                            // Optimizer
        0.001                              // Learning rate
    );
    
    network->summary();
    
    // Train the network
    auto start = std::chrono::high_resolution_clock::now();
    network->fit(train_data.features(), train_data.targets(), 100, 32, 0.2, true);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Training completed in " << duration.count() << "ms" << std::endl;
    
    // Evaluate on test set
    double test_loss = network->evaluate(test_data.features(), test_data.targets());
    std::cout << "Test loss: " << test_loss << std::endl;
    
    std::cout << std::endl;
}

void demonstrate_classification() {
    std::cout << "=== Classification Example ===" << std::endl;
    
    // Create a synthetic classification dataset
    auto dataset = clmodel::datasets::make_classification(800, 2, 3, 0.1, 42);
    dataset.normalize_features();
    
    std::cout << "Created classification dataset with " << dataset.size() << " samples" << std::endl;
    
    // Split into train and test
    auto [train_data, test_data] = dataset.train_test_split(0.2);
    
    // Create a neural network for classification
    auto network = clmodel::create_mlp(
        {2, 32, 16, 3},                    // Layer sizes
        {"relu", "relu", "softmax"},       // Activations
        "cross_entropy",                   // Loss function
        "adam",                            // Optimizer
        0.001                              // Learning rate
    );
    
    network->summary();
    
    // Train the network
    auto start = std::chrono::high_resolution_clock::now();
    network->fit(train_data.features(), train_data.targets(), 150, 32, 0.2, true);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Training completed in " << duration.count() << "ms" << std::endl;
    
    // Evaluate on test set
    double test_loss = network->evaluate(test_data.features(), test_data.targets());
    clmodel::Matrix predictions = network->predict(test_data.features());
    double test_accuracy = network->calculate_accuracy(predictions, test_data.targets());
    
    std::cout << "Test loss: " << test_loss << std::endl;
    std::cout << "Test accuracy: " << test_accuracy << std::endl;
    
    std::cout << std::endl;
}

void demonstrate_xor_problem() {
    std::cout << "=== XOR Problem (Non-linear Classification) ===" << std::endl;
    
    // Create XOR dataset
    auto dataset = clmodel::datasets::make_xor(400, 0.05);
    
    std::cout << "Created XOR dataset with " << dataset.size() << " samples" << std::endl;
    
    // Split into train and test
    auto [train_data, test_data] = dataset.train_test_split(0.3);
    
    // Create a neural network that can solve XOR
    clmodel::NeuralNetwork network;
    network.add_dense_layer(2, 8, "relu");
    network.add_dense_layer(8, 4, "relu");
    network.add_dense_layer(4, 1, "sigmoid");
    network.compile("binary_cross_entropy", "adam", 0.01);
    
    network.summary();
    
    // Train the network
    auto start = std::chrono::high_resolution_clock::now();
    network.fit(train_data.features(), train_data.targets(), 200, 16, 0.0, true);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Training completed in " << duration.count() << "ms" << std::endl;
    
    // Evaluate on test set
    double test_loss = network.evaluate(test_data.features(), test_data.targets());
    clmodel::Matrix predictions = network.predict(test_data.features());
    double test_accuracy = network.calculate_accuracy(predictions, test_data.targets());
    
    std::cout << "Test loss: " << test_loss << std::endl;
    std::cout << "Test accuracy: " << test_accuracy << std::endl;
    
    // Test on specific XOR inputs
    std::cout << "\\nTesting specific XOR combinations:" << std::endl;
    clmodel::Matrix test_inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    clmodel::Matrix test_outputs = network.predict(test_inputs);
    
    for (size_t i = 0; i < test_inputs.rows(); ++i) {
        std::cout << "Input: (" << test_inputs[i][0] << ", " << test_inputs[i][1] 
                  << ") -> Output: " << test_outputs[i][0] 
                  << " (Expected: " << ((static_cast<int>(test_inputs[i][0]) ^ static_cast<int>(test_inputs[i][1])) ? "1" : "0") 
                  << ")" << std::endl;
    }
    
    std::cout << std::endl;
}

void demonstrate_matrix_operations() {
    std::cout << "=== Matrix Operations Demo ===" << std::endl;
    
    // Create matrices
    clmodel::Matrix A = {{1, 2, 3}, {4, 5, 6}};
    clmodel::Matrix B = {{7, 8}, {9, 10}, {11, 12}};
    
    std::cout << "Matrix A:" << std::endl << A << std::endl;
    std::cout << "Matrix B:" << std::endl << B << std::endl;
    
    // Matrix multiplication
    clmodel::Matrix C = A * B;
    std::cout << "A * B:" << std::endl << C << std::endl;
    
    // Element-wise operations
    clmodel::Matrix D = A + A;
    std::cout << "A + A:" << std::endl << D << std::endl;
    
    // Transpose
    clmodel::Matrix At = A.transpose();
    std::cout << "A transpose:" << std::endl << At << std::endl;
    
    // Random matrix
    clmodel::Matrix random = clmodel::Matrix::random(3, 3, -1.0, 1.0);
    std::cout << "Random 3x3 matrix:" << std::endl << random << std::endl;
    
    // Apply function
    clmodel::Matrix sigmoid_applied = random.apply([](double x) { 
        return 1.0 / (1.0 + std::exp(-x)); 
    });
    std::cout << "Sigmoid applied to random matrix:" << std::endl << sigmoid_applied << std::endl;
    
    std::cout << std::endl;
}

void demonstrate_optimizers() {
    std::cout << "=== Optimizer Comparison ===" << std::endl;
    
    // Create the same dataset for all optimizers
    auto dataset = clmodel::datasets::make_regression(500, 2, 0.1, 42);
    dataset.normalize_features();
    auto [train_data, test_data] = dataset.train_test_split(0.2);
    
    std::vector<std::string> optimizers = {"sgd", "adam", "rmsprop", "adagrad"};
    
    for (const auto& opt_name : optimizers) {
        std::cout << "Testing " << opt_name << " optimizer:" << std::endl;
        
        auto network = clmodel::create_mlp(
            {2, 32, 16, 1},
            {"relu", "relu", "linear"},
            "mse",
            opt_name,
            0.01
        );
        
        auto start = std::chrono::high_resolution_clock::now();
        network->fit(train_data.features(), train_data.targets(), 50, 32, 0.0, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double test_loss = network->evaluate(test_data.features(), test_data.targets());
        
        std::cout << "  Training time: " << duration.count() << "ms" << std::endl;
        std::cout << "  Final test loss: " << test_loss << std::endl;
        
        const auto& history = network->get_history();
        if (!history.training_loss.empty()) {
            std::cout << "  Final training loss: " << history.training_loss.back() << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "CLModel Framework Demo" << std::endl;
    std::cout << "Version: " << clmodel::version() << std::endl;
    std::cout << "======================" << std::endl << std::endl;
    
    try {
        demonstrate_matrix_operations();
        demonstrate_regression();
        demonstrate_classification();
        demonstrate_xor_problem();
        demonstrate_optimizers();
        
        std::cout << "All demonstrations completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
