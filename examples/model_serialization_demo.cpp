#include "../include/clmodel.hpp"
#include "../include/model_serialization.hpp"
#include <iostream>
#include <filesystem>

using namespace clmodel;

int main() {
    std::cout << "========================================\n";
    std::cout << "    Model Serialization Demo\n";
    std::cout << "========================================\n\n";
    
    try {
        // Create a test network with various layer types
        std::cout << "=== Creating Test Network ===" << std::endl;
        NeuralNetwork network;
        
        // Add layers with different types for comprehensive testing
        network.add_dense_layer(4, 8, "relu");
        network.add_dropout_layer(0.3f);
        network.add_dense_layer(8, 6, "relu");
        network.add_batch_norm_layer();
        network.add_dropout_layer(0.2f);
        network.add_dense_layer(6, 3, "relu");
        network.add_dense_layer(3, 1, "sigmoid");
        
        // Compile the network
        network.compile("mse", "adam", 0.001);
        
        std::cout << "Network created with " << network.num_layers() << " layers" << std::endl;
        std::cout << "Total parameters: " << network.count_parameters() << std::endl;
        
        // Create some training data
        std::cout << "\n=== Training Network ===" << std::endl;
        Matrix X_train(100, 4);  // 100 samples, 4 features
        Matrix y_train(100, 1);  // 100 samples, 1 output
        
        // Generate simple synthetic data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        for (size_t i = 0; i < X_train.rows(); ++i) {
            for (size_t j = 0; j < X_train.cols(); ++j) {
                X_train(i, j) = dis(gen);
            }
            // Simple target: sum of features > 0 ? 1 : 0
            double sum = 0;
            for (size_t j = 0; j < X_train.cols(); ++j) {
                sum += X_train(i, j);
            }
            y_train(i, 0) = sum > 0 ? 1.0 : 0.0;
        }
        
        // Train for a few epochs
        std::cout << "Training for 5 epochs..." << std::endl;
        network.set_training_mode(true);
        
        for (int epoch = 0; epoch < 5; ++epoch) {
            double epoch_loss = 0.0;
            for (size_t i = 0; i < X_train.rows(); ++i) {
                Matrix input(1, X_train.cols());
                Matrix target(1, 1);
                
                for (size_t j = 0; j < X_train.cols(); ++j) {
                    input(0, j) = X_train(i, j);
                }
                target(0, 0) = y_train(i, 0);
                
                network.train_step(input, target);
                
                Matrix pred = network.predict(input);
                epoch_loss += network.compute_loss(pred, target);
            }
            
            std::cout << "Epoch " << (epoch + 1) << " - Loss: " 
                      << (epoch_loss / X_train.rows()) << std::endl;
        }
        
        // Test prediction before saving
        std::cout << "\n=== Testing Before Save ===" << std::endl;
        Matrix test_input(1, 4);
        test_input(0, 0) = 0.5; test_input(0, 1) = -0.3; 
        test_input(0, 2) = 0.8; test_input(0, 3) = -0.1;
        
        network.set_training_mode(false);
        Matrix prediction_before = network.predict(test_input);
        std::cout << "Prediction before save: " << prediction_before(0, 0) << std::endl;
        
        // Get model info
        std::cout << "\nModel info:\n" << network.get_model_info() << std::endl;
        
        // Test all serialization formats
        std::cout << "\n=== Testing Model Serialization ===" << std::endl;
        
        // Test 1: Hybrid format (recommended)
        std::cout << "\n--- Testing Hybrid Format ---" << std::endl;
        std::string hybrid_path = "test_model_hybrid.clmodel";
        bool save_success = network.save(hybrid_path, true, true);
        std::cout << "Hybrid format save: " << (save_success ? "SUCCESS" : "FAILED") << std::endl;
        
        if (save_success) {
            // Verify files were created
            if (std::filesystem::exists(hybrid_path) && 
                std::filesystem::exists(hybrid_path + ".arch") &&
                std::filesystem::exists(hybrid_path + ".weights")) {
                std::cout << "All hybrid format files created successfully" << std::endl;
                
                // Get model info from file
                std::string file_info = ModelSerializer::get_model_info(hybrid_path);
                std::cout << "File info: " << file_info << std::endl;
                
                // Verify model file
                bool verify_success = ModelSerializer::verify_model_file(hybrid_path);
                std::cout << "File verification: " << (verify_success ? "PASSED" : "FAILED") << std::endl;
            }
        }
        
        // Test 2: Binary format
        std::cout << "\n--- Testing Binary Format ---" << std::endl;
        std::string binary_path = "test_model_binary.clmodel";
        save_success = ModelSerializer::save(network, binary_path, SerializationFormat::BINARY);
        std::cout << "Binary format save: " << (save_success ? "SUCCESS" : "FAILED") << std::endl;
        
        if (save_success) {
            std::string file_info = ModelSerializer::get_model_info(binary_path);
            std::cout << "Binary file info: " << file_info << std::endl;
        }
        
        // Test 3: JSON format
        std::cout << "\n--- Testing JSON Format ---" << std::endl;
        std::string json_path = "test_model_json.clmodel";
        save_success = ModelSerializer::save(network, json_path, SerializationFormat::JSON);
        std::cout << "JSON format save: " << (save_success ? "SUCCESS" : "FAILED") << std::endl;
        
        // Test architecture export
        std::cout << "\n--- Testing Architecture Export ---" << std::endl;
        std::string architecture = network.export_architecture();
        std::cout << "Architecture JSON:\n" << architecture << std::endl;
        
        // Test loading
        std::cout << "\n=== Testing Model Loading ===" << std::endl;
        
        // Load the hybrid format model
        NeuralNetwork loaded_network;
        bool load_success = loaded_network.load(hybrid_path);
        std::cout << "Hybrid format load: " << (load_success ? "SUCCESS" : "FAILED") << std::endl;
        
        if (load_success) {
            std::cout << "Loaded network has " << loaded_network.num_layers() << " layers" << std::endl;
            
            // Test prediction with loaded model
            if (loaded_network.is_compiled()) {
                loaded_network.set_training_mode(false);
                Matrix prediction_after = loaded_network.predict(test_input);
                std::cout << "Prediction after load: " << prediction_after(0, 0) << std::endl;
                
                double prediction_diff = std::abs(prediction_before(0, 0) - prediction_after(0, 0));
                std::cout << "Prediction difference: " << prediction_diff << std::endl;
                
                if (prediction_diff < 1e-6) {
                    std::cout << "✅ Predictions match perfectly!" << std::endl;
                } else if (prediction_diff < 1e-3) {
                    std::cout << "✅ Predictions match with minor differences" << std::endl;
                } else {
                    std::cout << "⚠️  Significant difference in predictions" << std::endl;
                }
            } else {
                std::cout << "⚠️  Loaded network is not compiled" << std::endl;
            }
        }
          // Test SerializationCheckpoint
        std::cout << "\n=== Testing SerializationCheckpoint ===" << std::endl;
        SerializationCheckpoint checkpoint("checkpoint_epoch_{epoch}_loss_{loss:.4f}.clmodel", 
                                 "loss", true, false, 1, SerializationFormat::HYBRID);
        
        // Simulate training with checkpoints
        for (int epoch = 0; epoch < 3; ++epoch) {
            std::map<std::string, double> metrics;
            metrics["loss"] = 0.5 - epoch * 0.1;  // Simulated improving loss
            metrics["accuracy"] = 0.6 + epoch * 0.1;  // Simulated improving accuracy
            
            bool saved = checkpoint.on_epoch_end(network, epoch + 1, metrics);
            std::cout << "Epoch " << (epoch + 1) << " checkpoint: " 
                      << (saved ? "SAVED" : "NOT SAVED") 
                      << " (loss: " << metrics["loss"] << ")" << std::endl;
        }
        
        std::cout << "Best metric achieved: " << checkpoint.get_best_metric() << std::endl;
        
        // Clean up test files
        std::cout << "\n=== Cleaning Up Test Files ===" << std::endl;
        std::vector<std::string> test_files = {
            hybrid_path, hybrid_path + ".arch", hybrid_path + ".weights",
            binary_path, json_path
        };
        
        for (const auto& file : test_files) {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
                std::cout << "Removed: " << file << std::endl;
            }
        }
        
        // Clean up checkpoint files
        for (int epoch = 1; epoch <= 3; ++epoch) {
            std::string checkpoint_file = "checkpoint_epoch_" + std::to_string(epoch) + "_loss_";
            // Look for files starting with this pattern
            for (const auto& entry : std::filesystem::directory_iterator(".")) {
                if (entry.path().filename().string().find(checkpoint_file) == 0) {
                    std::filesystem::remove(entry.path());
                    std::cout << "Removed: " << entry.path().filename() << std::endl;
                }
            }
        }
        
        std::cout << "\n🎉 Model Serialization Demo Completed Successfully! 🎉" << std::endl;
        std::cout << "\nFeatures demonstrated:" << std::endl;
        std::cout << "✅ Hybrid format serialization (JSON + Binary)" << std::endl;
        std::cout << "✅ Binary format serialization" << std::endl;
        std::cout << "✅ JSON format serialization" << std::endl;
        std::cout << "✅ Model loading and verification" << std::endl;
        std::cout << "✅ Prediction consistency after save/load" << std::endl;
        std::cout << "✅ Architecture export" << std::endl;
        std::cout << "✅ Model checkpointing during training" << std::endl;
        std::cout << "✅ File verification and metadata extraction" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
