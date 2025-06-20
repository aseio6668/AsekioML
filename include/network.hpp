#pragma once

#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "training_history.hpp"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace clmodel {

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<LossFunction> loss_function_;
    std::unique_ptr<Optimizer> optimizer_;
    TrainingHistory history_;
    bool compiled_;

public:
    NeuralNetwork();
    ~NeuralNetwork() = default;
    
    // Copy and move operations
    NeuralNetwork(const NeuralNetwork& other);
    NeuralNetwork(NeuralNetwork&& other) noexcept;
    NeuralNetwork& operator=(const NeuralNetwork& other);
    NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;
    
    // Network building
    void add_layer(std::unique_ptr<Layer> layer);
    void add_dense_layer(size_t input_size, size_t output_size, const std::string& activation = "linear");
    void add_activation_layer(const std::string& activation, size_t size);
    void add_dropout_layer(float dropout_rate = 0.5f);
    void add_batch_norm_layer(float momentum = 0.9f, float epsilon = 1e-5f);
    
    // Compilation
    void compile(const std::string& loss_function, const std::string& optimizer, double learning_rate = 0.01);
    void compile(std::unique_ptr<LossFunction> loss_function, std::unique_ptr<Optimizer> optimizer);
    
    // Compilation (additional methods for modern API)
    void set_loss_function(const std::string& loss_function);
    void set_optimizer(const std::string& optimizer, double learning_rate = 0.01);
    
    // Prediction
    Matrix forward(const Matrix& input);
    Matrix predict(const Matrix& input);
    
    // Training
    void train_step(const Matrix& input, const Matrix& target);
    void fit(const Matrix& X_train, const Matrix& y_train, 
             int epochs, int batch_size = 32, double validation_split = 0.0,
             bool verbose = true);
    void fit(const Matrix& X_train, const Matrix& y_train,
             const Matrix& X_val, const Matrix& y_val,
             int epochs, int batch_size = 32, bool verbose = true);
    
    // Evaluation
    double evaluate(const Matrix& X_test, const Matrix& y_test, int batch_size = 32);
    double calculate_accuracy(const Matrix& predictions, const Matrix& targets);
    
    // Getters
    const TrainingHistory& get_history() const { return history_; }
    size_t num_layers() const { return layers_.size(); }
    bool is_compiled() const { return compiled_; }
    
    // Learning rate control
    void set_learning_rate(double lr);
    double get_learning_rate() const;
    
    // Loss computation
    double compute_loss(const Matrix& predictions, const Matrix& targets);
    
    // Network information
    void summary() const;
    size_t count_parameters() const;
    
    // Utility
    void set_training_mode(bool training);
    void reset_optimizer();
    void clear_history();
    
    // Save/Load (simplified version)
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
    
    // Model Serialization (comprehensive)
    bool save(const std::string& filepath, 
              bool include_optimizer = true, 
              bool include_history = true) const;
    bool load(const std::string& filepath);
    
    // Export/Import architecture only
    std::string export_architecture() const;
    bool import_architecture(const std::string& architecture_json);
      // Model information
    std::string get_model_info() const;
    std::map<std::string, std::string> get_layer_info() const;
    
    // Layer access for serialization
    const std::vector<std::unique_ptr<Layer>>& get_layers() const { return layers_; }
    const LossFunction* get_loss_function() const { return loss_function_.get(); }
    const Optimizer* get_optimizer() const { return optimizer_.get(); }

private:
    void validate_network_structure() const;
    std::vector<Matrix> create_batches(const Matrix& data, int batch_size) const;
    std::pair<std::pair<Matrix, Matrix>, std::pair<Matrix, Matrix>> train_validation_split(const Matrix& X, const Matrix& y, double validation_split) const;
    void copy_from(const NeuralNetwork& other);
};

} // namespace clmodel
