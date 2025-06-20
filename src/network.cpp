#include "network.hpp"
#include "regularization.hpp"
#include "model_serialization.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>

namespace clmodel {

NeuralNetwork::NeuralNetwork() : compiled_(false) {}

// Copy constructor
NeuralNetwork::NeuralNetwork(const NeuralNetwork& other) : compiled_(false) {
    copy_from(other);
}

// Move constructor
NeuralNetwork::NeuralNetwork(NeuralNetwork&& other) noexcept
    : layers_(std::move(other.layers_)),
      loss_function_(std::move(other.loss_function_)),
      optimizer_(std::move(other.optimizer_)),
      history_(std::move(other.history_)),
      compiled_(other.compiled_) {
    other.compiled_ = false;
}

// Copy assignment operator
NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other) {
    if (this != &other) {
        copy_from(other);
    }
    return *this;
}

// Move assignment operator
NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&& other) noexcept {
    if (this != &other) {
        layers_ = std::move(other.layers_);
        loss_function_ = std::move(other.loss_function_);
        optimizer_ = std::move(other.optimizer_);
        history_ = std::move(other.history_);
        compiled_ = other.compiled_;
        other.compiled_ = false;
    }
    return *this;
}

void NeuralNetwork::copy_from(const NeuralNetwork& other) {
    layers_.clear();
    for (const auto& layer : other.layers_) {
        layers_.push_back(layer->clone());
    }
    
    if (other.loss_function_) {
        loss_function_ = other.loss_function_->clone();
    }
    
    if (other.optimizer_) {
        optimizer_ = other.optimizer_->clone();
    }
    
    history_ = other.history_;
    compiled_ = other.compiled_;
}

// Network building
void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    // If this isn't the first layer, set its input size to match the previous layer's output size
    if (!layers_.empty()) {
        size_t previous_output_size = layers_.back()->output_size();
          // For layers that need input size configuration
        if (auto dropout_layer = dynamic_cast<DropoutLayer*>(layer.get())) {
            dropout_layer->set_input_size(previous_output_size);
        } else if (auto batch_norm_layer = dynamic_cast<BatchNormLayer*>(layer.get())) {
            batch_norm_layer->set_input_size(previous_output_size);
        }
    }
    
    layers_.push_back(std::move(layer));
    compiled_ = false; // Need to recompile after adding layers
}

void NeuralNetwork::add_dense_layer(size_t input_size, size_t output_size, const std::string& activation) {
    auto dense_layer = std::make_unique<DenseLayer>(input_size, output_size);
    add_layer(std::move(dense_layer));
    
    if (activation != "linear") {
        auto activation_layer = std::make_unique<ActivationLayer>(activation, output_size);
        add_layer(std::move(activation_layer));
    }
}

void NeuralNetwork::add_activation_layer(const std::string& activation, size_t size) {
    auto activation_layer = std::make_unique<ActivationLayer>(activation, size);
    add_layer(std::move(activation_layer));
}

void NeuralNetwork::add_dropout_layer(float dropout_rate) {
    auto dropout_layer = std::make_unique<DropoutLayer>(dropout_rate);
    this->add_layer(std::move(dropout_layer));
}

void NeuralNetwork::add_batch_norm_layer(float momentum, float epsilon) {
    auto batch_norm_layer = std::make_unique<BatchNormLayer>(momentum, epsilon);
    this->add_layer(std::move(batch_norm_layer));
}

// Compilation
void NeuralNetwork::compile(const std::string& loss_function, const std::string& optimizer, double learning_rate) {
    loss_function_ = create_loss_function(loss_function);
    optimizer_ = create_optimizer(optimizer, learning_rate);
    compiled_ = true;
    validate_network_structure();
}

void NeuralNetwork::compile(std::unique_ptr<LossFunction> loss_function, std::unique_ptr<Optimizer> optimizer) {
    loss_function_ = std::move(loss_function);
    optimizer_ = std::move(optimizer);
    compiled_ = true;
    validate_network_structure();
}

// Compilation (additional methods for modern API)
void NeuralNetwork::set_loss_function(const std::string& loss_function) {
    loss_function_ = create_loss_function(loss_function);
}

void NeuralNetwork::set_optimizer(const std::string& optimizer, double learning_rate) {
    optimizer_ = create_optimizer(optimizer, learning_rate);
    compiled_ = true;
    validate_network_structure();
}

// Forward pass
Matrix NeuralNetwork::forward(const Matrix& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    Matrix current_output = input;
    for (auto& layer : layers_) {
        current_output = layer->forward(current_output);
    }
    
    return current_output;
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    set_training_mode(false); // Set to inference mode
    Matrix result = forward(input);
    set_training_mode(true);  // Reset to training mode
    return result;
}

// Training
void NeuralNetwork::train_step(const Matrix& input, const Matrix& target) {
    if (!compiled_) {
        throw std::runtime_error("Network must be compiled before training");
    }
    
    // Forward pass
    Matrix predictions = forward(input);
    
    // Compute loss gradient
    Matrix loss_gradient = loss_function_->compute_gradient(predictions, target);
    
    // Backward pass
    Matrix current_gradient = loss_gradient;
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        current_gradient = layers_[i]->backward(current_gradient);
    }
    
    // Update weights using optimizer
    for (auto& layer : layers_) {
        if (auto dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
            // For dense layers, we need access to gradients
            // This is a simplified approach - in practice, you'd want better gradient handling
            layer->update_weights(0.01); // Use a default learning rate for now
        }
    }
}

void NeuralNetwork::fit(const Matrix& X_train, const Matrix& y_train, 
                       int epochs, int batch_size, double validation_split, bool verbose) {
    if (!compiled_) {
        throw std::runtime_error("Network must be compiled before training");
    }
    
    if (validation_split > 0.0) {
        auto [train_split, val_split] = train_validation_split(X_train, y_train, validation_split);
        // Extract X and y from the split datasets
        Matrix X_train_split(train_split.first.rows(), train_split.first.cols());
        Matrix y_train_split(train_split.second.rows(), train_split.second.cols());
        Matrix X_val(val_split.first.rows(), val_split.first.cols());
        Matrix y_val(val_split.second.rows(), val_split.second.cols());
        
        // Copy data
        for (size_t i = 0; i < train_split.first.rows(); ++i) {
            for (size_t j = 0; j < train_split.first.cols(); ++j) {
                X_train_split[i][j] = train_split.first[i][j];
            }
        }
        for (size_t i = 0; i < train_split.second.rows(); ++i) {
            for (size_t j = 0; j < train_split.second.cols(); ++j) {
                y_train_split[i][j] = train_split.second[i][j];
            }
        }
        for (size_t i = 0; i < val_split.first.rows(); ++i) {
            for (size_t j = 0; j < val_split.first.cols(); ++j) {
                X_val[i][j] = val_split.first[i][j];
            }
        }
        for (size_t i = 0; i < val_split.second.rows(); ++i) {
            for (size_t j = 0; j < val_split.second.cols(); ++j) {
                y_val[i][j] = val_split.second[i][j];
            }
        }
        
        fit(X_train_split, y_train_split, X_val, y_val, epochs, batch_size, verbose);
    } else {
        fit(X_train, y_train, Matrix(), Matrix(), epochs, batch_size, verbose);
    }
}

void NeuralNetwork::fit(const Matrix& X_train, const Matrix& y_train,
                       const Matrix& X_val, const Matrix& y_val,
                       int epochs, int batch_size, bool verbose) {
    if (!compiled_) {
        throw std::runtime_error("Network must be compiled before training");
    }
    
    bool has_validation = (X_val.rows() > 0 && y_val.rows() > 0);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Training
        double total_loss = 0.0;
        int num_batches = static_cast<int>((X_train.rows() + batch_size - 1) / batch_size);
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, static_cast<int>(X_train.rows()));
            
            // Create batch (simplified - in practice, you'd want better batch handling)
            Matrix X_batch(end_idx - start_idx, X_train.cols());
            Matrix y_batch(end_idx - start_idx, y_train.cols());
            
            for (int i = 0; i < end_idx - start_idx; ++i) {
                for (size_t j = 0; j < X_train.cols(); ++j) {
                    X_batch[i][j] = X_train[start_idx + i][j];
                }
                for (size_t j = 0; j < y_train.cols(); ++j) {
                    y_batch[i][j] = y_train[start_idx + i][j];
                }
            }
            
            // Train on batch
            Matrix predictions = forward(X_batch);
            double batch_loss = loss_function_->compute_loss(predictions, y_batch);
            total_loss += batch_loss;
            
            train_step(X_batch, y_batch);
        }
        
        double avg_training_loss = total_loss / num_batches;
        history_.training_loss.push_back(avg_training_loss);
        
        // Calculate training accuracy
        Matrix train_predictions = predict(X_train);
        double train_accuracy = calculate_accuracy(train_predictions, y_train);
        history_.training_accuracy.push_back(train_accuracy);
        
        // Validation
        double val_loss = 0.0;
        double val_accuracy = 0.0;
        if (has_validation) {
            Matrix val_predictions = predict(X_val);
            val_loss = loss_function_->compute_loss(val_predictions, y_val);
            val_accuracy = calculate_accuracy(val_predictions, y_val);
            history_.validation_loss.push_back(val_loss);
            history_.validation_accuracy.push_back(val_accuracy);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (verbose) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs;
            std::cout << " - " << duration.count() << "ms";
            std::cout << " - loss: " << std::fixed << std::setprecision(4) << avg_training_loss;
            std::cout << " - accuracy: " << std::setprecision(4) << train_accuracy;
            
            if (has_validation) {
                std::cout << " - val_loss: " << std::setprecision(4) << val_loss;
                std::cout << " - val_accuracy: " << std::setprecision(4) << val_accuracy;
            }
            std::cout << std::endl;
        }
    }
}

// Evaluation
double NeuralNetwork::evaluate(const Matrix& X_test, const Matrix& y_test, int /*batch_size*/) {
    if (!compiled_) {
        throw std::runtime_error("Network must be compiled before evaluation");
    }
    
    Matrix predictions = predict(X_test);
    return loss_function_->compute_loss(predictions, y_test);
}

double NeuralNetwork::calculate_accuracy(const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    int correct = 0;
    int total = static_cast<int>(predictions.rows());
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        // For classification: find the class with highest probability
        if (predictions.cols() > 1) {
            size_t pred_class = 0;
            size_t true_class = 0;
            
            for (size_t j = 1; j < predictions.cols(); ++j) {
                if (predictions[i][j] > predictions[i][pred_class]) {
                    pred_class = j;
                }
                if (targets[i][j] > targets[i][true_class]) {
                    true_class = j;
                }
            }
            
            if (pred_class == true_class) {
                correct++;
            }
        } else {
            // For binary classification or regression
            double pred = predictions[i][0] > 0.5 ? 1.0 : 0.0;
            double target = targets[i][0] > 0.5 ? 1.0 : 0.0;
            if (pred == target) {
                correct++;
            }
        }
    }
    
    return static_cast<double>(correct) / total;
}

// Network information
void NeuralNetwork::summary() const {
    std::cout << "Model Summary:" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left << std::setw(20) << "Layer (type)";
    std::cout << std::setw(20) << "Output Shape";
    std::cout << std::setw(15) << "Param #" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    size_t total_params = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        std::cout << std::setw(20) << layer->type();
        std::cout << std::setw(20) << ("(" + std::to_string(layer->output_size()) + ",)");
        
        size_t layer_params = 0;
        if (auto dense_layer = dynamic_cast<const DenseLayer*>(layer.get())) {
            layer_params = dense_layer->input_size() * dense_layer->output_size() + dense_layer->output_size();
        }
        
        std::cout << std::setw(15) << layer_params << std::endl;
        total_params += layer_params;
    }
    
    std::cout << "================================================================" << std::endl;
    std::cout << "Total params: " << total_params << std::endl;
    std::cout << "================================================================" << std::endl;
}

size_t NeuralNetwork::count_parameters() const {
    size_t total = 0;
    for (const auto& layer : layers_) {
        if (auto dense_layer = dynamic_cast<const DenseLayer*>(layer.get())) {
            total += dense_layer->input_size() * dense_layer->output_size() + dense_layer->output_size();
        }
    }
    return total;
}

// Utility functions
void NeuralNetwork::set_training_mode(bool training) {
    for (auto& layer : layers_) {
        // Check for regularization layers that need training mode
        if (auto dropout_layer = dynamic_cast<DropoutLayer*>(layer.get())) {
            dropout_layer->set_training_mode(training);
        } else if (auto batch_norm_layer = dynamic_cast<BatchNormLayer*>(layer.get())) {
            batch_norm_layer->set_training_mode(training);
        }
    }
}

void NeuralNetwork::reset_optimizer() {
    if (optimizer_) {
        optimizer_->reset();
    }
}

void NeuralNetwork::clear_history() {
    history_ = TrainingHistory();
}

void NeuralNetwork::validate_network_structure() const {
    if (layers_.empty()) {
        throw std::runtime_error("Network must have at least one layer");
    }
    
    // Check layer compatibility
    for (size_t i = 1; i < layers_.size(); ++i) {
        if (layers_[i-1]->output_size() != layers_[i]->input_size()) {
            throw std::runtime_error("Incompatible layer sizes at layer " + std::to_string(i));
        }
    }
}

std::pair<std::pair<Matrix, Matrix>, std::pair<Matrix, Matrix>> NeuralNetwork::train_validation_split(const Matrix& X, const Matrix& y, double validation_split) const {
    size_t split_idx = static_cast<size_t>((1.0 - validation_split) * X.rows());
    
    Matrix X_train(split_idx, X.cols());
    Matrix y_train(split_idx, y.cols());
    Matrix X_val(X.rows() - split_idx, X.cols());
    Matrix y_val(X.rows() - split_idx, y.cols());
    
    for (size_t i = 0; i < split_idx; ++i) {
        for (size_t j = 0; j < X.cols(); ++j) {
            X_train[i][j] = X[i][j];
        }
        for (size_t j = 0; j < y.cols(); ++j) {
            y_train[i][j] = y[i][j];
        }
    }
    
    for (size_t i = split_idx; i < X.rows(); ++i) {
        for (size_t j = 0; j < X.cols(); ++j) {
            X_val[i - split_idx][j] = X[i][j];
        }
        for (size_t j = 0; j < y.cols(); ++j) {
            y_val[i - split_idx][j] = y[i][j];
        }
    }
    
    return {std::make_pair(X_train, y_train), std::make_pair(X_val, y_val)};
}

// Simplified save/load functions
void NeuralNetwork::save_weights(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    for (const auto& layer : layers_) {
        if (auto dense_layer = dynamic_cast<const DenseLayer*>(layer.get())) {
            const auto& weights = dense_layer->weights();
            const auto& biases = dense_layer->biases();
            
            file << "DENSE_LAYER\n";
            file << weights.rows() << " " << weights.cols() << "\n";
            for (size_t i = 0; i < weights.rows(); ++i) {
                for (size_t j = 0; j < weights.cols(); ++j) {
                    file << weights[i][j] << " ";
                }
                file << "\n";
            }
            
            file << biases.rows() << " " << biases.cols() << "\n";
            for (size_t i = 0; i < biases.rows(); ++i) {
                for (size_t j = 0; j < biases.cols(); ++j) {
                    file << biases[i][j] << " ";
                }
                file << "\n";
            }
        }
    }
}

void NeuralNetwork::load_weights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // This is a simplified implementation
    // In practice, you'd want more robust serialization
    std::string layer_type;
    size_t layer_idx = 0;
    
    while (file >> layer_type && layer_idx < layers_.size()) {
        if (layer_type == "DENSE_LAYER") {
            if (auto dense_layer = dynamic_cast<DenseLayer*>(layers_[layer_idx].get())) {
                // Load weights and biases
                // Implementation would go here
                layer_idx++;
            }
        }
    }
}

// Learning rate control
void NeuralNetwork::set_learning_rate(double lr) {
    if (optimizer_) {
        optimizer_->set_learning_rate(lr);
    }
}

double NeuralNetwork::get_learning_rate() const {
    if (optimizer_) {
        return optimizer_->get_learning_rate();
    }
    return 0.0;
}

// Loss computation
double NeuralNetwork::compute_loss(const Matrix& predictions, const Matrix& targets) {
    if (!loss_function_) {
        throw std::runtime_error("Loss function not set");
    }
    return loss_function_->compute_loss(predictions, targets);
}

// Model Serialization
bool NeuralNetwork::save(const std::string& filepath, 
                        bool include_optimizer, 
                        bool include_history) const {
    return ModelSerializer::save(*this, filepath, SerializationFormat::HYBRID, 
                                include_optimizer, include_history);
}

bool NeuralNetwork::load(const std::string& filepath) {
    auto loaded_network = ModelSerializer::load(filepath, SerializationFormat::HYBRID);
    if (!loaded_network) {
        return false;
    }
    
    // Copy the loaded network to this instance
    *this = std::move(*loaded_network);
    return true;
}

// Export/Import architecture
std::string NeuralNetwork::export_architecture() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"clmodel_version\": \"1.0.0\",\n";
    ss << "  \"num_layers\": " << layers_.size() << ",\n";
    ss << "  \"layers\": [\n";
    
    for (size_t i = 0; i < layers_.size(); ++i) {
        ss << "    " << layers_[i]->serialize_to_json();
        if (i < layers_.size() - 1) ss << ",";
        ss << "\n";
    }
    
    ss << "  ]\n";
    ss << "}";
    
    return ss.str();
}

bool NeuralNetwork::import_architecture(const std::string& architecture_json) {
    // For now, return false as this would require full JSON parsing
    // In a production system, this would parse the JSON and reconstruct the network
    (void)architecture_json;
    return false;
}

// Model information
std::string NeuralNetwork::get_model_info() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"num_layers\": " << layers_.size() << ",\n";
    ss << "  \"compiled\": " << (compiled_ ? "true" : "false") << ",\n";
    ss << "  \"total_parameters\": " << count_parameters();
    
    if (compiled_) {
        if (loss_function_) {
            ss << ",\n  \"loss_function\": \"" << loss_function_->name() << "\"";
        }
        if (optimizer_) {
            ss << ",\n  \"optimizer\": \"" << optimizer_->name() << "\"";
            ss << ",\n  \"learning_rate\": " << get_learning_rate();
        }
    }
    
    ss << "\n}";
    return ss.str();
}

} // namespace clmodel
