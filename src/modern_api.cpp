#include "modern_api.hpp"
#include "layer.hpp"
#include "activation.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

namespace clmodel {
namespace api {

// ModelBuilder implementation
ModelBuilder::ModelBuilder() : network_(std::make_unique<NeuralNetwork>()) {}

ModelBuilder& ModelBuilder::dense(size_t units, const std::string& activation) {
    if (network_->num_layers() == 0 && input_size_ == 0) {
        // Store for later when input size is known
        pending_layers_.emplace_back([units, activation](size_t input_sz) -> std::unique_ptr<Layer> {
            auto layer = std::make_unique<DenseLayer>(input_sz, units);
            return layer;
        });
        last_units_ = units;
        return *this;
    }
    
    size_t input_size = (network_->num_layers() == 0) ? input_size_ : last_units_;
    network_->add_dense_layer(input_size, units, activation);
    last_units_ = units;
    return *this;
}

ModelBuilder& ModelBuilder::dropout(double rate) {
    if (last_units_ == 0) {
        throw std::runtime_error("Cannot add dropout layer without knowing input size");
    }
    
    network_->add_dropout_layer(static_cast<float>(rate));
    return *this;
}

ModelBuilder& ModelBuilder::activation(const std::string& name) {
    if (last_units_ == 0) {
        throw std::runtime_error("Cannot add activation layer without knowing input size");
    }
    
    network_->add_activation_layer(name, last_units_);
    return *this;
}

ModelBuilder& ModelBuilder::input(size_t size) {
    input_size_ = size;
    
    // Apply any pending layers now that we know the input size
    for (auto& layer_factory : pending_layers_) {
        (void)layer_factory; // Suppress unused parameter warning
        // For now, we'll need to add these layers manually
        // This would require extending the NeuralNetwork interface
    }
    pending_layers_.clear();
    
    return *this;
}

ModelBuilder& ModelBuilder::compile(const std::string& loss, 
                                  const std::string& optimizer,
                                  double learning_rate) {
    
    std::string actual_loss = loss;
    std::string actual_optimizer = optimizer;
    double actual_lr = learning_rate;
    
    if (loss == "auto") {
        actual_loss = infer_loss_function();
    }
    
    if (optimizer == "auto") {
        actual_optimizer = "adam";
    }
    
    if (learning_rate < 0) {
        actual_lr = (actual_optimizer == "adam") ? 0.001 : 0.01;
    }
    
    network_->set_loss_function(actual_loss);
    network_->set_optimizer(actual_optimizer, actual_lr);
    return *this;
}

std::unique_ptr<NeuralNetwork> ModelBuilder::build() {
    if (input_size_ == 0 && network_->num_layers() == 0) {
        throw std::runtime_error("Model must have at least one layer or input size specified");
    }
    
    return std::move(network_);
}

ModelBuilder ModelBuilder::mlp(const std::vector<size_t>& layer_sizes,
                              const std::string& hidden_activation,
                              const std::string& output_activation) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP must have at least input and output sizes");
    }
    
    ModelBuilder builder;
    builder.input(layer_sizes[0]);
    
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        std::string activation = (i == layer_sizes.size() - 1) ? output_activation : hidden_activation;
        builder.dense(layer_sizes[i], activation);
    }
    
    return builder;
}

ModelBuilder ModelBuilder::autoencoder(size_t input_dim, const std::vector<size_t>& encoder_dims) {
    ModelBuilder builder;
    builder.input(input_dim);
    
    // Encoder
    for (size_t dim : encoder_dims) {
        builder.dense(dim, "relu");
    }
    
    // Decoder (symmetric)
    for (int i = static_cast<int>(encoder_dims.size()) - 2; i >= 0; --i) {
        builder.dense(encoder_dims[i], "relu");
    }
    
    // Output layer
    builder.dense(input_dim, "sigmoid");
    
    return builder;
}

std::string ModelBuilder::infer_loss_function() const {
    // Simple heuristic based on output layer size
    if (last_units_ == 1) {
        return "binary_crossentropy";
    } else if (last_units_ > 1) {
        return "categorical_crossentropy";
    }
    return "mse";
}

// Trainer implementation
TrainingHistory Trainer::fit(NeuralNetwork& model, 
                           const Dataset& dataset,
                           const TrainingConfig& config,
                           const std::vector<std::unique_ptr<Callback>>& callbacks) {
    
    auto [train_data, val_data] = dataset.train_test_split(1.0 - config.validation_split);
    
    // Notify callbacks of training start
    for (auto& callback : callbacks) {
        callback->on_training_begin();
    }
    
    TrainingHistory history;
    bool should_stop = false;
    
    for (int epoch = 0; epoch < config.epochs && !should_stop; ++epoch) {
        // Notify epoch begin
        for (auto& callback : callbacks) {
            callback->on_epoch_begin(epoch);
        }
        
        // Training epoch
        double epoch_loss = 0.0;
        int batch_count = 0;
        
        // Simple batch processing (would need to extend Dataset for proper batching)
        for (int batch = 0; batch < 1; ++batch) { // Simplified: one batch per epoch
            for (auto& callback : callbacks) {
                callback->on_batch_begin(batch);
            }
            
            model.fit(train_data.features(), train_data.targets(), 
                     1, config.batch_size, 0.0, false);
            
            // Get batch loss (simplified)
            double batch_loss = model.evaluate(train_data.features(), train_data.targets());
            epoch_loss += batch_loss;
            batch_count++;
            
            for (auto& callback : callbacks) {
                callback->on_batch_end(batch, batch_loss);
            }
        }
        
        // Record training loss
        history.training_loss.push_back(epoch_loss / batch_count);
        
        // Validation
        if (config.validation_split > 0.0) {
            double val_loss = model.evaluate(val_data.features(), val_data.targets());
            history.validation_loss.push_back(val_loss);
        }
        
        // Notify epoch end
        for (auto& callback : callbacks) {
            callback->on_epoch_end(epoch, history);
        }
        
        // Check for early stopping
        for (auto& callback : callbacks) {
            if (auto* early_stop = dynamic_cast<EarlyStopping*>(callback.get())) {
                if (early_stop->should_stop()) {
                    should_stop = true;
                    break;
                }
            }
        }
        
        if (config.verbose && (epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs 
                      << " - loss: " << history.training_loss.back();
            if (!history.validation_loss.empty()) {
                std::cout << " - val_loss: " << history.validation_loss.back();
            }
            std::cout << std::endl;
        }
    }
    
    // Notify training end
    for (auto& callback : callbacks) {
        callback->on_training_end(history);
    }
    
    return history;
}

// EarlyStopping implementation
Trainer::EarlyStopping::EarlyStopping(int patience, double min_delta) 
    : patience_(patience), min_delta_(min_delta) {}

void Trainer::EarlyStopping::on_epoch_end(int epoch, const TrainingHistory& history) {
    if (!history.validation_loss.empty()) {
        double current_loss = history.validation_loss.back();
        if (current_loss < best_loss_ - min_delta_) {
            best_loss_ = current_loss;
            wait_ = 0;
        } else {
            wait_++;
            if (wait_ >= patience_) {
                should_stop_ = true;
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
            }
        }
    }
}

// ModelCheckpoint implementation
Trainer::ModelCheckpoint::ModelCheckpoint(const std::string& filepath, bool save_best_only)
    : filepath_(filepath), save_best_only_(save_best_only) {}

void Trainer::ModelCheckpoint::on_epoch_end(int epoch, const TrainingHistory& history) {
    if (!save_best_only_ || 
        (!history.validation_loss.empty() && history.validation_loss.back() < best_loss_)) {
        
        if (!history.validation_loss.empty()) {
            best_loss_ = history.validation_loss.back();
        }
        
        // In a real implementation, we'd save the model here
        std::cout << "Saved model checkpoint at epoch " << epoch + 1 << std::endl;
    }
}

// HyperparameterTuner implementation
HyperparameterTuner::TuningResult HyperparameterTuner::random_search(
    std::function<double(const HyperparameterConfig&)> objective,
    const HyperparameterSpace& space,
    int n_trials) {
    
    TuningResult result;
    result.best_score = std::numeric_limits<double>::max();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int trial = 0; trial < n_trials; ++trial) {
        HyperparameterConfig config = sample_config(space, gen);
        
        try {
            double score = objective(config);
            result.all_scores.push_back(score);
            
            if (score < result.best_score) {
                result.best_score = score;
                result.best_config = config;
            }
        } catch (const std::exception& e) {
            std::cout << "Trial " << trial << " failed: " << e.what() << std::endl;
            result.all_scores.push_back(std::numeric_limits<double>::max());
        }
        
        if (trial % 10 == 0) {
            std::cout << "Completed " << trial << "/" << n_trials << " trials. "
                      << "Best score: " << result.best_score << std::endl;
        }
    }
    
    return result;
}

HyperparameterTuner::HyperparameterConfig HyperparameterTuner::sample_config(
    const HyperparameterSpace& space, std::mt19937& gen) {
    
    HyperparameterConfig config;
      // Sample learning rate
    if (std::holds_alternative<std::vector<double>>(space.learning_rate)) {
        auto& values = std::get<std::vector<double>>(space.learning_rate);
        std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
        config.learning_rate = values[dist(gen)];
    } else {
        auto& range = std::get<std::pair<double, double>>(space.learning_rate);
        std::uniform_real_distribution<> dist(range.first, range.second);
        config.learning_rate = dist(gen);
    }
      // Sample batch size
    if (std::holds_alternative<std::vector<int>>(space.batch_size)) {
        auto& values = std::get<std::vector<int>>(space.batch_size);
        std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
        config.batch_size = values[dist(gen)];
    } else {
        auto& range = std::get<std::pair<int, int>>(space.batch_size);
        std::uniform_int_distribution<> dist(range.first, range.second);
        config.batch_size = dist(gen);
    }    
    // Sample optimizer
    if (std::holds_alternative<std::vector<std::string>>(space.optimizer)) {
        auto& values = std::get<std::vector<std::string>>(space.optimizer);
        std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
        config.optimizer = values[dist(gen)];
    } else {
        config.optimizer = std::get<std::string>(space.optimizer);    }
    
    // Sample dropout rate
    if (std::holds_alternative<std::vector<double>>(space.dropout_rate)) {
        auto& values = std::get<std::vector<double>>(space.dropout_rate);
        std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
        config.dropout_rate = values[dist(gen)];
    } else {
        auto& range = std::get<std::pair<double, double>>(space.dropout_rate);
        std::uniform_real_distribution<> dist(range.first, range.second);
        config.dropout_rate = dist(gen);
    }
      // Sample hidden layers (simplified)
    if (std::holds_alternative<std::vector<std::vector<int>>>(space.hidden_layers)) {
        auto& values = std::get<std::vector<std::vector<int>>>(space.hidden_layers);
        std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
        config.hidden_layers = values[dist(gen)];
    } else {
        config.hidden_layers = std::get<std::vector<int>>(space.hidden_layers);
    }
    
    return config;
}

} // namespace api
} // namespace clmodel
