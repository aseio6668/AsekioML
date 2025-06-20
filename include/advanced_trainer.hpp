#pragma once

#include "network.hpp"
#include "callbacks.hpp"
#include "schedulers.hpp"
#include "advanced_optimizers.hpp"
#include "gradient_clipping.hpp"
#include "advanced_losses.hpp"
#include <memory>
#include <vector>
#include <chrono>

namespace asekioml {

class AdvancedTrainer {
private:
    std::unique_ptr<NeuralNetwork> network_;
    std::vector<std::unique_ptr<TrainingCallback>> callbacks_;
    std::unique_ptr<LearningRateScheduler> lr_scheduler_;
    
    // Training configuration
    struct TrainingConfig {
        int epochs = 100;
        int batch_size = 32;
        bool shuffle = true;
        double validation_split = 0.2;
        int validation_freq = 1;
        bool verbose = true;
        
        // Gradient clipping
        bool use_gradient_clipping = false;
        std::string clip_type = "norm"; // "norm", "value", "global_norm"
        double clip_value = 1.0;
        
        // Mixed precision
        bool use_mixed_precision = false;
        
        // Gradient accumulation
        int accumulation_steps = 1;
        
        // Resume training
        bool resume_from_checkpoint = false;
        std::string checkpoint_path = "";
    } config_;
    
    TrainingHistory history_;
    
    void validate_inputs(const Matrix& X, const Matrix& y) {
        if (X.rows() != y.rows()) {
            throw std::invalid_argument("Number of samples in X and y must match");
        }
        if (X.rows() == 0) {
            throw std::invalid_argument("Training data cannot be empty");
        }
    }
    
    std::pair<Matrix, Matrix> create_validation_split(const Matrix& X, const Matrix& y) {
        size_t total_samples = X.rows();
        size_t val_samples = static_cast<size_t>(total_samples * config_.validation_split);
        size_t train_samples = total_samples - val_samples;
        
        // Simple split (in practice, you'd want to shuffle first)
        Matrix X_train(train_samples, X.cols());
        Matrix y_train(train_samples, y.cols());
        Matrix X_val(val_samples, X.cols());
        Matrix y_val(val_samples, y.cols());
        
        // Copy training data
        for (size_t i = 0; i < train_samples; ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                X_train(i, j) = X(i, j);
            }
            for (size_t j = 0; j < y.cols(); ++j) {
                y_train(i, j) = y(i, j);
            }
        }
        
        // Copy validation data
        for (size_t i = 0; i < val_samples; ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                X_val(i, j) = X(train_samples + i, j);
            }
            for (size_t j = 0; j < y.cols(); ++j) {
                y_val(i, j) = y(train_samples + i, j);
            }
        }
        
        return {X_val, y_val};
    }
    
    void apply_gradient_clipping(std::vector<Matrix*>& gradients) {
        if (!config_.use_gradient_clipping) return;
        
        if (config_.clip_type == "norm") {
            for (auto* grad : gradients) {
                gradient_clipping::clip_by_norm(*grad, config_.clip_value);
            }
        } else if (config_.clip_type == "value") {
            for (auto* grad : gradients) {
                gradient_clipping::clip_by_value(*grad, -config_.clip_value, config_.clip_value);
            }
        } else if (config_.clip_type == "global_norm") {
            gradient_clipping::clip_by_global_norm(gradients, config_.clip_value);
        }
    }
    
    double evaluate_accuracy(const Matrix& predictions, const Matrix& targets) {
        if (predictions.cols() == 1) {
            // Binary classification or regression
            if (targets.cols() == 1) {
                // Check if targets are 0/1 (classification) or continuous (regression)
                bool is_classification = true;
                for (size_t i = 0; i < targets.rows(); ++i) {
                    double val = targets(i, 0);
                    if (val != 0.0 && val != 1.0) {
                        is_classification = false;
                        break;
                    }
                }
                
                if (is_classification) {
                    // Binary classification accuracy
                    int correct = 0;
                    for (size_t i = 0; i < predictions.rows(); ++i) {
                        int pred_class = predictions(i, 0) > 0.5 ? 1 : 0;
                        int true_class = static_cast<int>(targets(i, 0));
                        if (pred_class == true_class) correct++;
                    }
                    return static_cast<double>(correct) / predictions.rows();
                } else {
                    // For regression, return negative MSE as accuracy metric
                    double mse = 0.0;
                    for (size_t i = 0; i < predictions.rows(); ++i) {
                        double diff = predictions(i, 0) - targets(i, 0);
                        mse += diff * diff;
                    }
                    return -mse / predictions.rows();
                }
            }
        } else {
            // Multi-class classification
            int correct = 0;
            for (size_t i = 0; i < predictions.rows(); ++i) {
                // Find predicted class
                size_t pred_class = 0;
                for (size_t j = 1; j < predictions.cols(); ++j) {
                    if (predictions(i, j) > predictions(i, pred_class)) {
                        pred_class = j;
                    }
                }
                
                // Find true class
                size_t true_class = 0;
                for (size_t j = 1; j < targets.cols(); ++j) {
                    if (targets(i, j) > targets(i, true_class)) {
                        true_class = j;
                    }
                }
                
                if (pred_class == true_class) correct++;
            }
            return static_cast<double>(correct) / predictions.rows();
        }
        
        return 0.0;
    }
    
public:
    AdvancedTrainer(std::unique_ptr<NeuralNetwork> network) 
        : network_(std::move(network)) {}
    
    // Configuration methods
    void set_epochs(int epochs) { config_.epochs = epochs; }
    void set_batch_size(int batch_size) { config_.batch_size = batch_size; }
    void set_validation_split(double split) { config_.validation_split = split; }
    void set_verbose(bool verbose) { config_.verbose = verbose; }
    
    void enable_gradient_clipping(const std::string& clip_type, double clip_value) {
        config_.use_gradient_clipping = true;
        config_.clip_type = clip_type;
        config_.clip_value = clip_value;
    }
    
    void set_gradient_accumulation(int steps) {
        config_.accumulation_steps = steps;
    }
    
    // Add callbacks
    void add_callback(std::unique_ptr<TrainingCallback> callback) {
        callbacks_.push_back(std::move(callback));
    }
    
    void set_lr_scheduler(std::unique_ptr<LearningRateScheduler> scheduler) {
        lr_scheduler_ = std::move(scheduler);
    }
    
    // Training method
    TrainingHistory fit(const Matrix& X, const Matrix& y, 
                       const Matrix* X_val = nullptr, const Matrix* y_val = nullptr) {
        validate_inputs(X, y);
        
        if (!network_) {
            throw std::runtime_error("Network not initialized");
        }
        
        history_.clear();
        
        // Setup validation data
        Matrix X_validation, y_validation;
        bool has_validation = false;
        
        if (X_val && y_val) {
            X_validation = *X_val;
            y_validation = *y_val;
            has_validation = true;
        } else if (config_.validation_split > 0.0) {
            auto val_data = create_validation_split(X, y);
            X_validation = val_data.first;
            y_validation = val_data.second;
            has_validation = true;
        }
        
        // Initialize callbacks
        for (auto& callback : callbacks_) {
            callback->on_training_begin(history_);
        }
        
        if (config_.verbose) {
            std::cout << "Training for " << config_.epochs << " epochs..." << std::endl;
            if (has_validation) {
                std::cout << "Using validation split: " << config_.validation_split << std::endl;
            }
        }
        
        // Training loop
        for (int epoch = 0; epoch < config_.epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            // Epoch begin callbacks
            for (auto& callback : callbacks_) {
                callback->on_epoch_begin(epoch, history_);
            }
            
            // Update learning rate
            if (lr_scheduler_) {
                double new_lr = lr_scheduler_->get_lr(epoch, network_->get_learning_rate());
                network_->set_learning_rate(new_lr);
            }
            
            // Training phase
            double epoch_loss = 0.0;
            int num_batches = 0;
            
            // Simplified batch processing (in practice, you'd implement proper batching)
            for (size_t i = 0; i < X.rows(); i += config_.batch_size) {
                size_t batch_end = std::min(i + config_.batch_size, X.rows());
                size_t batch_size = batch_end - i;
                
                // Create batch matrices
                Matrix X_batch(batch_size, X.cols());
                Matrix y_batch(batch_size, y.cols());
                
                for (size_t j = 0; j < batch_size; ++j) {
                    for (size_t k = 0; k < X.cols(); ++k) {
                        X_batch(j, k) = X(i + j, k);
                    }
                    for (size_t k = 0; k < y.cols(); ++k) {
                        y_batch(j, k) = y(i + j, k);
                    }
                }
                
                // Forward and backward pass
                Matrix predictions = network_->predict(X_batch);
                network_->train_step(X_batch, y_batch);
                
                // Compute batch loss
                double batch_loss = network_->compute_loss(predictions, y_batch);
                epoch_loss += batch_loss;
                num_batches++;
                
                // Batch callbacks
                for (auto& callback : callbacks_) {
                    callback->on_batch_end(num_batches - 1, batch_loss, history_);
                }
            }
            
            epoch_loss /= num_batches;
            
            // Validation phase
            double val_loss = 0.0;
            double val_accuracy = 0.0;
            if (has_validation && (epoch % config_.validation_freq == 0)) {
                Matrix val_predictions = network_->predict(X_validation);
                val_loss = network_->compute_loss(val_predictions, y_validation);
                val_accuracy = evaluate_accuracy(val_predictions, y_validation);
            }
            
            // Compute training accuracy
            Matrix train_predictions = network_->predict(X);
            double train_accuracy = evaluate_accuracy(train_predictions, y);
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
            
            // Update history
            history_.add_epoch(epoch_loss, val_loss, train_accuracy, val_accuracy, 
                             network_->get_learning_rate(), epoch_time);
            
            // Epoch end callbacks
            for (auto& callback : callbacks_) {
                callback->on_epoch_end(epoch, history_);
            }
            
            // Check for early stopping
            bool should_stop = false;
            for (auto& callback : callbacks_) {
                if (callback->should_stop()) {
                    should_stop = true;
                    break;
                }
            }
            
            if (should_stop) {
                if (config_.verbose) {
                    std::cout << "Training stopped early at epoch " << epoch + 1 << std::endl;
                }
                break;
            }
        }
        
        // Training end callbacks
        for (auto& callback : callbacks_) {
            callback->on_training_end(history_);
        }
        
        if (config_.verbose) {
            std::cout << "Training completed!" << std::endl;
        }
        
        return history_;
    }
    
    const TrainingHistory& get_history() const {
        return history_;
    }
    
    NeuralNetwork* get_network() {
        return network_.get();
    }
};

} // namespace asekioml
