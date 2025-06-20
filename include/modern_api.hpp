#pragma once

#include "network.hpp"
#include "dataset.hpp"
#include <functional>
#include <variant>
#include <memory>
#include <vector>
#include <string>
#include <random>

namespace asekioml {
namespace api {

// Forward declarations
class ModelBuilder;
class Trainer;
class HyperparameterTuner;

// Training history structure
struct TrainingHistory {
    std::vector<double> training_loss;
    std::vector<double> validation_loss;
    std::vector<double> training_accuracy;
    std::vector<double> validation_accuracy;
};

// Fluent API for model building
class ModelBuilder {
private:
    std::unique_ptr<NeuralNetwork> network_;
    std::vector<std::function<std::unique_ptr<Layer>(size_t)>> pending_layers_;
    size_t input_size_ = 0;
    size_t last_units_ = 0;
    
public:
    ModelBuilder();
    
    // Fluent layer addition
    ModelBuilder& dense(size_t units, const std::string& activation = "linear");
    ModelBuilder& dropout(double rate = 0.5);
    ModelBuilder& activation(const std::string& name);
    ModelBuilder& input(size_t size);
    
    // Regularization
    ModelBuilder& l1_regularization(double lambda = 0.01);
    ModelBuilder& l2_regularization(double lambda = 0.01);
    ModelBuilder& batch_norm();
    
    // Compile with automatic best practices
    ModelBuilder& compile(const std::string& loss = "auto", 
                         const std::string& optimizer = "auto",
                         double learning_rate = -1.0);
    
    // Build the final model
    std::unique_ptr<NeuralNetwork> build();
    
    // Quick model creation for common architectures
    static ModelBuilder mlp(const std::vector<size_t>& layer_sizes,
                           const std::string& hidden_activation = "relu",
                           const std::string& output_activation = "linear");
    
    static ModelBuilder autoencoder(size_t input_dim, const std::vector<size_t>& encoder_dims);

private:
    size_t get_last_output_size() const;
    std::string infer_loss_function() const;
};

// Convenient training interface with callbacks
class Trainer {
public:
    struct TrainingConfig {
        int epochs = 100;
        int batch_size = 32;
        double validation_split = 0.2;
        bool verbose = true;
        bool early_stopping = true;
        int patience = 10;
        double min_delta = 1e-4;
        bool reduce_lr_on_plateau = true;
        double lr_reduction_factor = 0.5;
        int lr_patience = 5;
    };
    
    // Callback system
    class Callback {
    public:
        virtual ~Callback() = default;        virtual void on_epoch_begin(int /*epoch*/) {}
        virtual void on_epoch_end(int /*epoch*/, const TrainingHistory& /*history*/) {}
        virtual void on_batch_begin(int /*batch*/) {}
        virtual void on_batch_end(int /*batch*/, double /*loss*/) {}
        virtual void on_training_begin() {}
        virtual void on_training_end(const TrainingHistory& /*history*/) {}
    };
    
    // Built-in callbacks
    class EarlyStopping : public Callback {
    private:
        int patience_;
        double min_delta_;
        double best_loss_ = std::numeric_limits<double>::max();
        int wait_ = 0;
        bool should_stop_ = false;
        
    public:
        EarlyStopping(int patience = 10, double min_delta = 1e-4);
        void on_epoch_end(int epoch, const TrainingHistory& history) override;
        bool should_stop() const { return should_stop_; }
    };
    
    class ModelCheckpoint : public Callback {
    private:
        std::string filepath_;
        bool save_best_only_;
        double best_loss_ = std::numeric_limits<double>::max();
        
    public:
        ModelCheckpoint(const std::string& filepath, bool save_best_only = true);
        void on_epoch_end(int epoch, const TrainingHistory& history) override;
    };
    
    class LearningRateScheduler : public Callback {
    private:
        std::function<double(int, double)> schedule_;
        NeuralNetwork* network_;
        
    public:
        LearningRateScheduler(std::function<double(int, double)> schedule, NeuralNetwork* network);
        void on_epoch_begin(int epoch) override;
    };
    
    static TrainingHistory fit(NeuralNetwork& model, 
                              const Dataset& dataset,
                              const TrainingConfig& config = {},
                              const std::vector<std::unique_ptr<Callback>>& callbacks = {});
};

// Hyperparameter tuning
class HyperparameterTuner {
public:
    struct HyperparameterSpace {
        std::variant<std::vector<double>, std::pair<double, double>> learning_rate;
        std::variant<std::vector<int>, std::pair<int, int>> batch_size;
        std::variant<std::vector<std::string>, std::string> optimizer;
        std::variant<std::vector<double>, std::pair<double, double>> dropout_rate;
        std::variant<std::vector<std::vector<int>>, std::vector<int>> hidden_layers;
    };
    
    struct HyperparameterConfig {
        double learning_rate;
        int batch_size;
        std::string optimizer;
        double dropout_rate;
        std::vector<int> hidden_layers;
    };
    
    struct TuningResult {
        HyperparameterConfig best_config;
        double best_score;
        std::vector<double> all_scores;
    };
    
    TuningResult random_search(std::function<double(const HyperparameterConfig&)> objective,
                              const HyperparameterSpace& space,
                              int n_trials = 50);

private:
    HyperparameterConfig sample_config(const HyperparameterSpace& space, std::mt19937& gen);
};

} // namespace api
} // namespace asekioml
