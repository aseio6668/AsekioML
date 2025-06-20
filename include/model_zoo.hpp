#pragma once

#include "modern_api.hpp"
#include "advanced_layers.hpp"
#include <functional>
#include <string>

namespace asekioml {
namespace models {

// Model registry for easy access to pre-built architectures
class ModelZoo {
public:
    // Computer Vision Models
    static std::unique_ptr<NeuralNetwork> LeNet5(size_t num_classes = 10);
    static std::unique_ptr<NeuralNetwork> AlexNet(size_t num_classes = 1000);
    static std::unique_ptr<NeuralNetwork> VGG16(size_t num_classes = 1000);
    static std::unique_ptr<NeuralNetwork> ResNet50(size_t num_classes = 1000);
    
    // Natural Language Processing Models
    static std::unique_ptr<NeuralNetwork> SimpleRNN(size_t vocab_size, size_t embedding_dim, 
                                                    size_t hidden_size, size_t num_classes);
    static std::unique_ptr<NeuralNetwork> LSTM_Classifier(size_t vocab_size, size_t embedding_dim,
                                                          size_t hidden_size, size_t num_classes);
    static std::unique_ptr<NeuralNetwork> Transformer(size_t vocab_size, size_t d_model,
                                                      size_t num_heads, size_t num_layers,
                                                      size_t max_length);
    
    // Generative Models
    static std::unique_ptr<NeuralNetwork> Autoencoder(const std::vector<size_t>& encoder_dims);
    static std::unique_ptr<NeuralNetwork> VariationalAutoencoder(size_t input_dim, size_t latent_dim);
    static std::unique_ptr<NeuralNetwork> GAN_Generator(size_t noise_dim, size_t output_dim);
    static std::unique_ptr<NeuralNetwork> GAN_Discriminator(size_t input_dim);
    
    // Specialized Models
    static std::unique_ptr<NeuralNetwork> RecommenderSystem(size_t num_users, size_t num_items,
                                                           size_t embedding_dim);
    static std::unique_ptr<NeuralNetwork> TimeSeriesPredictor(size_t input_length, size_t features,
                                                             size_t hidden_size, size_t output_length);
    
    // Transfer Learning Base Models
    static std::unique_ptr<NeuralNetwork> load_pretrained(const std::string& model_name,
                                                          const std::string& weights_path = "");
};

// High-level model interfaces for common tasks
namespace vision {

class ImageClassifier {
private:
    std::unique_ptr<NeuralNetwork> model_;
    std::vector<std::string> class_names_;
    
public:
    ImageClassifier(const std::string& architecture = "resnet50", size_t num_classes = 1000);
    
    // Training interface
    void fit(const Dataset& train_data, const Dataset& val_data, int epochs = 100);
    
    // Prediction interface
    std::vector<double> predict_proba(const Matrix& image);
    std::string predict_class(const Matrix& image);
    std::vector<std::pair<std::string, double>> predict_top_k(const Matrix& image, int k = 5);
    
    // Transfer learning
    void fine_tune(const Dataset& new_data, int epochs = 50, double learning_rate = 1e-4);
    
    // Model evaluation
    double evaluate_accuracy(const Dataset& test_data);
    void confusion_matrix(const Dataset& test_data);
    
    void set_class_names(const std::vector<std::string>& names) { class_names_ = names; }
};

class ObjectDetector {
    // Implementation for object detection models
    // YOLO, R-CNN style architectures
};

} // namespace vision

namespace nlp {

class TextClassifier {
private:
    std::unique_ptr<NeuralNetwork> model_;
    std::unordered_map<std::string, size_t> vocab_;
    std::vector<std::string> class_names_;
    size_t max_length_;
    
public:
    TextClassifier(const std::string& architecture = "lstm", 
                  size_t vocab_size = 10000, size_t max_length = 128);
    
    // Text preprocessing
    void build_vocabulary(const std::vector<std::string>& texts);
    Matrix tokenize(const std::string& text);
    Matrix tokenize_batch(const std::vector<std::string>& texts);
    
    // Training
    void fit(const std::vector<std::string>& texts, 
            const std::vector<std::string>& labels, int epochs = 100);
    
    // Prediction
    std::string predict(const std::string& text);
    std::vector<double> predict_proba(const std::string& text);
    
    // Evaluation
    double evaluate(const std::vector<std::string>& texts, 
                   const std::vector<std::string>& labels);
};

class LanguageModel {
private:
    std::unique_ptr<NeuralNetwork> model_;
    std::unordered_map<std::string, size_t> vocab_;
    std::unordered_map<size_t, std::string> reverse_vocab_;
    
public:
    LanguageModel(const std::string& architecture = "transformer");
    
    // Training on text corpus
    void fit(const std::vector<std::string>& corpus, int epochs = 100);
    
    // Text generation
    std::string generate(const std::string& prompt, size_t max_length = 100,
                        double temperature = 1.0);
    
    // Perplexity calculation
    double perplexity(const std::vector<std::string>& test_texts);
};

class SentimentAnalyzer {
private:
    std::unique_ptr<NeuralNetwork> model_;
    TextClassifier classifier_;
    
public:
    SentimentAnalyzer();
    
    void fit(const std::vector<std::string>& texts, 
            const std::vector<std::string>& sentiments);
    
    std::string analyze_sentiment(const std::string& text);
    double sentiment_score(const std::string& text);  // -1 to 1 scale
};

} // namespace nlp

namespace timeseries {

class TimeSeriesForecaster {
private:
    std::unique_ptr<NeuralNetwork> model_;
    size_t window_size_;
    size_t forecast_horizon_;
    
public:
    TimeSeriesForecaster(size_t window_size = 24, size_t forecast_horizon = 1,
                        const std::string& architecture = "lstm");
    
    // Data preparation
    Dataset prepare_data(const Matrix& time_series, size_t window_size);
    
    // Training
    void fit(const Matrix& time_series, int epochs = 100);
    
    // Forecasting
    Matrix forecast(const Matrix& recent_data, size_t steps = 1);
    
    // Evaluation metrics
    double mean_absolute_error(const Matrix& actual, const Matrix& predicted);
    double mean_squared_error(const Matrix& actual, const Matrix& predicted);
    double mean_absolute_percentage_error(const Matrix& actual, const Matrix& predicted);
};

class AnomalyDetector {
private:
    std::unique_ptr<NeuralNetwork> autoencoder_;
    double threshold_;
    
public:
    AnomalyDetector(size_t input_dim, const std::vector<size_t>& encoder_dims);
    
    // Training on normal data
    void fit(const Matrix& normal_data, int epochs = 100);
    
    // Anomaly detection
    std::vector<bool> detect_anomalies(const Matrix& data);
    std::vector<double> anomaly_scores(const Matrix& data);
    
    // Threshold tuning
    void set_threshold(double threshold) { threshold_ = threshold; }
    double auto_threshold(const Matrix& validation_data, double percentile = 95.0);
};

} // namespace timeseries

namespace reinforcement {

class DQNAgent {
    // Deep Q-Network for reinforcement learning
private:
    std::unique_ptr<NeuralNetwork> q_network_;
    std::unique_ptr<NeuralNetwork> target_network_;
    
public:
    DQNAgent(size_t state_dim, size_t action_dim, double learning_rate = 1e-3);
    
    // Action selection
    size_t select_action(const Matrix& state, double epsilon = 0.1);
    
    // Training
    void update(const Matrix& states, const Matrix& actions, 
               const Matrix& rewards, const Matrix& next_states, 
               const Matrix& dones, double gamma = 0.99);
    
    // Target network update
    void update_target_network();
};

} // namespace reinforcement

// Model comparison and benchmarking
class ModelBenchmark {
public:
    struct BenchmarkResult {
        std::string model_name;
        double training_time;
        double inference_time;
        double memory_usage;
        double accuracy;
        size_t parameters;
    };
    
    static std::vector<BenchmarkResult> compare_models(
        const std::vector<std::unique_ptr<NeuralNetwork>>& models,
        const Dataset& train_data,
        const Dataset& test_data,
        int epochs = 50);
    
    static void print_benchmark_table(const std::vector<BenchmarkResult>& results);
};

// AutoML functionality
class AutoML {
public:
    struct AutoMLConfig {
        int max_trials = 100;
        int epochs_per_trial = 30;
        double time_budget_hours = 24.0;
        std::vector<std::string> architectures = {"mlp", "lstm", "transformer"};
        bool use_early_stopping = true;
        bool use_hyperband = true;
    };
    
    static std::unique_ptr<NeuralNetwork> auto_classifier(
        const Dataset& data,
        const AutoMLConfig& config = {});
    
    static std::unique_ptr<NeuralNetwork> auto_regressor(
        const Dataset& data,
        const AutoMLConfig& config = {});

private:
    static double objective_function(const api::HyperparameterTuner::HyperparameterConfig& config,
                                   const Dataset& data, int epochs);
};

} // namespace models
} // namespace asekioml
