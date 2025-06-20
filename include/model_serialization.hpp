#pragma once

#include "network.hpp"
#include <string>
#include <fstream>
#include <memory>

namespace clmodel {

/**
 * @brief Model serialization formats
 */
enum class SerializationFormat {
    BINARY,    // Fast, compact binary format
    JSON,      // Human-readable JSON format
    HYBRID     // JSON architecture + binary weights (recommended)
};

/**
 * @brief Model serialization and deserialization utilities
 * 
 * Provides functionality to save and load trained neural networks,
 * including architecture, weights, optimizer state, and training history.
 */
class ModelSerializer {
public:
    /**
     * @brief Save a neural network to file
     * @param network The network to save
     * @param filepath Path to save the model
     * @param format Serialization format to use
     * @param include_optimizer Whether to save optimizer state
     * @param include_history Whether to save training history
     * @return True if save was successful
     */
    static bool save(const NeuralNetwork& network, 
                     const std::string& filepath,
                     SerializationFormat format = SerializationFormat::HYBRID,
                     bool include_optimizer = true,
                     bool include_history = true);
    
    /**
     * @brief Load a neural network from file
     * @param filepath Path to the saved model
     * @param format Serialization format (auto-detected if not specified)
     * @return Loaded neural network, or nullptr if load failed
     */
    static std::unique_ptr<NeuralNetwork> load(const std::string& filepath,
                                               SerializationFormat format = SerializationFormat::HYBRID);
    
    /**
     * @brief Get model metadata without loading the full model
     * @param filepath Path to the saved model
     * @return JSON string containing model metadata
     */
    static std::string get_model_info(const std::string& filepath);
    
    /**
     * @brief Verify model file integrity
     * @param filepath Path to the saved model
     * @return True if file is valid and can be loaded
     */
    static bool verify_model_file(const std::string& filepath);

private:
    // Binary format helpers
    static bool save_binary(const NeuralNetwork& network, const std::string& filepath,
                           bool include_optimizer, bool include_history);
    static std::unique_ptr<NeuralNetwork> load_binary(const std::string& filepath);
    
    // JSON format helpers
    static bool save_json(const NeuralNetwork& network, const std::string& filepath,
                         bool include_optimizer, bool include_history);
    static std::unique_ptr<NeuralNetwork> load_json(const std::string& filepath);
    
    // Hybrid format helpers
    static bool save_hybrid(const NeuralNetwork& network, const std::string& filepath,
                           bool include_optimizer, bool include_history);
    static std::unique_ptr<NeuralNetwork> load_hybrid(const std::string& filepath);    // Utility functions
    static std::string serialize_layer_to_json(const Layer& layer);
    static std::unique_ptr<Layer> deserialize_layer_from_json(const std::string& json);
    static void write_matrix_binary(std::ofstream& file, const Matrix& matrix);
    static Matrix read_matrix_binary(std::ifstream& file);
    static std::string matrix_to_json(const Matrix& matrix);
    static Matrix matrix_from_json(const std::string& json);
    static std::string escape_json_string(const std::string& str);
    static std::string format_double(double value, int precision = 6);
    
    // Layer-specific deserialization helpers
    static std::unique_ptr<Layer> deserialize_dense_layer(const std::string& json);
    static std::unique_ptr<Layer> deserialize_activation_layer(const std::string& json);
    static std::unique_ptr<Layer> deserialize_dropout_layer(const std::string& json);
    static std::unique_ptr<Layer> deserialize_batchnorm_layer(const std::string& json);
    static size_t find_matching_brace(const std::string& json, size_t start_pos);
    
    // Version and compatibility
    static const std::string MODEL_VERSION;
    static const std::string MODEL_MAGIC_NUMBER;
    static bool is_compatible_version(const std::string& version);
};

/**
 * @brief Model checkpoint manager for training with serialization support
 * 
 * Automatically saves model checkpoints during training based on
 * configurable criteria (best loss, periodic saves, etc.)
 */
class SerializationCheckpoint {
private:
    std::string filepath_template_;  // Template with {epoch}, {loss} placeholders
    std::string monitor_metric_;     // Metric to monitor (e.g., "val_loss", "val_accuracy")
    bool save_best_only_;           // Only save when metric improves
    bool save_weights_only_;        // Save only weights, not full model
    double best_metric_;            // Best metric value seen so far
    int save_frequency_;            // Save every N epochs (if save_best_only_ is false)
    SerializationFormat format_;    // Format to use for saving

public:
    /**
     * @brief Create a model checkpoint manager
     * @param filepath_template Path template (e.g., "model_epoch_{epoch}_loss_{loss:.4f}.clmodel")
     * @param monitor Metric to monitor ("loss", "val_loss", "accuracy", "val_accuracy")
     * @param save_best_only Only save when monitored metric improves
     * @param save_weights_only Save only weights, not optimizer state and history
     * @param save_frequency Save every N epochs (ignored if save_best_only is true)
     * @param format Serialization format to use
     */
    SerializationCheckpoint(const std::string& filepath_template,
                   const std::string& monitor = "val_loss",
                   bool save_best_only = true,
                   bool save_weights_only = false,
                   int save_frequency = 1,
                   SerializationFormat format = SerializationFormat::HYBRID);
    
    /**
     * @brief Check if model should be saved and save if needed
     * @param network Network to potentially save
     * @param epoch Current epoch number
     * @param metrics Map of metric names to values
     * @return True if model was saved
     */
    bool on_epoch_end(const NeuralNetwork& network, int epoch, 
                     const std::map<std::string, double>& metrics);
    
    /**
     * @brief Get the best metric value seen so far
     * @return Best metric value
     */
    double get_best_metric() const { return best_metric_; }
    
    /**
     * @brief Reset the checkpoint state
     */
    void reset();

private:
    std::string format_filepath(int epoch, const std::map<std::string, double>& metrics) const;
    bool should_save(const std::map<std::string, double>& metrics, int epoch) const;
};

} // namespace clmodel
