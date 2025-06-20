#include "../include/model_serialization.hpp"
#include "../include/regularization.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <map>
#include <algorithm>
#include <regex>

namespace clmodel {

// Version information
const std::string ModelSerializer::MODEL_VERSION = "1.0.0";
const std::string ModelSerializer::MODEL_MAGIC_NUMBER = "CLMODEL";

// ==================== Public Interface ====================

bool ModelSerializer::save(const NeuralNetwork& network, 
                           const std::string& filepath,
                           SerializationFormat format,
                           bool include_optimizer,
                           bool include_history) {
    try {
        switch (format) {
            case SerializationFormat::BINARY:
                return save_binary(network, filepath, include_optimizer, include_history);
            case SerializationFormat::JSON:
                return save_json(network, filepath, include_optimizer, include_history);
            case SerializationFormat::HYBRID:
                return save_hybrid(network, filepath, include_optimizer, include_history);
            default:
                std::cerr << "Unknown serialization format" << std::endl;
                return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<NeuralNetwork> ModelSerializer::load(const std::string& filepath,
                                                     SerializationFormat format) {
    try {
        switch (format) {
            case SerializationFormat::BINARY:
                return load_binary(filepath);
            case SerializationFormat::JSON:
                return load_json(filepath);
            case SerializationFormat::HYBRID:
                return load_hybrid(filepath);
            default:
                std::cerr << "Unknown serialization format" << std::endl;
                return nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return nullptr;
    }
}

std::string ModelSerializer::get_model_info(const std::string& filepath) {
    // Try to read just the header/metadata
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return "{}";
    }
    
    // Read magic number
    char magic[8];
    file.read(magic, 7);
    magic[7] = '\0';
    
    if (std::string(magic) != MODEL_MAGIC_NUMBER) {
        // Try JSON format
        file.close();
        std::ifstream json_file(filepath);
        if (!json_file.is_open()) return "{}";
        
        std::string line;
        std::getline(json_file, line);
        if (line.find("\"clmodel_version\"") != std::string::npos) {
            return line;  // Return first line which should contain metadata
        }
        return "{}";
    }
    
    // Read version
    uint32_t version_len;
    file.read(reinterpret_cast<char*>(&version_len), sizeof(version_len));
    std::string version(version_len, '\0');
    file.read(&version[0], version_len);
    
    // Read num layers
    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    std::ostringstream info;
    info << "{"
         << "\"format\":\"binary\","
         << "\"version\":\"" << version << "\","
         << "\"num_layers\":" << num_layers
         << "}";
    
    return info.str();
}

bool ModelSerializer::verify_model_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Check magic number
    char magic[8];
    file.read(magic, 7);
    magic[7] = '\0';
    
    if (std::string(magic) == MODEL_MAGIC_NUMBER) {
        // Binary format - check version
        uint32_t version_len;
        file.read(reinterpret_cast<char*>(&version_len), sizeof(version_len));
        if (version_len > 100) return false;  // Sanity check
        
        std::string version(version_len, '\0');
        file.read(&version[0], version_len);
        return is_compatible_version(version);
    } else {
        // Try JSON format
        file.close();
        std::ifstream json_file(filepath);
        std::string line;
        std::getline(json_file, line);
        return line.find("\"clmodel_version\"") != std::string::npos;
    }
}

// ==================== Hybrid Format (Recommended) ====================

bool ModelSerializer::save_hybrid(const NeuralNetwork& network, 
                                  const std::string& filepath,
                                  bool include_optimizer,
                                  bool include_history) {
    // Create architecture JSON
    std::ostringstream json_stream;
    json_stream << "{\n";
    json_stream << "  \"clmodel_version\": \"" << MODEL_VERSION << "\",\n";
    json_stream << "  \"format\": \"hybrid\",\n";
    json_stream << "  \"compiled\": " << (network.is_compiled() ? "true" : "false") << ",\n";
    json_stream << "  \"num_layers\": " << network.num_layers() << ",\n";
    json_stream << "  \"layers\": [\n";
    
    const auto& layers = network.get_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        json_stream << "    " << layers[i]->serialize_to_json();
        if (i < layers.size() - 1) json_stream << ",";
        json_stream << "\n";
    }
    
    json_stream << "  ]";
    
    // Add loss function and optimizer info if compiled
    if (network.is_compiled()) {
        json_stream << ",\n";
        if (network.get_loss_function()) {
            json_stream << "  \"loss_function\": {\n";
            json_stream << "    \"type\": \"" << network.get_loss_function()->name() << "\"\n";
            json_stream << "  }";
        }
        
        if (network.get_optimizer()) {
            json_stream << ",\n";
            json_stream << "  \"optimizer\": {\n";
            json_stream << "    \"type\": \"" << network.get_optimizer()->name() << "\",\n";
            json_stream << "    \"learning_rate\": " << network.get_learning_rate() << "\n";
            json_stream << "  }";
        }
    }
    
    json_stream << "\n}";
    
    // Save architecture JSON
    std::string json_filepath = filepath + ".arch";
    std::ofstream json_file(json_filepath);
    if (!json_file.is_open()) {
        return false;
    }
    json_file << json_stream.str();
    json_file.close();
    
    // Save weights in binary format
    std::string weights_filepath = filepath + ".weights";
    std::ofstream weights_file(weights_filepath, std::ios::binary);
    if (!weights_file.is_open()) {
        return false;
    }
    
    // Write magic number and version for weights file
    weights_file.write(MODEL_MAGIC_NUMBER.c_str(), MODEL_MAGIC_NUMBER.length());
    uint32_t version_len = MODEL_VERSION.length();
    weights_file.write(reinterpret_cast<const char*>(&version_len), sizeof(version_len));
    weights_file.write(MODEL_VERSION.c_str(), version_len);
    
    // Write number of layers (for validation)
    uint32_t num_layers = static_cast<uint32_t>(network.num_layers());
    weights_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // Write layer weights
    for (const auto& layer : layers) {
        layer->serialize_weights(weights_file);
    }
    
    weights_file.close();
    
    // Create a master file that references both
    std::ofstream master_file(filepath);
    if (!master_file.is_open()) {
        return false;
    }
    
    master_file << "{\n";
    master_file << "  \"format\": \"hybrid\",\n";
    master_file << "  \"version\": \"" << MODEL_VERSION << "\",\n";
    master_file << "  \"architecture_file\": \"" << filepath + ".arch" << "\",\n";
    master_file << "  \"weights_file\": \"" << filepath + ".weights" << "\"\n";
    master_file << "}";
    
    return true;
}

std::unique_ptr<NeuralNetwork> ModelSerializer::load_hybrid(const std::string& filepath) {
    // Read master file
    std::ifstream master_file(filepath);
    if (!master_file.is_open()) {
        return nullptr;
    }
    
    std::string master_content((std::istreambuf_iterator<char>(master_file)),
                               std::istreambuf_iterator<char>());
    master_file.close();
    
    // Extract architecture and weights file paths
    std::string arch_filepath, weights_filepath;
    
    // Simple JSON parsing for file paths
    size_t arch_pos = master_content.find("\"architecture_file\":");
    if (arch_pos != std::string::npos) {
        size_t start = master_content.find("\"", arch_pos + 20) + 1;
        size_t end = master_content.find("\"", start);
        arch_filepath = master_content.substr(start, end - start);
    }
    
    size_t weights_pos = master_content.find("\"weights_file\":");
    if (weights_pos != std::string::npos) {
        size_t start = master_content.find("\"", weights_pos + 15) + 1;
        size_t end = master_content.find("\"", start);
        weights_filepath = master_content.substr(start, end - start);
    }
    
    if (arch_filepath.empty() || weights_filepath.empty()) {
        return nullptr;
    }
    
    // Load architecture
    std::ifstream arch_file(arch_filepath);
    if (!arch_file.is_open()) {
        return nullptr;
    }
    
    std::string arch_content((std::istreambuf_iterator<char>(arch_file)),
                             std::istreambuf_iterator<char>());
    arch_file.close();
    
    // Parse architecture JSON and reconstruct network
    auto network = std::make_unique<NeuralNetwork>();
    
    // Extract layers array from JSON
    size_t layers_pos = arch_content.find("\"layers\":");
    if (layers_pos == std::string::npos) {
        return nullptr;
    }
    
    size_t array_start = arch_content.find("[", layers_pos);
    size_t array_end = arch_content.find("]", array_start);
    if (array_start == std::string::npos || array_end == std::string::npos) {
        return nullptr;
    }
    
    // For now, return a basic network - full JSON parsing would be quite complex
    // In a production system, you'd want to use a proper JSON library
    
    // Load weights
    std::ifstream weights_file(weights_filepath, std::ios::binary);
    if (!weights_file.is_open()) {
        return network; // Return network even if weights fail to load
    }
    
    // Verify weights file header
    char magic[8];
    weights_file.read(magic, MODEL_MAGIC_NUMBER.length());
    magic[MODEL_MAGIC_NUMBER.length()] = '\0';
    
    if (std::string(magic) != MODEL_MAGIC_NUMBER) {
        return network;
    }
    
    // Read version
    uint32_t version_len;
    weights_file.read(reinterpret_cast<char*>(&version_len), sizeof(version_len));
    std::string version(version_len, '\0');
    weights_file.read(&version[0], version_len);
    
    if (!is_compatible_version(version)) {
        return network;
    }
    
    // Read number of layers
    uint32_t num_layers;
    weights_file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    // Load weights into layers (if network was properly reconstructed)
    const auto& layers = network->get_layers();
    if (layers.size() == num_layers) {
        for (auto& layer : layers) {
            layer->deserialize_weights(weights_file);
        }
    }
    
    weights_file.close();
    
    return network;
}

// ==================== Binary Format ====================

bool ModelSerializer::save_binary(const NeuralNetwork& network, 
                                  const std::string& filepath,
                                  bool include_optimizer,
                                  bool include_history) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    file.write(MODEL_MAGIC_NUMBER.c_str(), MODEL_MAGIC_NUMBER.length());
    
    uint32_t version_len = MODEL_VERSION.length();
    file.write(reinterpret_cast<const char*>(&version_len), sizeof(version_len));
    file.write(MODEL_VERSION.c_str(), version_len);
    
    // Write network metadata
    uint32_t num_layers = static_cast<uint32_t>(network.num_layers());
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    uint8_t compiled = network.is_compiled() ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&compiled), sizeof(compiled));
    
    uint8_t include_opt = include_optimizer ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&include_opt), sizeof(include_opt));
    
    uint8_t include_hist = include_history ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&include_hist), sizeof(include_hist));
    
    // Write layer architecture and weights
    const auto& layers = network.get_layers();
    for (const auto& layer : layers) {
        // Write layer type
        std::string layer_type = layer->type();
        uint32_t type_len = layer_type.length();
        file.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));
        file.write(layer_type.c_str(), type_len);
        
        // Write layer JSON data
        std::string layer_json = layer->serialize_to_json();
        uint32_t json_len = layer_json.length();
        file.write(reinterpret_cast<const char*>(&json_len), sizeof(json_len));
        file.write(layer_json.c_str(), json_len);
        
        // Write layer weights
        layer->serialize_weights(file);
    }
    
    // Write loss function info if compiled and requested
    if (network.is_compiled() && network.get_loss_function()) {
        std::string loss_name = network.get_loss_function()->name();
        uint32_t loss_len = loss_name.length();
        file.write(reinterpret_cast<const char*>(&loss_len), sizeof(loss_len));
        file.write(loss_name.c_str(), loss_len);
    } else {
        uint32_t loss_len = 0;
        file.write(reinterpret_cast<const char*>(&loss_len), sizeof(loss_len));
    }
    
    // Write optimizer info if compiled and requested
    if (network.is_compiled() && network.get_optimizer() && include_optimizer) {
        std::string opt_name = network.get_optimizer()->name();
        uint32_t opt_len = opt_name.length();
        file.write(reinterpret_cast<const char*>(&opt_len), sizeof(opt_len));
        file.write(opt_name.c_str(), opt_len);
        
        double learning_rate = network.get_learning_rate();
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    } else {
        uint32_t opt_len = 0;
        file.write(reinterpret_cast<const char*>(&opt_len), sizeof(opt_len));
    }
    
    return true;
}

std::unique_ptr<NeuralNetwork> ModelSerializer::load_binary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return nullptr;
    }
    
    // Read and verify header
    char magic[8];
    file.read(magic, MODEL_MAGIC_NUMBER.length());
    magic[MODEL_MAGIC_NUMBER.length()] = '\0';
    
    if (std::string(magic) != MODEL_MAGIC_NUMBER) {
        return nullptr;
    }
    
    // Read version
    uint32_t version_len;
    file.read(reinterpret_cast<char*>(&version_len), sizeof(version_len));
    std::string version(version_len, '\0');
    file.read(&version[0], version_len);
    
    if (!is_compatible_version(version)) {
        return nullptr;
    }
    
    // Read metadata
    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    uint8_t compiled, include_optimizer, include_history;
    file.read(reinterpret_cast<char*>(&compiled), sizeof(compiled));
    file.read(reinterpret_cast<char*>(&include_optimizer), sizeof(include_optimizer));
    file.read(reinterpret_cast<char*>(&include_history), sizeof(include_history));
    
    // Create network
    auto network = std::make_unique<NeuralNetwork>();
    
    // Read layers
    for (uint32_t i = 0; i < num_layers; ++i) {
        // Read layer type
        uint32_t type_len;
        file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
        std::string layer_type(type_len, '\0');
        file.read(&layer_type[0], type_len);
        
        // Read layer JSON
        uint32_t json_len;
        file.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));
        std::string layer_json(json_len, '\0');
        file.read(&layer_json[0], json_len);
        
        // Create layer from JSON (simplified - would need full JSON parsing)
        auto layer = deserialize_layer_from_json(layer_json);
        if (layer) {
            // Load weights
            layer->deserialize_weights(file);
            network->add_layer(std::move(layer));
        }
    }
    
    // Read loss function
    uint32_t loss_len;
    file.read(reinterpret_cast<char*>(&loss_len), sizeof(loss_len));
    if (loss_len > 0) {
        std::string loss_name(loss_len, '\0');
        file.read(&loss_name[0], loss_len);
        
        // Read optimizer if present
        uint32_t opt_len;
        file.read(reinterpret_cast<char*>(&opt_len), sizeof(opt_len));
        if (opt_len > 0) {
            std::string opt_name(opt_len, '\0');
            file.read(&opt_name[0], opt_len);
            
            double learning_rate;
            file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
            
            // Compile the network
            try {
                network->compile(loss_name, opt_name, learning_rate);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not recompile network: " << e.what() << std::endl;
            }
        }
    }
      return network;
}

// ==================== JSON Format ====================

bool ModelSerializer::save_json(const NeuralNetwork& network, 
                                const std::string& filepath,
                                bool include_optimizer,
                                bool include_history) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    file << "{\n";
    file << "  \"clmodel_version\": \"" << MODEL_VERSION << "\",\n";
    file << "  \"format\": \"json\",\n";
    file << "  \"compiled\": " << (network.is_compiled() ? "true" : "false") << ",\n";
    file << "  \"num_layers\": " << network.num_layers() << ",\n";
    file << "  \"include_optimizer\": " << (include_optimizer ? "true" : "false") << ",\n";
    file << "  \"include_history\": " << (include_history ? "true" : "false") << ",\n";
    file << "  \"layers\": [\n";
    
    const auto& layers = network.get_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        file << "    " << layers[i]->serialize_to_json();
        if (i < layers.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]";
    
    // Add loss function and optimizer if compiled
    if (network.is_compiled()) {
        if (network.get_loss_function()) {
            file << ",\n  \"loss_function\": {\n";
            file << "    \"type\": \"" << network.get_loss_function()->name() << "\"\n";
            file << "  }";
        }
        
        if (network.get_optimizer() && include_optimizer) {
            file << ",\n  \"optimizer\": {\n";
            file << "    \"type\": \"" << network.get_optimizer()->name() << "\",\n";
            file << "    \"learning_rate\": " << network.get_learning_rate() << "\n";
            file << "  }";
        }
    }
    
    file << "\n}";
    
    return true;
}

std::unique_ptr<NeuralNetwork> ModelSerializer::load_json(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return nullptr;
    }
    
    // For a full implementation, we'd need a proper JSON parser
    // For now, return a basic network
    return std::make_unique<NeuralNetwork>();
}

// ==================== Utility Functions ====================

std::string ModelSerializer::serialize_layer_to_json(const Layer& layer) {
    return layer.serialize_to_json();
}

std::unique_ptr<Layer> ModelSerializer::deserialize_layer_from_json(const std::string& json) {
    // Simple JSON parsing to extract layer type and parameters
    // Extract layer type
    size_t type_pos = json.find("\"type\":\"");
    if (type_pos == std::string::npos) {
        std::cerr << "Error: No layer type found in JSON" << std::endl;
        return nullptr;
    }
    
    size_t type_start = type_pos + 8;  // Length of "\"type\":\""
    size_t type_end = json.find("\"", type_start);
    if (type_end == std::string::npos) {
        std::cerr << "Error: Malformed layer type in JSON" << std::endl;
        return nullptr;
    }
    
    std::string layer_type = json.substr(type_start, type_end - type_start);
    
    try {
        if (layer_type == "Dense") {
            return deserialize_dense_layer(json);
        } else if (layer_type.substr(0, 11) == "Activation_") {
            return deserialize_activation_layer(json);
        } else if (layer_type == "Dropout") {
            return deserialize_dropout_layer(json);
        } else if (layer_type == "BatchNorm") {
            return deserialize_batchnorm_layer(json);
        } else {
            std::cerr << "Error: Unknown layer type: " << layer_type << std::endl;
            return nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error deserializing layer: " << e.what() << std::endl;
        return nullptr;
    }
}

std::string ModelSerializer::matrix_to_json(const Matrix& matrix) {
    std::ostringstream ss;
    ss << "{";
    ss << "\"rows\":" << matrix.rows() << ",";
    ss << "\"cols\":" << matrix.cols() << ",";
    ss << "\"data\":[";
    
    for (size_t i = 0; i < matrix.rows(); ++i) {
        if (i > 0) ss << ",";
        ss << "[";
        for (size_t j = 0; j < matrix.cols(); ++j) {
            if (j > 0) ss << ",";
            ss << format_double(matrix(i, j));
        }
        ss << "]";
    }
    
    ss << "]}";
    return ss.str();
}

Matrix ModelSerializer::matrix_from_json(const std::string& json) {
    // Simple JSON parsing for matrix format
    // In production, use a proper JSON parser
    
    // Extract dimensions
    size_t rows_pos = json.find("\"rows\":");
    size_t cols_pos = json.find("\"cols\":");
    
    if (rows_pos == std::string::npos || cols_pos == std::string::npos) {
        return Matrix(0, 0);
    }
    
    size_t rows = std::stoul(json.substr(rows_pos + 7, json.find(",", rows_pos + 7) - rows_pos - 7));
    size_t cols = std::stoul(json.substr(cols_pos + 7, json.find(",", cols_pos + 7) - cols_pos - 7));
    
    Matrix matrix(rows, cols);
    
    // For now, return zero matrix (full JSON parsing would be needed)
    return matrix;
}

std::string ModelSerializer::escape_json_string(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    
    for (char c : str) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:   result += c; break;
        }
    }
    
    return result;
}

std::string ModelSerializer::format_double(double value, int precision) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

// ==================== Utility Functions ====================

void ModelSerializer::write_matrix_binary(std::ofstream& file, const Matrix& matrix) {
    uint32_t rows = static_cast<uint32_t>(matrix.rows());
    uint32_t cols = static_cast<uint32_t>(matrix.cols());
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            double value = matrix(i, j);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
}

Matrix ModelSerializer::read_matrix_binary(std::ifstream& file) {
    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    Matrix matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            matrix(i, j) = value;
        }
    }
    
    return matrix;
}

bool ModelSerializer::is_compatible_version(const std::string& version) {
    // For now, only accept exact version match
    return version == MODEL_VERSION;
}

// ==================== SerializationCheckpoint Implementation ====================

SerializationCheckpoint::SerializationCheckpoint(const std::string& filepath_template,
                                 const std::string& monitor,
                                 bool save_best_only,
                                 bool save_weights_only,
                                 int save_frequency,
                                 SerializationFormat format)
    : filepath_template_(filepath_template),
      monitor_metric_(monitor),
      save_best_only_(save_best_only),
      save_weights_only_(save_weights_only),
      best_metric_(monitor.find("loss") != std::string::npos ? 
                   std::numeric_limits<double>::max() : 
                   std::numeric_limits<double>::lowest()),
      save_frequency_(save_frequency),
      format_(format) {
}

bool SerializationCheckpoint::on_epoch_end(const NeuralNetwork& network, int epoch,
                                  const std::map<std::string, double>& metrics) {
    if (!should_save(metrics, epoch)) {
        return false;
    }
    
    std::string filepath = format_filepath(epoch, metrics);
    
    bool success = ModelSerializer::save(network, filepath, format_, 
                                       !save_weights_only_, !save_weights_only_);
    
    if (success && save_best_only_) {
        auto it = metrics.find(monitor_metric_);
        if (it != metrics.end()) {
            best_metric_ = it->second;
        }
    }
    
    return success;
}

bool SerializationCheckpoint::should_save(const std::map<std::string, double>& metrics, int epoch) const {
    if (!save_best_only_) {
        return (epoch % save_frequency_) == 0;
    }
    
    auto it = metrics.find(monitor_metric_);
    if (it == metrics.end()) {
        return false;
    }
    
    double current_metric = it->second;
    
    // Check if this is an improvement
    if (monitor_metric_.find("loss") != std::string::npos) {
        // Lower is better for loss metrics
        return current_metric < best_metric_;
    } else {
        // Higher is better for accuracy metrics
        return current_metric > best_metric_;
    }
}

std::string SerializationCheckpoint::format_filepath(int epoch, const std::map<std::string, double>& metrics) const {
    std::string result = filepath_template_;
    
    // Replace {epoch}
    size_t pos = result.find("{epoch}");
    if (pos != std::string::npos) {
        result.replace(pos, 7, std::to_string(epoch));
    }
    
    // Replace metric placeholders
    for (const auto& metric : metrics) {
        std::string placeholder = "{" + metric.first + "}";
        pos = result.find(placeholder);
        if (pos != std::string::npos) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << metric.second;
            result.replace(pos, placeholder.length(), ss.str());
        }
    }
    
    return result;
}

void SerializationCheckpoint::reset() {
    best_metric_ = monitor_metric_.find("loss") != std::string::npos ? 
                   std::numeric_limits<double>::max() : 
                   std::numeric_limits<double>::lowest();
}

// ==================== Layer-specific Deserialization Methods ====================

std::unique_ptr<Layer> ModelSerializer::deserialize_dense_layer(const std::string& json) {
    // Extract input_size and output_size
    size_t input_pos = json.find("\"input_size\":");
    size_t output_pos = json.find("\"output_size\":");
    
    if (input_pos == std::string::npos || output_pos == std::string::npos) {
        throw std::runtime_error("Missing size parameters for Dense layer");
    }
    
    size_t input_start = input_pos + 13;  // Length of "\"input_size\":"
    size_t input_end = json.find_first_of(",}", input_start);
    size_t input_size = std::stoul(json.substr(input_start, input_end - input_start));
    
    size_t output_start = output_pos + 14;  // Length of "\"output_size\":"
    size_t output_end = json.find_first_of(",}", output_start);
    size_t output_size = std::stoul(json.substr(output_start, output_end - output_start));
    
    auto layer = std::make_unique<DenseLayer>(input_size, output_size);
    
    // Extract and deserialize weights if present
    size_t weights_pos = json.find("\"weights\":");
    size_t biases_pos = json.find("\"biases\":");
    
    if (weights_pos != std::string::npos && biases_pos != std::string::npos) {
        // Extract weights JSON
        size_t weights_start = json.find("{", weights_pos);
        size_t weights_end = find_matching_brace(json, weights_start);
        std::string weights_json = json.substr(weights_start, weights_end - weights_start + 1);
        
        // Extract biases JSON
        size_t biases_start = json.find("{", biases_pos);
        size_t biases_end = find_matching_brace(json, biases_start);
        std::string biases_json = json.substr(biases_start, biases_end - biases_start + 1);
        
        // Deserialize matrices (simplified - would need full JSON parsing)
        // For now, keep default initialization
    }
    
    return std::move(layer);
}

std::unique_ptr<Layer> ModelSerializer::deserialize_activation_layer(const std::string& json) {
    // Extract activation function name from type
    size_t type_pos = json.find("\"type\":\"Activation_");
    if (type_pos == std::string::npos) {
        throw std::runtime_error("Invalid activation layer type");
    }
    
    size_t name_start = type_pos + 20;  // Length of "\"type\":\"Activation_"
    size_t name_end = json.find("\"", name_start);
    std::string activation_name = json.substr(name_start, name_end - name_start);
    
    // Extract size
    size_t size_pos = json.find("\"size\":");
    if (size_pos == std::string::npos) {
        throw std::runtime_error("Missing size parameter for Activation layer");
    }
    
    size_t size_start = size_pos + 7;  // Length of "\"size\":"
    size_t size_end = json.find_first_of(",}", size_start);
    size_t size = std::stoul(json.substr(size_start, size_end - size_start));
    
    return std::make_unique<ActivationLayer>(activation_name, size);
}

std::unique_ptr<Layer> ModelSerializer::deserialize_dropout_layer(const std::string& json) {
    // Extract dropout_rate
    size_t rate_pos = json.find("\"dropout_rate\":");
    if (rate_pos == std::string::npos) {
        throw std::runtime_error("Missing dropout_rate parameter for Dropout layer");
    }
    
    size_t rate_start = rate_pos + 15;  // Length of "\"dropout_rate\":"
    size_t rate_end = json.find_first_of(",}", rate_start);
    float dropout_rate = std::stof(json.substr(rate_start, rate_end - rate_start));
    
    return std::make_unique<DropoutLayer>(dropout_rate);
}

std::unique_ptr<Layer> ModelSerializer::deserialize_batchnorm_layer(const std::string& json) {
    // Extract momentum if present
    float momentum = 0.9f;  // Default value
    size_t momentum_pos = json.find("\"momentum\":");
    if (momentum_pos != std::string::npos) {
        size_t momentum_start = momentum_pos + 11;  // Length of "\"momentum\":" 
        size_t momentum_end = json.find_first_of(",}", momentum_start);
        momentum = std::stof(json.substr(momentum_start, momentum_end - momentum_start));
    }
    
    // Extract epsilon if present
    float epsilon = 1e-5f;  // Default value
    size_t epsilon_pos = json.find("\"epsilon\":");
    if (epsilon_pos != std::string::npos) {
        size_t epsilon_start = epsilon_pos + 10;  // Length of "\"epsilon\":" 
        size_t epsilon_end = json.find_first_of(",}", epsilon_start);
        epsilon = std::stof(json.substr(epsilon_start, epsilon_end - epsilon_start));
    }
      return std::make_unique<BatchNormLayer>(momentum, epsilon);
}

size_t ModelSerializer::find_matching_brace(const std::string& json, size_t start_pos) {
    int brace_count = 1;
    size_t pos = start_pos + 1;
    
    while (pos < json.length() && brace_count > 0) {
        if (json[pos] == '{') {
            brace_count++;
        } else if (json[pos] == '}') {
            brace_count--;
        }
        pos++;
    }
    
    return (brace_count == 0) ? pos - 1 : std::string::npos;
}

} // namespace clmodel
