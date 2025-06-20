#include "dataset.hpp"
#include <random>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace clmodel {

Dataset::Dataset(const Matrix& features, const Matrix& targets)
    : features_(features), targets_(targets) {
    validate_data();
}

Dataset::Dataset(const Matrix& features, const Matrix& targets,
                const std::vector<std::string>& feature_names,
                const std::vector<std::string>& target_names)
    : features_(features), targets_(targets),
      feature_names_(feature_names), target_names_(target_names) {
    validate_data();
}

void Dataset::validate_data() const {
    if (features_.rows() != targets_.rows()) {
        throw std::invalid_argument("Features and targets must have the same number of samples");
    }
    
    if (!feature_names_.empty() && feature_names_.size() != features_.cols()) {
        throw std::invalid_argument("Number of feature names must match number of features");
    }
    
    if (!target_names_.empty() && target_names_.size() != targets_.cols()) {
        throw std::invalid_argument("Number of target names must match number of targets");
    }
}

void Dataset::shuffle() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::vector<size_t> indices(size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    Matrix new_features(size(), num_features());
    Matrix new_targets(size(), num_targets());
    
    for (size_t i = 0; i < size(); ++i) {
        for (size_t j = 0; j < num_features(); ++j) {
            new_features[i][j] = features_[indices[i]][j];
        }
        for (size_t j = 0; j < num_targets(); ++j) {
            new_targets[i][j] = targets_[indices[i]][j];
        }
    }
    
    features_ = std::move(new_features);
    targets_ = std::move(new_targets);
}

Dataset Dataset::sample(size_t n) const {
    if (n > size()) {
        throw std::invalid_argument("Sample size cannot be larger than dataset size");
    }
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::vector<size_t> indices(size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    Matrix sample_features(n, num_features());
    Matrix sample_targets(n, num_targets());
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < num_features(); ++j) {
            sample_features[i][j] = features_[indices[i]][j];
        }
        for (size_t j = 0; j < num_targets(); ++j) {
            sample_targets[i][j] = targets_[indices[i]][j];
        }
    }
    
    return Dataset(sample_features, sample_targets, feature_names_, target_names_);
}

std::pair<Dataset, Dataset> Dataset::train_test_split(double test_size) const {
    if (test_size <= 0.0 || test_size >= 1.0) {
        throw std::invalid_argument("Test size must be between 0 and 1");
    }
    
    size_t train_size = static_cast<size_t>((1.0 - test_size) * size());
    
    Dataset train_set = slice(0, train_size);
    Dataset test_set = slice(train_size, size());
    
    return {train_set, test_set};
}

void Dataset::normalize_features() {
    Matrix means = feature_means();
    Matrix std_devs = feature_std_devs();
    
    for (size_t i = 0; i < features_.rows(); ++i) {
        for (size_t j = 0; j < features_.cols(); ++j) {
            if (std_devs[0][j] > 1e-8) { // Avoid division by zero
                features_[i][j] = (features_[i][j] - means[0][j]) / std_devs[0][j];
            }
        }
    }
}

void Dataset::normalize_features_minmax() {
    Matrix min_vals = feature_min();
    Matrix max_vals = feature_max();
    
    for (size_t i = 0; i < features_.rows(); ++i) {
        for (size_t j = 0; j < features_.cols(); ++j) {
            double range = max_vals[0][j] - min_vals[0][j];
            if (range > 1e-8) { // Avoid division by zero
                features_[i][j] = (features_[i][j] - min_vals[0][j]) / range;
            }
        }
    }
}

void Dataset::normalize_targets() {
    double mean = targets_.mean();
    double std_dev = targets_.std_dev();
    
    if (std_dev > 1e-8) {
        for (size_t i = 0; i < targets_.rows(); ++i) {
            for (size_t j = 0; j < targets_.cols(); ++j) {
                targets_[i][j] = (targets_[i][j] - mean) / std_dev;
            }
        }
    }
}

Dataset Dataset::load_csv(const std::string& filename, bool has_header, 
                         char delimiter, int target_columns) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::vector<std::string> feature_names;
    std::vector<std::string> target_names;
    std::string line;
    
    // Read header if present
    if (has_header && std::getline(file, line)) {
        std::stringstream ss(line);
        std::string column;
        std::vector<std::string> headers;
        
        while (std::getline(ss, column, delimiter)) {
            headers.push_back(column);
        }
        
        size_t num_features = headers.size() - target_columns;
        feature_names.assign(headers.begin(), headers.begin() + num_features);
        target_names.assign(headers.begin() + num_features, headers.end());
    }
    
    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        
        while (std::getline(ss, value, delimiter)) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                throw std::runtime_error("Invalid numeric value in CSV: " + value);
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    if (data.empty()) {
        throw std::runtime_error("No data found in file");
    }
    
    size_t num_features = data[0].size() - target_columns;
    size_t num_samples = data.size();
    
    Matrix features(num_samples, num_features);
    Matrix targets(num_samples, target_columns);
    
    for (size_t i = 0; i < num_samples; ++i) {
        if (data[i].size() != num_features + target_columns) {
            throw std::runtime_error("Inconsistent number of columns in row " + std::to_string(i));
        }
        
        for (size_t j = 0; j < num_features; ++j) {
            features[i][j] = data[i][j];
        }
        
        for (size_t j = 0; j < target_columns; ++j) {
            targets[i][j] = data[i][num_features + j];
        }
    }
    
    return Dataset(features, targets, feature_names, target_names);
}

void Dataset::save_csv(const std::string& filename, bool include_header, char delimiter) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    if (include_header) {
        for (size_t i = 0; i < feature_names_.size(); ++i) {
            file << feature_names_[i];
            if (i < feature_names_.size() - 1 || !target_names_.empty()) {
                file << delimiter;
            }
        }
        
        for (size_t i = 0; i < target_names_.size(); ++i) {
            file << target_names_[i];
            if (i < target_names_.size() - 1) {
                file << delimiter;
            }
        }
        file << "\n";
    }
    
    // Write data
    for (size_t i = 0; i < size(); ++i) {
        for (size_t j = 0; j < num_features(); ++j) {
            file << features_[i][j];
            if (j < num_features() - 1 || num_targets() > 0) {
                file << delimiter;
            }
        }
        
        for (size_t j = 0; j < num_targets(); ++j) {
            file << targets_[i][j];
            if (j < num_targets() - 1) {
                file << delimiter;
            }
        }
        file << "\n";
    }
}

void Dataset::add_bias_column() {
    Matrix new_features(size(), num_features() + 1);
    
    // Copy existing features
    for (size_t i = 0; i < size(); ++i) {
        new_features[i][0] = 1.0; // Bias column
        for (size_t j = 0; j < num_features(); ++j) {
            new_features[i][j + 1] = features_[i][j];
        }
    }
    
    features_ = std::move(new_features);
    feature_names_.insert(feature_names_.begin(), "bias");
}

Matrix Dataset::feature_means() const {
    return features_.sum_cols() * (1.0 / features_.rows());
}

Matrix Dataset::feature_std_devs() const {
    Matrix means = feature_means();
    Matrix variance(1, num_features(), 0.0);
    
    for (size_t i = 0; i < features_.rows(); ++i) {
        for (size_t j = 0; j < features_.cols(); ++j) {
            double diff = features_[i][j] - means[0][j];
            variance[0][j] += diff * diff;
        }
    }
    
    for (size_t j = 0; j < features_.cols(); ++j) {
        variance[0][j] = std::sqrt(variance[0][j] / features_.rows());
    }
    
    return variance;
}

Matrix Dataset::feature_min() const {
    Matrix min_vals(1, num_features());
    
    for (size_t j = 0; j < features_.cols(); ++j) {
        min_vals[0][j] = features_[0][j];
        for (size_t i = 1; i < features_.rows(); ++i) {
            min_vals[0][j] = std::min(min_vals[0][j], features_[i][j]);
        }
    }
    
    return min_vals;
}

Matrix Dataset::feature_max() const {
    Matrix max_vals(1, num_features());
    
    for (size_t j = 0; j < features_.cols(); ++j) {
        max_vals[0][j] = features_[0][j];
        for (size_t i = 1; i < features_.rows(); ++i) {
            max_vals[0][j] = std::max(max_vals[0][j], features_[i][j]);
        }
    }
    
    return max_vals;
}

void Dataset::info() const {
    std::cout << "Dataset Information:" << std::endl;
    std::cout << "Number of samples: " << size() << std::endl;
    std::cout << "Number of features: " << num_features() << std::endl;
    std::cout << "Number of targets: " << num_targets() << std::endl;
    
    if (!feature_names_.empty()) {
        std::cout << "Feature names: ";
        for (size_t i = 0; i < feature_names_.size(); ++i) {
            std::cout << feature_names_[i];
            if (i < feature_names_.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    if (!target_names_.empty()) {
        std::cout << "Target names: ";
        for (size_t i = 0; i < target_names_.size(); ++i) {
            std::cout << target_names_[i];
            if (i < target_names_.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}

Dataset Dataset::slice(size_t start, size_t end) const {
    if (start >= end || end > size()) {
        throw std::invalid_argument("Invalid slice indices");
    }
    
    size_t slice_size = end - start;
    Matrix slice_features(slice_size, num_features());
    Matrix slice_targets(slice_size, num_targets());
    
    for (size_t i = 0; i < slice_size; ++i) {
        for (size_t j = 0; j < num_features(); ++j) {
            slice_features[i][j] = features_[start + i][j];
        }
        for (size_t j = 0; j < num_targets(); ++j) {
            slice_targets[i][j] = targets_[start + i][j];
        }
    }
    
    return Dataset(slice_features, slice_targets, feature_names_, target_names_);
}

void Dataset::append(const Dataset& other) {
    if (num_features() != other.num_features() || num_targets() != other.num_targets()) {
        throw std::invalid_argument("Cannot append datasets with different feature/target dimensions");
    }
    
    Matrix new_features(size() + other.size(), num_features());
    Matrix new_targets(size() + other.size(), num_targets());
    
    // Copy current data
    for (size_t i = 0; i < size(); ++i) {
        for (size_t j = 0; j < num_features(); ++j) {
            new_features[i][j] = features_[i][j];
        }
        for (size_t j = 0; j < num_targets(); ++j) {
            new_targets[i][j] = targets_[i][j];
        }
    }
    
    // Copy other data
    for (size_t i = 0; i < other.size(); ++i) {
        for (size_t j = 0; j < num_features(); ++j) {
            new_features[size() + i][j] = other.features_[i][j];
        }
        for (size_t j = 0; j < num_targets(); ++j) {
            new_targets[size() + i][j] = other.targets_[i][j];
        }
    }
    
    features_ = std::move(new_features);
    targets_ = std::move(new_targets);
}

// Synthetic dataset creation functions
namespace datasets {

Dataset make_regression(size_t n_samples, size_t n_features, double noise, int random_state) {
    std::mt19937 gen(random_state);
    std::normal_distribution<double> feature_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise);
    std::uniform_real_distribution<double> coeff_dist(-1.0, 1.0);
    
    Matrix features(n_samples, n_features);
    Matrix targets(n_samples, 1);
    
    // Generate random coefficients
    std::vector<double> coefficients(n_features);
    for (size_t i = 0; i < n_features; ++i) {
        coefficients[i] = coeff_dist(gen);
    }
    
    // Generate samples
    for (size_t i = 0; i < n_samples; ++i) {
        double target_value = 0.0;
        
        for (size_t j = 0; j < n_features; ++j) {
            features[i][j] = feature_dist(gen);
            target_value += coefficients[j] * features[i][j];
        }
        
        targets[i][0] = target_value + noise_dist(gen);
    }
    
    return Dataset(features, targets);
}

Dataset make_classification(size_t n_samples, size_t n_features, size_t n_classes, 
                           double noise, int random_state) {    std::mt19937 gen(random_state);
    std::normal_distribution<double> feature_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise);
    std::uniform_int_distribution<int> class_dist(0, static_cast<int>(n_classes) - 1);
    
    Matrix features(n_samples, n_features);
    Matrix targets(n_samples, n_classes);
    
    // Generate class centers
    std::vector<std::vector<double>> centers(n_classes, std::vector<double>(n_features));
    for (size_t i = 0; i < n_classes; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            centers[i][j] = feature_dist(gen) * 3.0; // Spread out centers
        }
    }
    
    // Generate samples
    for (size_t i = 0; i < n_samples; ++i) {
        int class_label = class_dist(gen);
        
        // Initialize targets (one-hot encoding)
        for (size_t k = 0; k < n_classes; ++k) {
            targets[i][k] = (k == class_label) ? 1.0 : 0.0;
        }
        
        // Generate features around class center
        for (size_t j = 0; j < n_features; ++j) {
            features[i][j] = centers[class_label][j] + feature_dist(gen) + noise_dist(gen);
        }
    }
    
    return Dataset(features, targets);
}

Dataset make_xor(size_t n_samples, double noise) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> coord_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise);
    
    Matrix features(n_samples, 2);
    Matrix targets(n_samples, 1);
    
    for (size_t i = 0; i < n_samples; ++i) {
        double x = coord_dist(gen);
        double y = coord_dist(gen);
        
        features[i][0] = x + noise_dist(gen);
        features[i][1] = y + noise_dist(gen);
        
        // XOR logic: (x > 0.5) XOR (y > 0.5)
        bool x_high = x > 0.5;
        bool y_high = y > 0.5;
        targets[i][0] = (x_high != y_high) ? 1.0 : 0.0;
    }
    
    return Dataset(features, targets);
}

Dataset make_circles(size_t n_samples, double noise, double factor) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::normal_distribution<double> noise_dist(0.0, noise);
    std::uniform_int_distribution<int> circle_dist(0, 1);
    
    Matrix features(n_samples, 2);
    Matrix targets(n_samples, 1);
    
    for (size_t i = 0; i < n_samples; ++i) {
        double angle = angle_dist(gen);
        int circle = circle_dist(gen);
        double radius = circle == 0 ? 1.0 : factor;
        
        features[i][0] = radius * std::cos(angle) + noise_dist(gen);
        features[i][1] = radius * std::sin(angle) + noise_dist(gen);
        targets[i][0] = static_cast<double>(circle);
    }
    
    return Dataset(features, targets);
}

} // namespace datasets

} // namespace clmodel
