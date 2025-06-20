#pragma once

#include "matrix.hpp"
#include <vector>
#include <string>
#include <fstream>

namespace clmodel {

class Dataset {
private:
    Matrix features_;
    Matrix targets_;
    std::vector<std::string> feature_names_;
    std::vector<std::string> target_names_;

public:
    Dataset() = default;
    Dataset(const Matrix& features, const Matrix& targets);
    Dataset(const Matrix& features, const Matrix& targets,
            const std::vector<std::string>& feature_names,
            const std::vector<std::string>& target_names);
    
    // Data access
    const Matrix& features() const { return features_; }
    const Matrix& targets() const { return targets_; }
    Matrix& features() { return features_; }
    Matrix& targets() { return targets_; }
    
    // Dataset properties
    size_t size() const { return features_.rows(); }
    size_t num_features() const { return features_.cols(); }
    size_t num_targets() const { return targets_.cols(); }
    
    // Data manipulation
    void shuffle();
    Dataset sample(size_t n) const;
    std::pair<Dataset, Dataset> train_test_split(double test_size = 0.2) const;
    
    // Normalization
    void normalize_features(); // Z-score normalization
    void normalize_features_minmax(); // Min-max normalization
    void normalize_targets();
    
    // Data loading/saving
    static Dataset load_csv(const std::string& filename, bool has_header = true, 
                           char delimiter = ',', int target_columns = 1);
    void save_csv(const std::string& filename, bool include_header = true, 
                  char delimiter = ',') const;
    
    // Feature engineering
    void add_polynomial_features(int degree = 2);
    void add_bias_column(); // Add column of ones
    
    // Statistics
    Matrix feature_means() const;
    Matrix feature_std_devs() const;
    Matrix feature_min() const;
    Matrix feature_max() const;
    
    // Utility
    void info() const;
    Dataset slice(size_t start, size_t end) const;
    void append(const Dataset& other);

private:
    void validate_data() const;
};

// Utility functions for creating synthetic datasets
namespace datasets {

// Create a simple linear regression dataset
Dataset make_regression(size_t n_samples = 100, size_t n_features = 1, 
                       double noise = 0.1, int random_state = 42);

// Create a classification dataset
Dataset make_classification(size_t n_samples = 100, size_t n_features = 2,
                           size_t n_classes = 2, double noise = 0.1,
                           int random_state = 42);

// Create XOR dataset (classic non-linear problem)
Dataset make_xor(size_t n_samples = 100, double noise = 0.1);

// Create circles dataset
Dataset make_circles(size_t n_samples = 100, double noise = 0.1, 
                    double factor = 0.8);

} // namespace datasets

} // namespace clmodel
