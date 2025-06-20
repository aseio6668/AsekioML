#include "loss.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace clmodel {

// MeanSquaredError Implementation
double MeanSquaredError::compute_loss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    Matrix diff = predictions - targets;
    double sum_squared = 0.0;
    
    for (size_t i = 0; i < diff.rows(); ++i) {
        for (size_t j = 0; j < diff.cols(); ++j) {
            sum_squared += diff[i][j] * diff[i][j];
        }
    }
    
    return sum_squared / (2.0 * predictions.rows()); // Average over batch size
}

Matrix MeanSquaredError::compute_gradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    Matrix diff = predictions - targets;
    double scale = 1.0 / predictions.rows();
    return diff * scale; // Average over batch size
}

std::unique_ptr<LossFunction> MeanSquaredError::clone() const {
    return std::make_unique<MeanSquaredError>();
}

// CrossEntropyLoss Implementation
double CrossEntropyLoss::compute_loss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    double total_loss = 0.0;
    const double epsilon = 1e-15; // To prevent log(0)
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions[i][j]));
            total_loss -= targets[i][j] * std::log(pred);
        }
    }
    
    return total_loss / predictions.rows(); // Average over batch size
}

Matrix CrossEntropyLoss::compute_gradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    const double epsilon = 1e-15;
    Matrix gradient(predictions.rows(), predictions.cols());
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions[i][j]));
            gradient[i][j] = (-targets[i][j] / pred) / predictions.rows();
        }
    }
    
    return gradient;
}

std::unique_ptr<LossFunction> CrossEntropyLoss::clone() const {
    return std::make_unique<CrossEntropyLoss>();
}

// BinaryCrossEntropyLoss Implementation
double BinaryCrossEntropyLoss::compute_loss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    double total_loss = 0.0;
    const double epsilon = 1e-15;
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions[i][j]));
            double target = targets[i][j];
            total_loss -= target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred);
        }
    }
    
    return total_loss / predictions.rows();
}

Matrix BinaryCrossEntropyLoss::compute_gradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    const double epsilon = 1e-15;
    Matrix gradient(predictions.rows(), predictions.cols());
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions[i][j]));
            double target = targets[i][j];
            gradient[i][j] = (-(target / pred) + (1.0 - target) / (1.0 - pred)) / predictions.rows();
        }
    }
    
    return gradient;
}

std::unique_ptr<LossFunction> BinaryCrossEntropyLoss::clone() const {
    return std::make_unique<BinaryCrossEntropyLoss>();
}

// MeanAbsoluteError Implementation
double MeanAbsoluteError::compute_loss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    double total_loss = 0.0;
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            total_loss += std::abs(predictions[i][j] - targets[i][j]);
        }
    }
    
    return total_loss / predictions.rows();
}

Matrix MeanAbsoluteError::compute_gradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    Matrix gradient(predictions.rows(), predictions.cols());
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            double diff = predictions[i][j] - targets[i][j];
            gradient[i][j] = (diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0)) / predictions.rows();
        }
    }
    
    return gradient;
}

std::unique_ptr<LossFunction> MeanAbsoluteError::clone() const {
    return std::make_unique<MeanAbsoluteError>();
}

// HuberLoss Implementation
double HuberLoss::compute_loss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    double total_loss = 0.0;
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            double diff = std::abs(predictions[i][j] - targets[i][j]);
            if (diff <= delta_) {
                total_loss += 0.5 * diff * diff;
            } else {
                total_loss += delta_ * diff - 0.5 * delta_ * delta_;
            }
        }
    }
    
    return total_loss / predictions.rows();
}

Matrix HuberLoss::compute_gradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    
    Matrix gradient(predictions.rows(), predictions.cols());
    
    for (size_t i = 0; i < predictions.rows(); ++i) {
        for (size_t j = 0; j < predictions.cols(); ++j) {
            double diff = predictions[i][j] - targets[i][j];
            double abs_diff = std::abs(diff);
            
            if (abs_diff <= delta_) {
                gradient[i][j] = diff / predictions.rows();
            } else {
                gradient[i][j] = (diff > 0.0 ? delta_ : -delta_) / predictions.rows();
            }
        }
    }
    
    return gradient;
}

std::unique_ptr<LossFunction> HuberLoss::clone() const {
    return std::make_unique<HuberLoss>(delta_);
}

// Factory function
std::unique_ptr<LossFunction> create_loss_function(const std::string& name) {
    if (name == "mse" || name == "mean_squared_error") {
        return std::make_unique<MeanSquaredError>();
    } else if (name == "cross_entropy" || name == "categorical_crossentropy") {
        return std::make_unique<CrossEntropyLoss>();
    } else if (name == "binary_cross_entropy" || name == "binary_crossentropy") {
        return std::make_unique<BinaryCrossEntropyLoss>();
    } else if (name == "mae" || name == "mean_absolute_error") {
        return std::make_unique<MeanAbsoluteError>();
    } else if (name == "huber") {
        return std::make_unique<HuberLoss>();
    } else {
        throw std::invalid_argument("Unknown loss function: " + name);
    }
}

} // namespace clmodel
