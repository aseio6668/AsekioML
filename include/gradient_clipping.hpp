#pragma once

#include "matrix.hpp"
#include <cmath>
#include <algorithm>

namespace clmodel {

namespace gradient_clipping {

// Gradient clipping by value
void clip_by_value(Matrix& gradients, double min_value, double max_value) {
    for (size_t i = 0; i < gradients.rows(); ++i) {
        for (size_t j = 0; j < gradients.cols(); ++j) {
            gradients(i, j) = std::max(min_value, std::min(max_value, gradients(i, j)));
        }
    }
}

// Gradient clipping by norm (L2)
void clip_by_norm(Matrix& gradients, double max_norm) {
    // Compute L2 norm
    double norm = 0.0;
    for (size_t i = 0; i < gradients.rows(); ++i) {
        for (size_t j = 0; j < gradients.cols(); ++j) {
            norm += gradients(i, j) * gradients(i, j);
        }
    }
    norm = std::sqrt(norm);
    
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (size_t i = 0; i < gradients.rows(); ++i) {
            for (size_t j = 0; j < gradients.cols(); ++j) {
                gradients(i, j) *= scale;
            }
        }
    }
}

// Gradient clipping by global norm (for multiple gradient matrices)
void clip_by_global_norm(std::vector<Matrix*>& gradient_matrices, double max_norm) {
    // Compute global norm
    double global_norm = 0.0;
    for (auto* gradients : gradient_matrices) {
        for (size_t i = 0; i < gradients->rows(); ++i) {
            for (size_t j = 0; j < gradients->cols(); ++j) {
                double val = (*gradients)(i, j);
                global_norm += val * val;
            }
        }
    }
    global_norm = std::sqrt(global_norm);
    
    if (global_norm > max_norm) {
        double scale = max_norm / global_norm;
        for (auto* gradients : gradient_matrices) {
            for (size_t i = 0; i < gradients->rows(); ++i) {
                for (size_t j = 0; j < gradients->cols(); ++j) {
                    (*gradients)(i, j) *= scale;
                }
            }
        }
    }
}

} // namespace gradient_clipping

} // namespace clmodel
