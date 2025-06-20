#pragma once

#include "loss.hpp"
#include <cmath>
#include <algorithm>

namespace asekioml {

// Focal Loss for handling class imbalance
class FocalLoss : public LossFunction {
private:
    double alpha_;  // Weighting factor for rare class
    double gamma_;  // Focusing parameter
    
public:    FocalLoss(double alpha = 1.0, double gamma = 2.0) : alpha_(alpha), gamma_(gamma) {}
    
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override {
        double total_loss = 0.0;
        size_t num_samples = predictions.rows();
        
        for (size_t i = 0; i < num_samples; ++i) {
            for (size_t j = 0; j < predictions.cols(); ++j) {
                double p = predictions(i, j);
                double y = targets(i, j);
                
                // Prevent log(0)
                p = std::max(1e-15, std::min(1.0 - 1e-15, p));
                
                double ce_loss = -y * std::log(p) - (1.0 - y) * std::log(1.0 - p);
                double pt = y * p + (1.0 - y) * (1.0 - p);
                double focal_weight = alpha_ * std::pow(1.0 - pt, gamma_);
                
                total_loss += focal_weight * ce_loss;
            }
        }
          return total_loss / num_samples;
    }
    
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override {
        Matrix gradients(predictions.rows(), predictions.cols());
        
        for (size_t i = 0; i < predictions.rows(); ++i) {
            for (size_t j = 0; j < predictions.cols(); ++j) {
                double p = predictions(i, j);
                double y = targets(i, j);
                
                // Prevent division by 0
                p = std::max(1e-15, std::min(1.0 - 1e-15, p));
                
                double pt = y * p + (1.0 - y) * (1.0 - p);
                double focal_weight = alpha_ * std::pow(1.0 - pt, gamma_);
                
                // Gradient computation
                double grad = -y / p + (1.0 - y) / (1.0 - p);
                double focal_grad = gamma_ * alpha_ * std::pow(1.0 - pt, gamma_ - 1.0) * 
                                   (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
                
                gradients(i, j) = focal_weight * grad + focal_grad;
            }
        }
        
        return gradients;    }
    
    std::unique_ptr<LossFunction> clone() const override {
        return std::make_unique<FocalLoss>(alpha_, gamma_);
    }
    
    std::string name() const override { return "FocalLoss"; }
};

// Label Smoothing Cross Entropy
class LabelSmoothingCrossEntropy : public LossFunction {
private:
    double smoothing_;
    
public:    LabelSmoothingCrossEntropy(double smoothing = 0.1) : smoothing_(smoothing) {}
    
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override {
        double total_loss = 0.0;
        size_t num_samples = predictions.rows();
        size_t num_classes = predictions.cols();
        
        for (size_t i = 0; i < num_samples; ++i) {
            for (size_t j = 0; j < num_classes; ++j) {
                double p = std::max(1e-15, std::min(1.0 - 1e-15, predictions(i, j)));
                double y_smooth = targets(i, j) * (1.0 - smoothing_) + smoothing_ / num_classes;                total_loss -= y_smooth * std::log(p);
            }
        }
        
        return total_loss / num_samples;
    }
    
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override {
        Matrix gradients(predictions.rows(), predictions.cols());
        size_t num_classes = predictions.cols();
        
        for (size_t i = 0; i < predictions.rows(); ++i) {
            for (size_t j = 0; j < predictions.cols(); ++j) {
                double p = std::max(1e-15, std::min(1.0 - 1e-15, predictions(i, j)));
                double y_smooth = targets(i, j) * (1.0 - smoothing_) + smoothing_ / num_classes;
                gradients(i, j) = -y_smooth / p;
            }
        }
        
        return gradients;
    }
      std::unique_ptr<LossFunction> clone() const override {
        return std::make_unique<LabelSmoothingCrossEntropy>(smoothing_);
    }
    
    std::string name() const override { return "LabelSmoothingCrossEntropy"; }
};

// Dice Loss (for segmentation)
class DiceLoss : public LossFunction {
private:
    double smooth_;
    
public:    DiceLoss(double smooth = 1.0) : smooth_(smooth) {}
    
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override {
        double intersection = 0.0;
        double pred_sum = 0.0;
        double target_sum = 0.0;
        
        for (size_t i = 0; i < predictions.rows(); ++i) {
            for (size_t j = 0; j < predictions.cols(); ++j) {
                double p = predictions(i, j);
                double t = targets(i, j);
                
                intersection += p * t;
                pred_sum += p * p;
                target_sum += t * t;
            }
        }
          double dice_coeff = (2.0 * intersection + smooth_) / (pred_sum + target_sum + smooth_);
        return 1.0 - dice_coeff;
    }
    
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override {
        Matrix gradients(predictions.rows(), predictions.cols());
        
        double intersection = 0.0;
        double pred_sum = 0.0;
        double target_sum = 0.0;
        
        for (size_t i = 0; i < predictions.rows(); ++i) {
            for (size_t j = 0; j < predictions.cols(); ++j) {
                double p = predictions(i, j);
                double t = targets(i, j);
                
                intersection += p * t;
                pred_sum += p * p;
                target_sum += t * t;
            }
        }
        
        double denominator = pred_sum + target_sum + smooth_;
        
        for (size_t i = 0; i < predictions.rows(); ++i) {
            for (size_t j = 0; j < predictions.cols(); ++j) {
                double p = predictions(i, j);
                double t = targets(i, j);
                
                double numerator_grad = 2.0 * t;
                double denominator_grad = 2.0 * p;
                
                gradients(i, j) = -(numerator_grad * denominator - 
                                   (2.0 * intersection + smooth_) * denominator_grad) / 
                                  (denominator * denominator);
            }
        }
        
        return gradients;
    }
      std::unique_ptr<LossFunction> clone() const override {
        return std::make_unique<DiceLoss>(smooth_);
    }
    
    std::string name() const override { return "DiceLoss"; }
};

// Contrastive Loss (for metric learning)
class ContrastiveLoss : public LossFunction {
private:
    double margin_;
    
public:    ContrastiveLoss(double margin = 1.0) : margin_(margin) {}
    
    double compute_loss(const Matrix& predictions, const Matrix& targets) const override {
        // Predictions should be distances, targets should be 0 (similar) or 1 (dissimilar)
        double total_loss = 0.0;
        size_t num_samples = predictions.rows();
        
        for (size_t i = 0; i < num_samples; ++i) {
            double distance = predictions(i, 0);  // Assuming single output
            double label = targets(i, 0);
            
            if (label == 0.0) {
                // Similar pair
                total_loss += 0.5 * distance * distance;
            } else {                // Dissimilar pair
                total_loss += 0.5 * std::max(0.0, margin_ - distance) * std::max(0.0, margin_ - distance);
            }
        }
        
        return total_loss / num_samples;
    }
    
    Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override {
        Matrix gradients(predictions.rows(), predictions.cols());
        
        for (size_t i = 0; i < predictions.rows(); ++i) {
            double distance = predictions(i, 0);
            double label = targets(i, 0);
            
            if (label == 0.0) {
                // Similar pair
                gradients(i, 0) = distance;
            } else {
                // Dissimilar pair
                if (distance < margin_) {
                    gradients(i, 0) = -(margin_ - distance);
                } else {
                    gradients(i, 0) = 0.0;
                }
            }
        }
        
        return gradients;
    }
      std::unique_ptr<LossFunction> clone() const override {
        return std::make_unique<ContrastiveLoss>(margin_);
    }
    
    std::string name() const override { return "ContrastiveLoss"; }
};

} // namespace asekioml
