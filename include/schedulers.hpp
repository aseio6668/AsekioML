#pragma once

#include <cmath>
#include <vector>
#include <functional>

// Define M_PI for Windows
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asekioml {

// Base class for learning rate schedulers
class LearningRateScheduler {
public:
    virtual ~LearningRateScheduler() = default;
    virtual double get_lr(int epoch, double base_lr) const = 0;
    virtual std::unique_ptr<LearningRateScheduler> clone() const = 0;
};

// Step decay scheduler: lr *= gamma every step_size epochs
class StepLR : public LearningRateScheduler {
private:
    int step_size_;
    double gamma_;
    
public:
    StepLR(int step_size, double gamma = 0.1) 
        : step_size_(step_size), gamma_(gamma) {}
    
    double get_lr(int epoch, double base_lr) const override {
        int steps = epoch / step_size_;
        return base_lr * std::pow(gamma_, steps);
    }
    
    std::unique_ptr<LearningRateScheduler> clone() const override {
        return std::make_unique<StepLR>(step_size_, gamma_);
    }
};

// Exponential decay scheduler: lr *= gamma every epoch
class ExponentialLR : public LearningRateScheduler {
private:
    double gamma_;
    
public:
    ExponentialLR(double gamma) : gamma_(gamma) {}
    
    double get_lr(int epoch, double base_lr) const override {
        return base_lr * std::pow(gamma_, epoch);
    }
    
    std::unique_ptr<LearningRateScheduler> clone() const override {
        return std::make_unique<ExponentialLR>(gamma_);
    }
};

// Cosine annealing scheduler
class CosineAnnealingLR : public LearningRateScheduler {
private:
    int T_max_;  // Maximum number of iterations
    double eta_min_;  // Minimum learning rate
    
public:
    CosineAnnealingLR(int T_max, double eta_min = 0.0) 
        : T_max_(T_max), eta_min_(eta_min) {}
    
    double get_lr(int epoch, double base_lr) const override {
        if (epoch >= T_max_) {
            return eta_min_;
        }
        return eta_min_ + (base_lr - eta_min_) * 
               (1.0 + std::cos(M_PI * epoch / T_max_)) / 2.0;
    }
    
    std::unique_ptr<LearningRateScheduler> clone() const override {
        return std::make_unique<CosineAnnealingLR>(T_max_, eta_min_);
    }
};

// Polynomial decay scheduler
class PolynomialLR : public LearningRateScheduler {
private:
    int max_epochs_;
    double power_;
    double min_lr_;
    
public:
    PolynomialLR(int max_epochs, double power = 1.0, double min_lr = 0.0)
        : max_epochs_(max_epochs), power_(power), min_lr_(min_lr) {}
    
    double get_lr(int epoch, double base_lr) const override {
        if (epoch >= max_epochs_) {
            return min_lr_;
        }
        double factor = std::pow(1.0 - static_cast<double>(epoch) / max_epochs_, power_);
        return (base_lr - min_lr_) * factor + min_lr_;
    }
    
    std::unique_ptr<LearningRateScheduler> clone() const override {
        return std::make_unique<PolynomialLR>(max_epochs_, power_, min_lr_);
    }
};

// Multi-step scheduler: lr *= gamma at specific milestones
class MultiStepLR : public LearningRateScheduler {
private:
    std::vector<int> milestones_;
    double gamma_;
    
public:
    MultiStepLR(const std::vector<int>& milestones, double gamma = 0.1)
        : milestones_(milestones), gamma_(gamma) {}
    
    double get_lr(int epoch, double base_lr) const override {
        int decay_count = 0;
        for (int milestone : milestones_) {
            if (epoch >= milestone) {
                decay_count++;
            } else {
                break;
            }
        }
        return base_lr * std::pow(gamma_, decay_count);
    }
    
    std::unique_ptr<LearningRateScheduler> clone() const override {
        return std::make_unique<MultiStepLR>(milestones_, gamma_);
    }
};

// Warm restart scheduler (SGDR)
class CosineAnnealingWarmRestarts : public LearningRateScheduler {
private:
    int T_0_;  // Number of iterations for the first restart
    int T_mult_;  // Multiplication factor for increasing T_i after each restart
    double eta_min_;
    
public:
    CosineAnnealingWarmRestarts(int T_0, int T_mult = 1, double eta_min = 0.0)
        : T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min) {}
    
    double get_lr(int epoch, double base_lr) const override {
        int T_i = T_0_;
        int epoch_in_cycle = epoch;
        
        while (epoch_in_cycle >= T_i) {
            epoch_in_cycle -= T_i;
            T_i *= T_mult_;
        }
        
        return eta_min_ + (base_lr - eta_min_) * 
               (1.0 + std::cos(M_PI * epoch_in_cycle / T_i)) / 2.0;
    }
    
    std::unique_ptr<LearningRateScheduler> clone() const override {
        return std::make_unique<CosineAnnealingWarmRestarts>(T_0_, T_mult_, eta_min_);
    }
};

// Lambda scheduler: custom function
class LambdaLR : public LearningRateScheduler {
private:
    std::function<double(int)> lr_lambda_;
    
public:
    LambdaLR(std::function<double(int)> lr_lambda) : lr_lambda_(lr_lambda) {}
    
    double get_lr(int epoch, double base_lr) const override {
        return base_lr * lr_lambda_(epoch);
    }
    
    std::unique_ptr<LearningRateScheduler> clone() const override {
        return std::make_unique<LambdaLR>(lr_lambda_);
    }
};

} // namespace asekioml
