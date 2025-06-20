#pragma once

#include "matrix.hpp"
#include "training_history.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <iostream>
#include <fstream>
#include <chrono>
#include <limits>

namespace asekioml {

// Forward declaration
class NeuralNetwork;

// Base callback class
class TrainingCallback {
public:
    virtual ~TrainingCallback() = default;
    
    // Called at the beginning of training
    virtual void on_training_begin(const TrainingHistory& /*history*/) {}
    
    // Called at the end of training
    virtual void on_training_end(const TrainingHistory& /*history*/) {}
    
    // Called at the beginning of each epoch
    virtual void on_epoch_begin(int /*epoch*/, const TrainingHistory& /*history*/) {}
    
    // Called at the end of each epoch
    virtual void on_epoch_end(int epoch, const TrainingHistory& history) = 0;
    
    // Called at the beginning of each batch
    virtual void on_batch_begin(int /*batch*/, const TrainingHistory& /*history*/) {}
    
    // Called at the end of each batch
    virtual void on_batch_end(int /*batch*/, double /*loss*/, const TrainingHistory& /*history*/) {}
    
    // Returns true if training should stop
    virtual bool should_stop() const { return false; }
    
    virtual std::unique_ptr<TrainingCallback> clone() const = 0;
};

// Early stopping callback
class EarlyStopping : public TrainingCallback {
private:
    std::string monitor_;
    double min_delta_;
    int patience_;
    bool restore_best_weights_;
    std::string mode_;
    
    int wait_;
    double best_score_;
    bool stopped_;
    std::vector<Matrix> best_weights_;
    
    double get_monitor_value(const TrainingHistory& history) const {
        if (history.size() == 0) return 0.0;
        
        if (monitor_ == "val_loss") {
            return history.val_loss.back();
        } else if (monitor_ == "val_accuracy") {
            return history.val_accuracy.back();
        } else if (monitor_ == "train_loss") {
            return history.train_loss.back();
        } else if (monitor_ == "train_accuracy") {
            return history.train_accuracy.back();
        }
        return 0.0;
    }
    
    bool is_improvement(double current, double best) const {
        if (mode_ == "min") {
            return current < best - min_delta_;
        } else {
            return current > best + min_delta_;
        }
    }
    
public:
    EarlyStopping(const std::string& monitor = "val_loss", double min_delta = 0.0, 
                  int patience = 0, bool restore_best_weights = false, 
                  const std::string& mode = "auto")
        : monitor_(monitor), min_delta_(min_delta), patience_(patience),
          restore_best_weights_(restore_best_weights), mode_(mode),
          wait_(0), stopped_(false) {
        
        if (mode_ == "auto") {
            if (monitor_.find("acc") != std::string::npos) {
                mode_ = "max";
                best_score_ = -std::numeric_limits<double>::infinity();
            } else {
                mode_ = "min";
                best_score_ = std::numeric_limits<double>::infinity();
            }
        } else if (mode_ == "min") {
            best_score_ = std::numeric_limits<double>::infinity();
        } else {
            best_score_ = -std::numeric_limits<double>::infinity();
        }
    }
    
    void on_training_begin(const TrainingHistory& /*history*/) override {
        wait_ = 0;
        stopped_ = false;
        if (mode_ == "min") {
            best_score_ = std::numeric_limits<double>::infinity();
        } else {
            best_score_ = -std::numeric_limits<double>::infinity();
        }
    }
    
    void on_epoch_end(int epoch, const TrainingHistory& history) override {
        double current = get_monitor_value(history);
        
        if (is_improvement(current, best_score_)) {
            best_score_ = current;
            wait_ = 0;
            // TODO: Save best weights if restore_best_weights_ is true
        } else {
            wait_++;
            if (wait_ >= patience_) {
                stopped_ = true;
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                // TODO: Restore best weights if needed
            }
        }
    }
    
    bool should_stop() const override {
        return stopped_;
    }
    
    std::unique_ptr<TrainingCallback> clone() const override {
        return std::make_unique<EarlyStopping>(monitor_, min_delta_, patience_, 
                                             restore_best_weights_, mode_);
    }
};

// Model checkpoint callback
class ModelCheckpoint : public TrainingCallback {
private:
    std::string filepath_;
    std::string monitor_;
    bool save_best_only_;
    std::string mode_;
    
    double best_score_;
    
    double get_monitor_value(const TrainingHistory& history) const {
        if (history.size() == 0) return 0.0;
        
        if (monitor_ == "val_loss") {
            return history.val_loss.back();
        } else if (monitor_ == "val_accuracy") {
            return history.val_accuracy.back();
        } else if (monitor_ == "train_loss") {
            return history.train_loss.back();
        } else if (monitor_ == "train_accuracy") {
            return history.train_accuracy.back();
        }
        return 0.0;
    }
    
    bool is_improvement(double current, double best) const {
        if (mode_ == "min") {
            return current < best;
        } else {
            return current > best;
        }
    }
    
public:
    ModelCheckpoint(const std::string& filepath, const std::string& monitor = "val_loss",
                   bool save_best_only = false, const std::string& mode = "auto")
        : filepath_(filepath), monitor_(monitor), save_best_only_(save_best_only), mode_(mode) {
        
        if (mode_ == "auto") {
            if (monitor_.find("acc") != std::string::npos) {
                mode_ = "max";
                best_score_ = -std::numeric_limits<double>::infinity();
            } else {
                mode_ = "min";
                best_score_ = std::numeric_limits<double>::infinity();
            }
        } else if (mode_ == "min") {
            best_score_ = std::numeric_limits<double>::infinity();
        } else {
            best_score_ = -std::numeric_limits<double>::infinity();
        }
    }
    
    void on_epoch_end(int epoch, const TrainingHistory& history) override {
        double current = get_monitor_value(history);
        
        bool should_save = !save_best_only_;
        if (save_best_only_ && is_improvement(current, best_score_)) {
            best_score_ = current;
            should_save = true;
        }
        
        if (should_save) {
            std::string filename = filepath_;
            // Replace {epoch} and {monitor} placeholders
            size_t pos = filename.find("{epoch}");
            if (pos != std::string::npos) {
                filename.replace(pos, 7, std::to_string(epoch + 1));
            }
            pos = filename.find("{" + monitor_ + "}");
            if (pos != std::string::npos) {
                filename.replace(pos, monitor_.length() + 2, std::to_string(current));
            }
            
            std::cout << "Saving model to " << filename << std::endl;
            // TODO: Implement actual model saving
        }
    }
    
    std::unique_ptr<TrainingCallback> clone() const override {
        return std::make_unique<ModelCheckpoint>(filepath_, monitor_, save_best_only_, mode_);
    }
};

// Learning rate scheduler callback
class LearningRateSchedulerCallback : public TrainingCallback {
private:
    std::unique_ptr<class LearningRateScheduler> scheduler_;
    double base_lr_;
    
public:
    LearningRateSchedulerCallback(std::unique_ptr<class LearningRateScheduler> scheduler, double base_lr)
        : scheduler_(std::move(scheduler)), base_lr_(base_lr) {}
    
    void on_epoch_end(int epoch, const TrainingHistory& /*history*/) override {
        double new_lr = scheduler_->get_lr(epoch, base_lr_);
        // TODO: Update optimizer learning rate
        std::cout << "Learning rate updated to: " << new_lr << std::endl;
    }
    
    std::unique_ptr<TrainingCallback> clone() const override {
        return std::make_unique<LearningRateSchedulerCallback>(scheduler_->clone(), base_lr_);
    }
};

// Progress bar callback
class ProgressBar : public TrainingCallback {
private:
    int total_epochs_;
    int bar_length_;
    
    void print_progress_bar(int current, int total, double loss, double acc) const {
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(bar_length_ * progress);
        
        std::cout << "\rEpoch " << current << "/" << total << " [";
        for (int i = 0; i < bar_length_; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] - loss: " << std::fixed << std::setprecision(4) << loss;
        if (acc >= 0) {
            std::cout << " - acc: " << std::fixed << std::setprecision(4) << acc;
        }
        std::cout << std::flush;
    }
    
public:
    ProgressBar(int total_epochs, int bar_length = 30) 
        : total_epochs_(total_epochs), bar_length_(bar_length) {}
    
    void on_epoch_end(int epoch, const TrainingHistory& history) override {
        if (history.size() > 0) {
            double loss = history.train_loss.back();
            double acc = history.train_accuracy.size() > 0 ? history.train_accuracy.back() : -1;
            print_progress_bar(epoch + 1, total_epochs_, loss, acc);
            
            if (epoch + 1 == total_epochs_) {
                std::cout << std::endl;
            }
        }
    }
    
    std::unique_ptr<TrainingCallback> clone() const override {
        return std::make_unique<ProgressBar>(total_epochs_, bar_length_);
    }
};

// CSV logger callback
class CSVLogger : public TrainingCallback {
private:
    std::string filename_;
    std::ofstream file_;
    bool append_;
    
public:
    CSVLogger(const std::string& filename, bool append = false)
        : filename_(filename), append_(append) {}
    
    void on_training_begin(const TrainingHistory& /*history*/) override {
        file_.open(filename_, append_ ? std::ios::app : std::ios::out);
        if (!append_) {
            file_ << "epoch,train_loss,val_loss,train_accuracy,val_accuracy,learning_rate,epoch_time\n";
        }
    }
    
    void on_epoch_end(int epoch, const TrainingHistory& history) override {
        if (history.size() > 0) {
            file_ << epoch + 1 << ","
                  << history.train_loss.back() << ","
                  << history.val_loss.back() << ","
                  << history.train_accuracy.back() << ","
                  << history.val_accuracy.back() << ","
                  << history.learning_rates.back() << ","
                  << history.epoch_times.back() << "\n";
            file_.flush();
        }
    }
    
    void on_training_end(const TrainingHistory& /*history*/) override {
        file_.close();
    }
    
    std::unique_ptr<TrainingCallback> clone() const override {
        return std::make_unique<CSVLogger>(filename_, append_);
    }
};

// Lambda callback for custom functions
class LambdaCallback : public TrainingCallback {
private:
    std::function<void(int, const TrainingHistory&)> on_epoch_end_func_;
    
public:    LambdaCallback(const std::function<void(int, const TrainingHistory&)>& func)
        : on_epoch_end_func_(func) {}
    
    void on_epoch_end(int epoch, const TrainingHistory& history) override {
        if (on_epoch_end_func_) {
            on_epoch_end_func_(epoch, history);
        }
    }
    
    std::unique_ptr<TrainingCallback> clone() const override {
        return std::make_unique<LambdaCallback>(on_epoch_end_func_);
    }
};

} // namespace asekioml
