#pragma once

#include <vector>

namespace clmodel {

// Training history structure
struct TrainingHistory {
    std::vector<double> train_loss;
    std::vector<double> val_loss;
    std::vector<double> train_accuracy;
    std::vector<double> val_accuracy;
    std::vector<double> learning_rates;
    std::vector<double> epoch_times;
    
    // For compatibility with old TrainingHistory
    std::vector<double> training_loss;
    std::vector<double> validation_loss;
    std::vector<double> training_accuracy;
    std::vector<double> validation_accuracy;
    
    void add_epoch(double tr_loss, double v_loss, double tr_acc, double v_acc, double lr, double time) {
        train_loss.push_back(tr_loss);
        val_loss.push_back(v_loss);
        train_accuracy.push_back(tr_acc);
        val_accuracy.push_back(v_acc);
        learning_rates.push_back(lr);
        epoch_times.push_back(time);
        
        // Also populate old fields for compatibility
        training_loss.push_back(tr_loss);
        validation_loss.push_back(v_loss);
        training_accuracy.push_back(tr_acc);
        validation_accuracy.push_back(v_acc);
    }
    
    void clear() {
        train_loss.clear();
        val_loss.clear();
        train_accuracy.clear();
        val_accuracy.clear();
        learning_rates.clear();
        epoch_times.clear();
        training_loss.clear();
        validation_loss.clear();
        training_accuracy.clear();
        validation_accuracy.clear();
    }
    
    size_t size() const {
        return train_loss.size();
    }
};

} // namespace clmodel
