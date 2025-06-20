#include "ai/advanced_fusion_networks.hpp"
#include "ai/multimodal_attention.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>

using namespace clmodel::ai;

class AdvancedFusionTest {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<float> dist;
    
public:
    AdvancedFusionTest() : gen(rd()), dist(0.0f, 1.0f) {}
    
    void fill_random(Tensor& tensor) {
        auto& data = tensor.data();
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = static_cast<double>(dist(gen));
        }
    }
    
    bool test_advanced_fusion_creation() {
        std::cout << "Testing advanced fusion creation..." << std::endl;
        
        try {
            AdvancedMultiModalFusion::FusionConfig config;
            config.input_dims[Modality::TEXT] = 512;
            config.input_dims[Modality::IMAGE] = 1024;
            config.output_dim = 256;
            config.fusion_level = FusionLevel::HIERARCHICAL;
            config.num_fusion_layers = 2;
            
            AdvancedMultiModalFusion fusion(config);
            std::cout << "✓ Advanced fusion created successfully" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Failed to create advanced fusion: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_hierarchical_fusion() {
        std::cout << "Testing hierarchical fusion..." << std::endl;
        
        try {
            AdvancedMultiModalFusion::FusionConfig config;
            config.input_dims[Modality::TEXT] = 512;
            config.input_dims[Modality::IMAGE] = 1024;
            config.intermediate_dims[Modality::TEXT] = {512, 256};
            config.intermediate_dims[Modality::IMAGE] = {1024, 512};
            config.output_dim = 256;
            config.fusion_level = FusionLevel::HIERARCHICAL;
            config.num_fusion_layers = 2;
            
            AdvancedMultiModalFusion fusion(config);
            
            // Create test inputs
            std::unordered_map<Modality, Tensor> inputs;
            inputs[Modality::TEXT] = Tensor({2, 512});
            inputs[Modality::IMAGE] = Tensor({2, 1024});
            
            fill_random(inputs[Modality::TEXT]);
            fill_random(inputs[Modality::IMAGE]);
            
            // Test hierarchical fusion
            auto result = fusion.hierarchical_fusion(inputs);
            
            // Validate output structure
            assert(!result.fused_features.is_empty());
            assert(result.fused_features.shape()[0] == 2);  // batch size
            assert(result.fused_features.shape()[1] == 256); // output dim
            
            std::cout << "✓ Hierarchical fusion test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Hierarchical fusion test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_adaptive_fusion() {
        std::cout << "Testing adaptive fusion..." << std::endl;
        
        try {
            AdvancedMultiModalFusion::FusionConfig config;
            config.input_dims[Modality::TEXT] = 256;
            config.input_dims[Modality::IMAGE] = 512;
            config.input_dims[Modality::AUDIO] = 128;
            config.output_dim = 128;
            config.fusion_level = FusionLevel::HIERARCHICAL;
            config.num_fusion_layers = 2;
            
            AdvancedMultiModalFusion fusion(config);
            
            // Create test inputs
            std::unordered_map<Modality, Tensor> inputs;
            inputs[Modality::TEXT] = Tensor({4, 256});
            inputs[Modality::IMAGE] = Tensor({4, 512});
            inputs[Modality::AUDIO] = Tensor({4, 128});
            
            fill_random(inputs[Modality::TEXT]);
            fill_random(inputs[Modality::IMAGE]);
            fill_random(inputs[Modality::AUDIO]);
            
            // Test adaptive fusion
            auto result = fusion.adaptive_fusion(inputs);
            
            // Validate output
            assert(!result.fused_features.is_empty());
            assert(result.fused_features.shape()[0] == 4);   // batch size
            assert(result.fused_features.shape()[1] == 128); // output dim
            
            std::cout << "✓ Adaptive fusion test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Adaptive fusion test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_unified_encoder() {
        std::cout << "Testing unified encoder..." << std::endl;
        
        try {
            UnifiedMultiModalEncoder::EncoderConfig config;
            config.modality_dims[Modality::TEXT] = 512;
            config.modality_dims[Modality::IMAGE] = 1024;
            config.unified_dim = 256;
            config.num_encoder_layers = 4;
            config.num_attention_heads = 8;
            
            UnifiedMultiModalEncoder encoder(config);
            
            // Create test inputs
            std::unordered_map<Modality, Tensor> inputs;
            inputs[Modality::TEXT] = Tensor({2, 10, 512});   // [batch, seq, features]
            inputs[Modality::IMAGE] = Tensor({2, 16, 1024}); // [batch, patches, features]
            
            fill_random(inputs[Modality::TEXT]);
            fill_random(inputs[Modality::IMAGE]);
            
            // Test encoding
            auto result = encoder.encode(inputs);
            
            // Validate output
            assert(result.unified_representations.size() == 2);
            assert(!result.cross_modal_similarity.is_empty());
            
            for (const auto& [modality, repr] : result.unified_representations) {
                assert(!repr.is_empty());
                assert(repr.shape()[0] == 2);    // batch size
                // Unified representations should have unified_dim as last dimension
            }
            
            std::cout << "✓ Unified encoder test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Unified encoder test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_consistency_losses() {
        std::cout << "Testing consistency losses..." << std::endl;
        
        try {
            CrossModalConsistency::ConsistencyConfig config;
            config.loss_types = {
                CrossModalConsistency::ConsistencyType::SEMANTIC_CONSISTENCY,
                CrossModalConsistency::ConsistencyType::FEATURE_CONSISTENCY
            };
            config.loss_weights[CrossModalConsistency::ConsistencyType::SEMANTIC_CONSISTENCY] = 1.0f;
            config.loss_weights[CrossModalConsistency::ConsistencyType::FEATURE_CONSISTENCY] = 0.5f;
            
            CrossModalConsistency consistency(config);
            
            // Create test representations
            std::unordered_map<Modality, Tensor> representations;
            representations[Modality::TEXT] = Tensor({4, 256});
            representations[Modality::IMAGE] = Tensor({4, 256});
            
            fill_random(representations[Modality::TEXT]);
            fill_random(representations[Modality::IMAGE]);
            
            // Test consistency loss computation
            auto loss = consistency.compute_consistency_loss(representations);
            
            // Validate loss structure
            assert(loss.total_loss >= 0.0f);
            assert(loss.individual_losses.size() >= 1);
            
            // Test individual loss functions
            float semantic_loss = consistency.semantic_consistency_loss(representations);
            assert(semantic_loss >= 0.0f);
            
            float temporal_loss = consistency.temporal_consistency_loss(representations);
            assert(temporal_loss >= 0.0f);
            
            float spatial_loss = consistency.spatial_consistency_loss(representations);
            assert(spatial_loss >= 0.0f);
            
            std::cout << "✓ Consistency losses test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Consistency losses test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_training_pipeline() {
        std::cout << "Testing training pipeline..." << std::endl;
        
        try {
            MultiModalTrainer::TrainingConfig config;
            config.learning_rate = 1e-3f;
            config.batch_size = 8;
            config.max_epochs = 2;
            
            MultiModalTrainer trainer(config);
            
            // Create model for training
            AdvancedMultiModalFusion::FusionConfig fusion_config;
            fusion_config.input_dims[Modality::TEXT] = 256;
            fusion_config.input_dims[Modality::IMAGE] = 512;
            fusion_config.output_dim = 128;
            fusion_config.fusion_level = FusionLevel::HIERARCHICAL;
            fusion_config.num_fusion_layers = 2;
            
            AdvancedMultiModalFusion model(fusion_config);
            
            // Create test batch
            std::unordered_map<Modality, Tensor> batch;
            batch[Modality::TEXT] = Tensor({8, 256});
            batch[Modality::IMAGE] = Tensor({8, 512});
            
            fill_random(batch[Modality::TEXT]);
            fill_random(batch[Modality::IMAGE]);
            
            // Create dummy loss function
            auto loss_fn = [](const Tensor& prediction, const Tensor& target) -> float {
                return 0.5f; // Dummy loss
            };
            
            // Test training step
            auto metrics = trainer.training_step(model, batch, loss_fn);
            
            // Validate metrics
            assert(metrics.loss >= 0.0f);
            assert(metrics.accuracy >= 0.0f && metrics.accuracy <= 1.0f);
            
            std::cout << "✓ Training pipeline test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Training pipeline test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_data_augmentation() {
        std::cout << "Testing data augmentation..." << std::endl;
        
        try {
            MultiModalAugmentation::AugmentationConfig config;
            config.enabled_augmentations = {
                MultiModalAugmentation::AugmentationType::NOISE_INJECTION
            };
            config.augmentation_probs[MultiModalAugmentation::AugmentationType::NOISE_INJECTION] = 0.5f;
            config.noise_std = 0.01f;
            
            MultiModalAugmentation augmentator(config);
            
            // Create test batch
            std::unordered_map<Modality, Tensor> batch;
            batch[Modality::TEXT] = Tensor({4, 256});
            batch[Modality::IMAGE] = Tensor({4, 512});
            
            fill_random(batch[Modality::TEXT]);
            fill_random(batch[Modality::IMAGE]);
            
            // Test augmentation
            auto augmented = augmentator.augment_batch(batch);
            
            // Validate augmentation results
            assert(augmented.size() == batch.size());
            for (const auto& [modality, tensor] : augmented) {
                assert(!tensor.is_empty());
                assert(tensor.shape() == batch[modality].shape());
            }
            
            // Test cross-modal mixup
            std::unordered_map<Modality, Tensor> batch2 = batch;
            auto mixup_result = augmentator.cross_modal_mixup(batch, batch2, 0.5f);
            
            assert(mixup_result.size() == batch.size());
            
            std::cout << "✓ Data augmentation test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Data augmentation test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_evaluation_metrics() {
        std::cout << "Testing evaluation metrics..." << std::endl;
        
        try {
            MultiModalEvaluator evaluator;
            
            // Create model for evaluation
            AdvancedMultiModalFusion::FusionConfig config;
            config.input_dims[Modality::TEXT] = 256;
            config.input_dims[Modality::IMAGE] = 512;
            config.output_dim = 128;
            config.fusion_level = FusionLevel::HIERARCHICAL;
            config.num_fusion_layers = 2;
            
            AdvancedMultiModalFusion model(config);
            
            // Create test data
            std::vector<std::unordered_map<Modality, Tensor>> test_data;
            std::vector<Tensor> ground_truth;
            
            for (int i = 0; i < 3; ++i) {
                std::unordered_map<Modality, Tensor> sample;
                sample[Modality::TEXT] = Tensor({1, 256});
                sample[Modality::IMAGE] = Tensor({1, 512});
                
                fill_random(sample[Modality::TEXT]);
                fill_random(sample[Modality::IMAGE]);
                
                test_data.push_back(sample);
                
                Tensor target({1, 128});
                fill_random(target);
                ground_truth.push_back(target);
            }
            
            // Test evaluation
            auto metrics = evaluator.evaluate(model, test_data, ground_truth);
            
            // Validate metrics
            assert(metrics.fusion_accuracy >= 0.0f && metrics.fusion_accuracy <= 1.0f);
            assert(metrics.modality_contribution_balance >= 0.0f);
            assert(metrics.representation_separability >= 0.0f);
            
            std::cout << "✓ Evaluation metrics test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Evaluation metrics test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_fusion_configurations() {
        std::cout << "Testing different fusion configurations..." << std::endl;
        
        // Test different fusion levels
        std::vector<FusionLevel> fusion_levels = {
            FusionLevel::INPUT_LEVEL,
            FusionLevel::FEATURE_LEVEL,
            FusionLevel::DECISION_LEVEL,
            FusionLevel::HIERARCHICAL
        };
        
        for (auto level : fusion_levels) {
            try {
                AdvancedMultiModalFusion::FusionConfig config;
                config.input_dims[Modality::TEXT] = 256;
                config.input_dims[Modality::IMAGE] = 512;
                config.output_dim = 128;
                config.fusion_level = level;
                config.num_fusion_layers = 2;
                
                AdvancedMultiModalFusion fusion(config);
                
                // Test with simple inputs
                std::unordered_map<Modality, Tensor> inputs;
                inputs[Modality::TEXT] = Tensor({2, 256});
                inputs[Modality::IMAGE] = Tensor({2, 512});
                
                fill_random(inputs[Modality::TEXT]);
                fill_random(inputs[Modality::IMAGE]);
                
                auto result = fusion.forward(inputs);
                assert(!result.fused_features.is_empty());
                
            } catch (const std::exception& e) {
                std::cout << "✗ Failed for fusion level " << static_cast<int>(level) 
                         << ": " << e.what() << std::endl;
                return false;
            }
        }
        
        std::cout << "✓ Fusion configurations test passed" << std::endl;
        return true;
    }
    
    bool test_contrastive_learning() {
        std::cout << "Testing contrastive learning..." << std::endl;
        
        try {
            UnifiedMultiModalEncoder::EncoderConfig config;
            config.modality_dims[Modality::TEXT] = 256;
            config.modality_dims[Modality::IMAGE] = 512;
            config.unified_dim = 128;
            config.num_encoder_layers = 2;
            config.num_attention_heads = 4;
            config.temperature = 0.1f;
            
            UnifiedMultiModalEncoder encoder(config);
            
            // Create test data
            std::unordered_map<Modality, Tensor> inputs;
            inputs[Modality::TEXT] = Tensor({4, 256});
            inputs[Modality::IMAGE] = Tensor({4, 512});
            
            fill_random(inputs[Modality::TEXT]);
            fill_random(inputs[Modality::IMAGE]);
            
            // Test encoding
            auto encoding = encoder.encode(inputs);
            
            // Test contrastive loss
            std::vector<bool> positive_pairs = {true, false, true, false};
            float loss = encoder.compute_contrastive_loss(encoding, positive_pairs);
            
            assert(loss >= 0.0f);
            
            // Test momentum encoder update
            encoder.update_momentum_encoder();
            
            std::cout << "✓ Contrastive learning test passed" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Contrastive learning test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    void run_all_tests() {
        std::cout << "Advanced Fusion Networks Test Suite" << std::endl;
        std::cout << "====================================" << std::endl;
        
        int passed = 0;
        int total = 0;
        
        std::vector<std::pair<std::string, std::function<bool()>>> tests = {
            {"Advanced Fusion Creation", [this]() { return test_advanced_fusion_creation(); }},
            {"Hierarchical Fusion", [this]() { return test_hierarchical_fusion(); }},
            {"Adaptive Fusion", [this]() { return test_adaptive_fusion(); }},
            {"Unified Encoder", [this]() { return test_unified_encoder(); }},
            {"Consistency Losses", [this]() { return test_consistency_losses(); }},
            {"Training Pipeline", [this]() { return test_training_pipeline(); }},
            {"Data Augmentation", [this]() { return test_data_augmentation(); }},
            {"Evaluation Metrics", [this]() { return test_evaluation_metrics(); }},
            {"Fusion Configurations", [this]() { return test_fusion_configurations(); }},
            {"Contrastive Learning", [this]() { return test_contrastive_learning(); }}
        };
        
        for (auto& [name, test] : tests) {
            std::cout << "\n--- " << name << " ---" << std::endl;
            total++;
            if (test()) {
                passed++;
            }
        }
        
        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Passed: " << passed << "/" << total << std::endl;
        
        if (passed == total) {
            std::cout << "✓ All tests passed!" << std::endl;
        } else {
            std::cout << "✗ " << (total - passed) << " tests failed" << std::endl;
        }
    }
};

int main() {
    try {
        AdvancedFusionTest test_suite;
        test_suite.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
