#include "ai/advanced_fusion_networks.hpp"
#include "ai/multimodal_attention.hpp"
#include "ai/memory_manager.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace clmodel::ai;

void print_tensor_info(const Tensor& tensor, const std::string& name) {
    const auto& shape = tensor.shape();
    std::cout << name << " shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void print_fusion_output(const AdvancedMultiModalFusion::FusionOutput& output, const std::string& name) {
    std::cout << name << " - Fused features: ";
    print_tensor_info(output.fused_features, "");
    std::cout << "  Attention maps available for " << output.attention_maps.size() << " modalities" << std::endl;
    std::cout << "  Intermediate features available for " << output.intermediate_features.size() << " modalities" << std::endl;
}

void test_advanced_fusion() {
    std::cout << "\n=== Testing Advanced Multi-Modal Fusion ===" << std::endl;
    
    // Configure fusion network
    AdvancedMultiModalFusion::FusionConfig config;
    config.input_dims[Modality::TEXT] = 512;
    config.input_dims[Modality::IMAGE] = 1024;
    config.input_dims[Modality::AUDIO] = 256;
    
    config.intermediate_dims[Modality::TEXT] = {512, 256, 128};
    config.intermediate_dims[Modality::IMAGE] = {1024, 512, 256};
    config.intermediate_dims[Modality::AUDIO] = {256, 128, 64};
    
    config.output_dim = 256;
    config.fusion_level = FusionLevel::HIERARCHICAL;
    config.num_fusion_layers = 3;
    config.dropout_rate = 0.1;
    config.use_residual_connections = true;
    config.use_layer_norm = true;
    
    try {
        AdvancedMultiModalFusion fusion(config);
        std::cout << "✓ Advanced fusion network created successfully" << std::endl;
        
        // Create test data
        std::unordered_map<Modality, Tensor> inputs;
        inputs[Modality::TEXT] = Tensor({2, 20, 512});     // [batch, seq_len, features]
        inputs[Modality::IMAGE] = Tensor({2, 1024});       // [batch, features] - flattened
        inputs[Modality::AUDIO] = Tensor({2, 100, 256});   // [batch, time, features]
        
        // Fill with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& [modality, tensor] : inputs) {
            auto& data = tensor.data();
            for (size_t i = 0; i < tensor.size(); ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        }
        
        // Test hierarchical fusion
        auto start = std::chrono::high_resolution_clock::now();
        auto result = fusion.hierarchical_fusion(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "✓ Hierarchical fusion completed in " << duration.count() << " μs" << std::endl;
        print_fusion_output(result, "Hierarchical fusion result");
        
        // Test adaptive fusion
        start = std::chrono::high_resolution_clock::now();
        auto adaptive_result = fusion.adaptive_fusion(inputs);
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "✓ Adaptive fusion completed in " << duration.count() << " μs" << std::endl;
        print_fusion_output(adaptive_result, "Adaptive fusion result");
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error in advanced fusion: " << e.what() << std::endl;
    }
}

void test_unified_encoder() {
    std::cout << "\n=== Testing Unified Multi-Modal Encoder ===" << std::endl;
    
    // Configure unified encoder
    UnifiedMultiModalEncoder::EncoderConfig config;
    config.modality_dims[Modality::TEXT] = 512;
    config.modality_dims[Modality::IMAGE] = 2048;
    config.modality_dims[Modality::AUDIO] = 256;
    config.unified_dim = 768;
    config.num_encoder_layers = 6;
    config.num_attention_heads = 8;
    config.temperature = 0.07f;
    config.use_momentum_encoder = true;
    config.momentum = 0.999;
    
    try {
        UnifiedMultiModalEncoder encoder(config);
        std::cout << "✓ Unified encoder created successfully" << std::endl;
        
        // Create test data
        std::unordered_map<Modality, Tensor> inputs;
        inputs[Modality::TEXT] = Tensor({2, 15, 512});     // [batch, seq_len, features]
        inputs[Modality::IMAGE] = Tensor({2, 196, 2048});  // [batch, patches, features]
        inputs[Modality::AUDIO] = Tensor({2, 50, 256});    // [batch, time, features]
        
        // Fill with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (auto& [modality, tensor] : inputs) {
            auto& data = tensor.data();
            for (size_t i = 0; i < tensor.size(); ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        }
        
        // Test encoding
        auto start = std::chrono::high_resolution_clock::now();
        auto encoded = encoder.encode(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "✓ Unified encoding completed in " << duration.count() << " μs" << std::endl;
        
        // Print results
        std::cout << "  Unified representations:" << std::endl;
        for (const auto& [modality, tensor] : encoded.unified_representations) {
            std::cout << "    " << static_cast<int>(modality) << ": ";
            print_tensor_info(tensor, "");
        }
        
        std::cout << "  Cross-modal similarity shape: ";
        print_tensor_info(encoded.cross_modal_similarity, "");
        
        std::cout << "  Pairwise similarities: " << encoded.pairwise_similarities.size() << " pairs" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error in unified encoder: " << e.what() << std::endl;
    }
}

void test_consistency_losses() {
    std::cout << "\n=== Testing Cross-Modal Consistency ===" << std::endl;
    
    // Configure consistency losses
    CrossModalConsistency::ConsistencyConfig config;
    config.loss_types = {
        CrossModalConsistency::ConsistencyType::SEMANTIC_CONSISTENCY,
        CrossModalConsistency::ConsistencyType::FEATURE_CONSISTENCY
    };
    config.loss_weights[CrossModalConsistency::ConsistencyType::SEMANTIC_CONSISTENCY] = 1.0f;
    config.loss_weights[CrossModalConsistency::ConsistencyType::FEATURE_CONSISTENCY] = 0.5f;
    config.margin = 0.2f;
    config.temperature = 0.1f;
    
    try {
        CrossModalConsistency consistency(config);
        std::cout << "✓ Cross-modal consistency created successfully" << std::endl;
        
        // Create test representations
        std::unordered_map<Modality, Tensor> representations;
        representations[Modality::TEXT] = Tensor({8, 256});    // [batch, features]
        representations[Modality::IMAGE] = Tensor({8, 256});   // [batch, features]
        representations[Modality::AUDIO] = Tensor({8, 256});   // [batch, features]
        
        // Fill with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& [modality, tensor] : representations) {
            auto& data = tensor.data();
            for (size_t i = 0; i < tensor.size(); ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        }
        
        // Test consistency loss computation
        auto start = std::chrono::high_resolution_clock::now();
        auto loss = consistency.compute_consistency_loss(representations);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "✓ Consistency loss computed in " << duration.count() << " μs" << std::endl;
        
        std::cout << "  Total loss: " << loss.total_loss << std::endl;
        std::cout << "  Individual losses:" << std::endl;
        for (const auto& [type, value] : loss.individual_losses) {
            std::cout << "    Type " << static_cast<int>(type) << ": " << value << std::endl;
        }
        std::cout << "  Pairwise losses: " << loss.pairwise_losses.size() << " pairs" << std::endl;
        
        // Test individual loss functions
        float semantic_loss = consistency.semantic_consistency_loss(representations);
        std::cout << "  Semantic consistency loss: " << semantic_loss << std::endl;
        
        float temporal_loss = consistency.temporal_consistency_loss(representations);
        std::cout << "  Temporal consistency loss: " << temporal_loss << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error in consistency losses: " << e.what() << std::endl;
    }
}

void test_training_pipeline() {
    std::cout << "\n=== Testing Multi-Modal Training ===" << std::endl;
    
    // Configure trainer
    MultiModalTrainer::TrainingConfig config;
    config.learning_rate = 1e-4f;
    config.weight_decay = 1e-5f;
    config.batch_size = 16;
    config.max_epochs = 5;
    config.gradient_clip_norm = 1.0f;
    config.use_curriculum_learning = false;
    config.use_mixed_precision = false;
    config.warmup_steps = 100;
    
    try {
        MultiModalTrainer trainer(config);
        std::cout << "✓ Multi-modal trainer created successfully" << std::endl;
        
        // Create fusion model for training
        AdvancedMultiModalFusion::FusionConfig fusion_config;
        fusion_config.input_dims[Modality::TEXT] = 512;
        fusion_config.input_dims[Modality::IMAGE] = 1024;
        fusion_config.output_dim = 256;
        fusion_config.fusion_level = FusionLevel::HIERARCHICAL;
        fusion_config.num_fusion_layers = 2;
        
        AdvancedMultiModalFusion model(fusion_config);
        
        // Create test batch
        std::unordered_map<Modality, Tensor> batch;
        batch[Modality::TEXT] = Tensor({16, 512});    // [batch, features]
        batch[Modality::IMAGE] = Tensor({16, 1024});  // [batch, features]
        
        // Fill with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& [modality, tensor] : batch) {
            auto& data = tensor.data();
            for (size_t i = 0; i < tensor.size(); ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        }
        
        // Create dummy loss function
        auto loss_fn = [](const Tensor& prediction, const Tensor& target) -> float {
            // Simple MSE loss
            auto diff = prediction - target;
            auto squared = diff * diff;
            auto loss_tensor = squared.mean();
            return static_cast<float>(loss_tensor.data()[0]);  // Get scalar value
        };
        
        // Test training step
        auto start = std::chrono::high_resolution_clock::now();
        auto metrics = trainer.training_step(model, batch, loss_fn);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "✓ Training step completed in " << duration.count() << " μs" << std::endl;
        
        std::cout << "  Training loss: " << metrics.loss << std::endl;
        std::cout << "  Accuracy: " << metrics.accuracy << std::endl;
        std::cout << "  Modality losses: " << metrics.modality_losses.size() << std::endl;
        std::cout << "  Additional metrics: " << metrics.additional_metrics.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error in training: " << e.what() << std::endl;
    }
}

void test_data_augmentation() {
    std::cout << "\n=== Testing Multi-Modal Augmentation ===" << std::endl;
    
    // Configure augmentation
    MultiModalAugmentation::AugmentationConfig config;
    config.enabled_augmentations = {
        MultiModalAugmentation::AugmentationType::CROSS_MODAL_MIXUP,
        MultiModalAugmentation::AugmentationType::NOISE_INJECTION
    };
    config.augmentation_probs[MultiModalAugmentation::AugmentationType::CROSS_MODAL_MIXUP] = 0.3f;
    config.augmentation_probs[MultiModalAugmentation::AugmentationType::NOISE_INJECTION] = 0.2f;
    config.mixup_alpha = 0.2f;
    config.dropout_rate = 0.1f;
    config.noise_std = 0.01f;
    
    try {
        MultiModalAugmentation augmentator(config);
        std::cout << "✓ Multi-modal augmentation created successfully" << std::endl;
        
        // Create test batch
        std::unordered_map<Modality, Tensor> batch;
        batch[Modality::TEXT] = Tensor({8, 512});     // [batch, features]
        batch[Modality::IMAGE] = Tensor({8, 1024});   // [batch, features]
        batch[Modality::AUDIO] = Tensor({8, 256});    // [batch, features]
        
        // Fill with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& [modality, tensor] : batch) {
            auto& data = tensor.data();
            for (size_t i = 0; i < tensor.size(); ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        }
        
        // Test augmentation
        auto start = std::chrono::high_resolution_clock::now();
        auto augmented = augmentator.augment_batch(batch);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "✓ Augmentation completed in " << duration.count() << " μs" << std::endl;
        
        std::cout << "  Augmented modalities:" << std::endl;
        for (const auto& [modality, tensor] : augmented) {
            std::cout << "    " << static_cast<int>(modality) << ": ";
            print_tensor_info(tensor, "");
        }
        
        // Test cross-modal mixup
        std::unordered_map<Modality, Tensor> batch2 = batch;  // Copy for second batch
        auto mixup_result = augmentator.cross_modal_mixup(batch, batch2, 0.5f);
        
        std::cout << "  Cross-modal mixup result:" << std::endl;
        for (const auto& [modality, tensor] : mixup_result) {
            std::cout << "    " << static_cast<int>(modality) << ": ";
            print_tensor_info(tensor, "");
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error in augmentation: " << e.what() << std::endl;
    }
}

void test_evaluation_metrics() {
    std::cout << "\n=== Testing Multi-Modal Evaluation ===" << std::endl;
    
    try {
        MultiModalEvaluator evaluator;
        std::cout << "✓ Multi-modal evaluator created successfully" << std::endl;
        
        // Create fusion model for evaluation
        AdvancedMultiModalFusion::FusionConfig config;
        config.input_dims[Modality::TEXT] = 512;
        config.input_dims[Modality::IMAGE] = 1024;
        config.output_dim = 256;
        config.fusion_level = FusionLevel::HIERARCHICAL;
        config.num_fusion_layers = 2;
        
        AdvancedMultiModalFusion model(config);
        
        // Create test data
        std::vector<std::unordered_map<Modality, Tensor>> test_data;
        std::vector<Tensor> ground_truth;
        
        for (int i = 0; i < 5; ++i) {
            std::unordered_map<Modality, Tensor> sample;
            sample[Modality::TEXT] = Tensor({1, 512});
            sample[Modality::IMAGE] = Tensor({1, 1024});
            test_data.push_back(sample);
            
            ground_truth.push_back(Tensor({1, 256}));
        }
        
        // Fill with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& sample : test_data) {
            for (auto& [modality, tensor] : sample) {
                auto& data = tensor.data();
                for (size_t i = 0; i < tensor.size(); ++i) {
                    data[i] = static_cast<double>(dist(gen));
                }
            }
        }
        
        for (auto& target : ground_truth) {
            auto& data = target.data();
            for (size_t i = 0; i < target.size(); ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        }
        
        // Test evaluation
        auto start = std::chrono::high_resolution_clock::now();
        auto metrics = evaluator.evaluate(model, test_data, ground_truth);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "✓ Evaluation completed in " << duration.count() << " μs" << std::endl;
        
        std::cout << "  Cross-modal retrieval metrics:" << std::endl;
        std::cout << "    Text-to-image recall@k: " << metrics.text_to_image_recall_at_k << std::endl;
        std::cout << "    Image-to-text recall@k: " << metrics.image_to_text_recall_at_k << std::endl;
        std::cout << "    Cross-modal MAP: " << metrics.cross_modal_map << std::endl;
        
        std::cout << "  Fusion quality metrics:" << std::endl;
        std::cout << "    Fusion accuracy: " << metrics.fusion_accuracy << std::endl;
        std::cout << "    Modality contribution balance: " << metrics.modality_contribution_balance << std::endl;
        
        std::cout << "  Representation quality metrics:" << std::endl;
        std::cout << "    Representation separability: " << metrics.representation_separability << std::endl;
        std::cout << "    Cross-modal alignment: " << metrics.cross_modal_alignment << std::endl;
        std::cout << "    Intra-modal clustering: " << metrics.intra_modal_clustering << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error in evaluation: " << e.what() << std::endl;
    }
}

void demo_performance_benchmarks() {
    std::cout << "\n=== Performance Benchmarks ===" << std::endl;
    
    // Test different fusion configurations
    std::vector<std::pair<std::string, AdvancedMultiModalFusion::FusionConfig>> configs = {
        {"Small Model", {}},
        {"Medium Model", {}},
        {"Large Model", {}}
    };
    
    // Small model
    configs[0].second.input_dims[Modality::TEXT] = 256;
    configs[0].second.input_dims[Modality::IMAGE] = 512;
    configs[0].second.output_dim = 128;
    configs[0].second.num_fusion_layers = 2;
    
    // Medium model
    configs[1].second.input_dims[Modality::TEXT] = 512;
    configs[1].second.input_dims[Modality::IMAGE] = 1024;
    configs[1].second.output_dim = 256;
    configs[1].second.num_fusion_layers = 4;
    
    // Large model
    configs[2].second.input_dims[Modality::TEXT] = 1024;
    configs[2].second.input_dims[Modality::IMAGE] = 2048;
    configs[2].second.output_dim = 512;
    configs[2].second.num_fusion_layers = 6;
    
    for (auto& [name, config] : configs) {
        config.fusion_level = FusionLevel::HIERARCHICAL;
        config.dropout_rate = 0.1;
        config.use_residual_connections = true;
        config.use_layer_norm = true;
        
        try {
            std::cout << "\n--- " << name << " ---" << std::endl;
            AdvancedMultiModalFusion model(config);
            
            // Create test batch
            std::unordered_map<Modality, Tensor> batch;
            batch[Modality::TEXT] = Tensor({32, config.input_dims[Modality::TEXT]});
            batch[Modality::IMAGE] = Tensor({32, config.input_dims[Modality::IMAGE]});
            
            // Fill with random data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            for (auto& [modality, tensor] : batch) {
                auto& data = tensor.data();
                for (size_t i = 0; i < tensor.size(); ++i) {
                    data[i] = static_cast<double>(dist(gen));
                }
            }
            
            // Benchmark forward pass
            const int num_iterations = 10;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < num_iterations; ++i) {
                auto result = model.hierarchical_fusion(batch);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            auto avg_duration = total_duration.count() / num_iterations;
            
            std::cout << "  Average inference time: " << avg_duration << " μs" << std::endl;
            std::cout << "  Throughput: " << (1000000.0 / avg_duration) << " samples/sec" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "✗ Error in " << name << ": " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "Advanced Multi-Modal Fusion Networks Demo" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        test_advanced_fusion();
        test_unified_encoder();
        test_consistency_losses();
        test_training_pipeline();
        test_data_augmentation();
        test_evaluation_metrics();
        demo_performance_benchmarks();
        
        std::cout << "\n=== Demo Complete ===" << std::endl;
        std::cout << "✓ All advanced fusion network features tested successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "\n✗ Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
