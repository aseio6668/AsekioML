/**
 * @file video_audio_text_fusion_demo.cpp
 * @brief Comprehensive demo for Week 15: Video-Audio-Text Fusion Pipeline
 * 
 * This demo showcases the complete multi-modal fusion system with:
 * - Multi-modal content processing and synchronization
 * - Various fusion strategies (early, late, attention, hierarchical, temporal)
 * - Content generation and quality assessment
 * - Real-time streaming capabilities
 * - Performance benchmarking and analysis
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <numeric>

#include "ai/video_audio_text_fusion.hpp"
#include "tensor.hpp"

using namespace clmodel::ai;

// Helper function to create synthetic video data
Tensor createSyntheticVideo(const std::vector<size_t>& shape) {
    Tensor video(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.5, 0.2);
    
    for (size_t i = 0; i < video.size(); ++i) {
        video.data()[i] = std::max(0.0, std::min(1.0, dis(gen)));
    }
    
    return video;
}

// Helper function to create synthetic audio data
Tensor createSyntheticAudio(const std::vector<size_t>& shape) {
    Tensor audio(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 0.3);
    
    for (size_t i = 0; i < audio.size(); ++i) {
        audio.data()[i] = dis(gen);
    }
    
    return audio;
}

// Helper function to create synthetic text features
Tensor createSyntheticText(const std::vector<size_t>& shape) {
    Tensor text(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (size_t i = 0; i < text.size(); ++i) {
        text.data()[i] = dis(gen);
    }
    
    return text;
}

// Helper function to generate timestamps
std::vector<double> generateTimestamps(size_t count, double duration) {
    std::vector<double> timestamps;
    timestamps.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        timestamps.push_back((static_cast<double>(i) / count) * duration);
    }
    
    return timestamps;
}

// Helper function to print quality metrics
void printQualityMetrics(const ContentQualityMetrics& metrics, const std::string& label) {
    std::cout << "\n=== " << label << " Quality Metrics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Overall Quality:        " << metrics.overall_quality << std::endl;
    std::cout << "Visual Quality:         " << metrics.visual_quality << std::endl;
    std::cout << "Audio Quality:          " << metrics.audio_quality << std::endl;
    std::cout << "Text Coherence:         " << metrics.text_coherence << std::endl;
    std::cout << "Temporal Consistency:   " << metrics.temporal_consistency << std::endl;
    std::cout << "Semantic Alignment:     " << metrics.semantic_alignment << std::endl;
    std::cout << "Computational Efficiency: " << metrics.computational_efficiency << std::endl;
    std::cout << "Generation Mode:        " << metrics.generation_mode << std::endl;
}

// Demo 1: Basic Multi-Modal Content Creation and Processing
void demoBasicContentProcessing() {
    std::cout << "\n=== DEMO 1: Basic Multi-Modal Content Processing ===" << std::endl;
    
    try {
        // Create multi-modal content
        MultiModalContent content;
        content.video_features = createSyntheticVideo({16, 3, 224, 224}); // 16 frames, 3 channels, 224x224
        content.audio_features = createSyntheticAudio({16, 128});          // 16 timesteps, 128 features
        content.text_features = createSyntheticText({16, 512});           // 16 tokens, 512 embedding dim
        content.timestamps = generateTimestamps(16, 2.0);                 // 2 seconds duration
        
        content.metadata["resolution"] = "224x224";
        content.metadata["sample_rate"] = "44100";
        content.metadata["language"] = "en";
        
        std::cout << "Created multi-modal content:" << std::endl;
        std::cout << "  Video shape: [" << content.video_features.shape()[0] << ", " 
                  << content.video_features.shape()[1] << ", " 
                  << content.video_features.shape()[2] << ", "
                  << content.video_features.shape()[3] << "]" << std::endl;
        std::cout << "  Audio shape: [" << content.audio_features.shape()[0] << ", " 
                  << content.audio_features.shape()[1] << "]" << std::endl;
        std::cout << "  Text shape: [" << content.text_features.shape()[0] << ", " 
                  << content.text_features.shape()[1] << "]" << std::endl;
        std::cout << "  Duration: " << content.timestamps.back() << " seconds" << std::endl;
        
        // Process with VideoAudioTextProcessor
        VideoAudioTextProcessor processor;
        MultiModalContent processed = processor.processContent(content);
        MultiModalContent synchronized = processor.synchronizeModalities(processed);
        
        std::cout << "\nProcessed and synchronized content" << std::endl;
        
        // Assess quality
        ContentQualityMetrics quality = processor.assessContentQuality(synchronized);
        printQualityMetrics(quality, "Processed Content");
        
        // Analyze modality contributions
        auto contributions = processor.analyzeModalityContributions(synchronized);
        std::cout << "\nModality Contributions:" << std::endl;
        for (const auto& [modality, contribution] : contributions) {
            std::cout << "  " << modality << ": " << std::fixed << std::setprecision(3) 
                      << contribution << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in basic content processing: " << e.what() << std::endl;
    }
}

// Demo 2: Fusion Strategy Comparison
void demoFusionStrategies() {
    std::cout << "\n=== DEMO 2: Fusion Strategy Comparison ===" << std::endl;
    
    try {
        // Create test content
        MultiModalContent content;
        content.video_features = createSyntheticVideo({8, 3, 112, 112});
        content.audio_features = createSyntheticAudio({8, 64});
        content.text_features = createSyntheticText({8, 256});
        content.timestamps = generateTimestamps(8, 1.0);
          // Test different fusion strategies
        std::vector<VideoAudioTextFusionStrategy> strategies = {
            VideoAudioTextFusionStrategy::EARLY_FUSION,
            VideoAudioTextFusionStrategy::LATE_FUSION,
            VideoAudioTextFusionStrategy::ATTENTION_FUSION,
            VideoAudioTextFusionStrategy::HIERARCHICAL_FUSION,
            VideoAudioTextFusionStrategy::TEMPORAL_FUSION
        };
        
        std::vector<std::string> strategy_names = {
            "Early Fusion",
            "Late Fusion", 
            "Attention Fusion",
            "Hierarchical Fusion",
            "Temporal Fusion"
        };
        
        for (size_t i = 0; i < strategies.size(); ++i) {
            std::cout << "\n--- Testing " << strategy_names[i] << " ---" << std::endl;
            
            FusionConfig config;
            config.strategy = strategies[i];
            config.fusion_weight_video = 0.4;
            config.fusion_weight_audio = 0.35;
            config.fusion_weight_text = 0.25;
            config.enable_quality_feedback = true;
            
            MultiModalFusionPipeline pipeline(config);
            
            auto start = std::chrono::high_resolution_clock::now();
            MultiModalContent fused = pipeline.fusePipeline(content);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
            printQualityMetrics(fused.quality, strategy_names[i]);
            
            // Get pipeline statistics
            auto stats = pipeline.getPipelineStatistics();
            std::cout << "Pipeline Statistics:" << std::endl;
            for (const auto& [key, value] : stats) {
                std::cout << "  " << key << ": " << std::fixed << std::setprecision(4) 
                          << value << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in fusion strategy comparison: " << e.what() << std::endl;
    }
}

// Demo 3: Content Generation
void demoContentGeneration() {
    std::cout << "\n=== DEMO 3: Content Generation ===" << std::endl;
    
    try {
        ContentGenerationEngine generator;
        
        // Set generation parameters
        generator.setGenerationStyle("cinematic");
        generator.setQualityTarget(0.8);
        
        std::cout << "Available generation styles:" << std::endl;
        auto styles = generator.getAvailableStyles();
        for (const auto& style : styles) {
            std::cout << "  - " << style << std::endl;
        }
        
        // Generate content from text prompt
        std::cout << "\n--- Generating content from prompt ---" << std::endl;        FusionConfig gen_config;
        gen_config.strategy = VideoAudioTextFusionStrategy::ATTENTION_FUSION;
        gen_config.enable_real_time = false;
        gen_config.max_sequence_length = 12;
        
        MultiModalContent generated = generator.generateContent(
            "A serene sunset over mountains with gentle acoustic music", gen_config);
        
        std::cout << "Generated content:" << std::endl;
        std::cout << "  Video shape: [" << generated.video_features.shape()[0] << ", " 
                  << generated.video_features.shape()[1] << ", " 
                  << generated.video_features.shape()[2] << ", "
                  << generated.video_features.shape()[3] << "]" << std::endl;
        std::cout << "  Audio shape: [" << generated.audio_features.shape()[0] << ", " 
                  << generated.audio_features.shape()[1] << "]" << std::endl;
        std::cout << "  Text shape: [" << generated.text_features.shape()[0] << ", " 
                  << generated.text_features.shape()[1] << "]" << std::endl;
        
        printQualityMetrics(generated.quality, "Generated Content");
        
        // Enhance existing content
        std::cout << "\n--- Enhancing content ---" << std::endl;
        MultiModalContent enhanced = generator.enhanceContent(generated, "increase_clarity");
        
        std::cout << "Enhanced content quality improvement:" << std::endl;
        std::cout << "  Original overall quality: " << std::fixed << std::setprecision(3) 
                  << generated.quality.overall_quality << std::endl;
        std::cout << "  Enhanced overall quality: " << enhanced.quality.overall_quality << std::endl;
        std::cout << "  Improvement: " << (enhanced.quality.overall_quality - generated.quality.overall_quality) 
                  << std::endl;
        
        // Get generation statistics
        auto gen_stats = generator.getGenerationStatistics();
        std::cout << "\nGeneration Statistics:" << std::endl;
        for (const auto& [key, value] : gen_stats) {
            std::cout << "  " << key << ": " << std::fixed << std::setprecision(4) 
                      << value << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in content generation: " << e.what() << std::endl;
    }
}

// Demo 4: Quality Assessment and Optimization
void demoQualityAssessment() {
    std::cout << "\n=== DEMO 4: Quality Assessment and Optimization ===" << std::endl;
    
    try {
        QualityAssessmentSystem quality_system;
        
        // Set custom quality thresholds
        std::map<std::string, double> thresholds = {
            {"overall_quality", 0.75},
            {"visual_quality", 0.7},
            {"audio_quality", 0.7},
            {"text_coherence", 0.8},
            {"temporal_consistency", 0.85},
            {"semantic_alignment", 0.75}
        };
        quality_system.setQualityThresholds(thresholds);
        
        // Create test contents with different quality levels
        std::vector<MultiModalContent> test_contents;
        std::vector<std::string> content_labels = {"Low Quality", "Medium Quality", "High Quality"};
        
        for (int quality_level = 0; quality_level < 3; ++quality_level) {
            MultiModalContent content;
            
            // Simulate different quality levels
            double noise_level = 0.5 - (quality_level * 0.2);
            double feature_quality = 0.3 + (quality_level * 0.3);
            
            content.video_features = createSyntheticVideo({10, 3, 64, 64});
            content.audio_features = createSyntheticAudio({10, 32});
            content.text_features = createSyntheticText({10, 128});
            content.timestamps = generateTimestamps(10, 1.0);
            
            // Add noise to simulate quality differences
            for (size_t i = 0; i < content.video_features.size(); ++i) {
                content.video_features.data()[i] = std::max(0.0, std::min(1.0, 
                    content.video_features.data()[i] + noise_level * ((rand() % 100) / 100.0 - 0.5)));
            }
            
            test_contents.push_back(content);
        }
        
        // Assess quality for each content
        std::vector<ContentQualityMetrics> quality_history;
        for (size_t i = 0; i < test_contents.size(); ++i) {
            std::cout << "\n--- Assessing " << content_labels[i] << " ---" << std::endl;
            
            ContentQualityMetrics metrics = quality_system.assessContent(test_contents[i]);
            quality_history.push_back(metrics);
            
            printQualityMetrics(metrics, content_labels[i]);
            
            // Get improvement suggestions
            auto suggestions = quality_system.suggestImprovements(metrics);
            std::cout << "\nImprovement Suggestions:" << std::endl;
            for (const auto& suggestion : suggestions) {
                std::cout << "  - " << suggestion << std::endl;
            }
            
            // Get improvement priorities
            auto priorities = quality_system.computeImprovementPriorities(metrics);
            std::cout << "\nImprovement Priorities:" << std::endl;
            for (const auto& [aspect, priority] : priorities) {
                std::cout << "  " << aspect << ": " << std::fixed << std::setprecision(3) 
                          << priority << std::endl;
            }
        }
        
        // Analyze quality trends
        std::cout << "\n--- Quality Trend Analysis ---" << std::endl;
        auto trends = quality_system.analyzeQualityTrends(quality_history);
        std::cout << "Quality Trends:" << std::endl;
        for (const auto& [metric, trend] : trends) {
            std::cout << "  " << metric << ": " << std::fixed << std::setprecision(4) 
                      << trend << std::endl;
        }
        
        // Compare content quality
        if (quality_history.size() >= 2) {
            std::cout << "\n--- Content Comparison ---" << std::endl;
            ContentQualityMetrics comparison = quality_system.compareContent(
                test_contents[0], test_contents[2]);
            printQualityMetrics(comparison, "Quality Comparison");
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in quality assessment: " << e.what() << std::endl;
    }
}

// Demo 5: Real-time Streaming
void demoStreamingProcessing() {
    std::cout << "\n=== DEMO 5: Real-time Streaming Processing ===" << std::endl;
    
    try {        // Configure streaming fusion
        FusionConfig config;
        config.strategy = VideoAudioTextFusionStrategy::ATTENTION_FUSION;
        config.enable_real_time = true;
        config.temporal_window_size = 1.0;
        
        MultiModalFusionPipeline pipeline(config);
        StreamingFusionManager& streaming_manager = pipeline.getStreamingManager();
        
        streaming_manager.setBufferSize(10);
        streaming_manager.enableRealTimeMode(true);
        streaming_manager.setLatencyTarget(50.0); // 50ms target latency
        
        std::cout << "Streaming configuration:" << std::endl;
        std::cout << "  Buffer size: " << streaming_manager.getBufferSize() << std::endl;
        std::cout << "  Latency target: 50ms" << std::endl;
        
        // Simulate streaming chunks
        std::cout << "\n--- Simulating streaming data ---" << std::endl;
        std::vector<double> processing_times;
        
        for (size_t chunk_id = 0; chunk_id < 15; ++chunk_id) {
            StreamingChunk chunk;
            chunk.chunk_id = chunk_id;
            chunk.is_final_chunk = (chunk_id == 14);
            chunk.timestamp = std::chrono::high_resolution_clock::now();
            
            // Create chunk content
            chunk.content.video_features = createSyntheticVideo({4, 3, 32, 32});
            chunk.content.audio_features = createSyntheticAudio({4, 16});
            chunk.content.text_features = createSyntheticText({4, 64});
            chunk.content.timestamps = generateTimestamps(4, 0.25); // 250ms per chunk
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Add chunk to streaming buffer
            bool added = streaming_manager.addStreamingChunk(chunk);
            
            // Process available chunks
            if (added && chunk_id % 3 == 0) { // Process every 3 chunks
                auto processed_chunks = streaming_manager.processAvailableChunks();
                
                std::cout << "Chunk " << chunk_id << ": Processed " << processed_chunks.size() 
                          << " chunks" << std::endl;
                
                if (!processed_chunks.empty()) {
                    auto& last_processed = processed_chunks.back();
                    std::cout << "  Quality: " << std::fixed << std::setprecision(3)
                              << last_processed.quality.overall_quality << std::endl;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            processing_times.push_back(duration.count() / 1000.0); // Convert to ms
            
            // Simulate real-time processing delay
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        
        // Get streaming statistics
        std::cout << "\n--- Streaming Performance ---" << std::endl;
        auto streaming_stats = streaming_manager.getStreamingStatistics();
        std::cout << "Streaming Statistics:" << std::endl;
        for (const auto& [key, value] : streaming_stats) {
            std::cout << "  " << key << ": " << std::fixed << std::setprecision(2) 
                      << value << std::endl;
        }
        
        std::cout << "Current latency: " << std::fixed << std::setprecision(2) 
                  << streaming_manager.getCurrentLatency() << " ms" << std::endl;
        
        // Calculate average processing time
        double avg_processing_time = std::accumulate(processing_times.begin(), 
                                                    processing_times.end(), 0.0) / processing_times.size();
        std::cout << "Average processing time: " << std::fixed << std::setprecision(2) 
                  << avg_processing_time << " ms" << std::endl;
        
        // Flush remaining chunks
        streaming_manager.flushBuffer();
        std::cout << "Buffer flushed, final buffer size: " << streaming_manager.getBufferSize() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error in streaming processing: " << e.what() << std::endl;
    }
}

// Demo 6: Utility Functions and Advanced Operations
void demoUtilityFunctions() {
    std::cout << "\n=== DEMO 6: Utility Functions and Advanced Operations ===" << std::endl;
    
    try {
        // Test content creation utilities
        std::cout << "\n--- Testing FusionUtils ---" << std::endl;
        
        MultiModalContent empty_content = FusionUtils::createEmptyContent(8);
        std::cout << "Created empty content with sequence length: " << empty_content.get_sequence_length() << std::endl;
        
        // Create multiple contents for merging
        std::vector<MultiModalContent> contents_to_merge;
        for (int i = 0; i < 3; ++i) {
            MultiModalContent content;
            content.video_features = createSyntheticVideo({4, 3, 16, 16});
            content.audio_features = createSyntheticAudio({4, 8});
            content.text_features = createSyntheticText({4, 32});
            content.timestamps = generateTimestamps(4, 0.5);
            contents_to_merge.push_back(content);
        }
        
        MultiModalContent merged = FusionUtils::mergeContents(contents_to_merge);
        std::cout << "Merged " << contents_to_merge.size() << " contents:" << std::endl;
        std::cout << "  Final sequence length: " << merged.get_sequence_length() << std::endl;
        
        // Test temporal extraction
        MultiModalContent time_window = FusionUtils::extractTimeWindow(merged, 0.2, 1.0);
        std::cout << "Extracted time window (0.2s - 1.0s):" << std::endl;
        std::cout << "  Sequence length: " << time_window.get_sequence_length() << std::endl;
        
        // Test feature utilities
        Tensor features = createSyntheticVideo({10, 64});
        Tensor normalized = FusionUtils::normalizeFeatures(features);
        
        // Calculate normalization effect
        double original_mean = std::accumulate(features.data().begin(), features.data().end(), 0.0) / features.size();
        double normalized_mean = std::accumulate(normalized.data().begin(), normalized.data().end(), 0.0) / normalized.size();
        
        std::cout << "\nFeature normalization:" << std::endl;
        std::cout << "  Original mean: " << std::fixed << std::setprecision(4) << original_mean << std::endl;
        std::cout << "  Normalized mean: " << normalized_mean << std::endl;
        
        // Test quality utilities
        std::vector<ContentQualityMetrics> metrics_list;
        for (int i = 0; i < 3; ++i) {
            ContentQualityMetrics metrics;
            metrics.overall_quality = 0.6 + (i * 0.1);
            metrics.visual_quality = 0.5 + (i * 0.15);
            metrics.audio_quality = 0.7 + (i * 0.1);
            metrics.text_coherence = 0.8 + (i * 0.05);
            metrics.temporal_consistency = 0.75 + (i * 0.08);
            metrics.semantic_alignment = 0.65 + (i * 0.12);
            metrics.computational_efficiency = 0.9 - (i * 0.05);
            metrics.generation_mode = "test_mode_" + std::to_string(i);
            metrics_list.push_back(metrics);
        }
        
        ContentQualityMetrics combined = FusionUtils::combineQualityMetrics(metrics_list);
        std::cout << "\nCombined quality metrics:" << std::endl;
        printQualityMetrics(combined, "Combined");
        
        // Test configuration optimization
        MultiModalContent sample_content;
        sample_content.video_features = createSyntheticVideo({12, 3, 32, 32});
        sample_content.audio_features = createSyntheticAudio({12, 16});
        sample_content.text_features = createSyntheticText({12, 64});
        sample_content.timestamps = generateTimestamps(12, 1.5);
        
        FusionConfig optimal_config = FusionUtils::createOptimalConfig(sample_content);
        std::cout << "\nOptimal fusion configuration:" << std::endl;
        std::cout << "  Strategy: " << static_cast<int>(optimal_config.strategy) << std::endl;
        std::cout << "  Video weight: " << std::fixed << std::setprecision(3) << optimal_config.fusion_weight_video << std::endl;
        std::cout << "  Audio weight: " << optimal_config.fusion_weight_audio << std::endl;
        std::cout << "  Text weight: " << optimal_config.fusion_weight_text << std::endl;
        std::cout << "  Temporal window: " << optimal_config.temporal_window_size << " seconds" << std::endl;
        
        // Analyze content characteristics
        auto characteristics = FusionUtils::analyzeContentCharacteristics(sample_content);
        std::cout << "\nContent characteristics:" << std::endl;
        for (const auto& [characteristic, value] : characteristics) {
            std::cout << "  " << characteristic << ": " << std::fixed << std::setprecision(4) 
                      << value << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in utility functions demo: " << e.what() << std::endl;
    }
}

// Performance benchmark
void performanceBenchmark() {
    std::cout << "\n=== PERFORMANCE BENCHMARK ===" << std::endl;
    
    try {
        std::vector<std::tuple<std::string, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>> test_cases = {
            {"Small", {8, 3, 32, 32}, {8, 16}, {8, 64}},
            {"Medium", {16, 3, 64, 64}, {16, 32}, {16, 128}},
            {"Large", {32, 3, 128, 128}, {32, 64}, {32, 256}}
        };
          FusionConfig config;
        config.strategy = VideoAudioTextFusionStrategy::ATTENTION_FUSION;
        config.enable_quality_feedback = true;
        
        for (const auto& [size_name, video_shape, audio_shape, text_shape] : test_cases) {
            std::cout << "\n--- " << size_name << " Content Benchmark ---" << std::endl;
            
            // Create test content
            MultiModalContent content;
            content.video_features = createSyntheticVideo(video_shape);
            content.audio_features = createSyntheticAudio(audio_shape);
            content.text_features = createSyntheticText(text_shape);
            content.timestamps = generateTimestamps(video_shape[0], 2.0);
            
            std::cout << "Content size:" << std::endl;
            std::cout << "  Video: " << content.video_features.size() << " elements" << std::endl;
            std::cout << "  Audio: " << content.audio_features.size() << " elements" << std::endl;
            std::cout << "  Text: " << content.text_features.size() << " elements" << std::endl;
            
            // Benchmark fusion pipeline
            MultiModalFusionPipeline pipeline(config);
            
            auto start = std::chrono::high_resolution_clock::now();
            MultiModalContent result = pipeline.fusePipeline(content);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
                      << (content.video_features.size() + content.audio_features.size() + content.text_features.size()) 
                         / static_cast<double>(duration.count()) * 1000.0 << " elements/second" << std::endl;
            
            printQualityMetrics(result.quality, size_name + " Result");
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in performance benchmark: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CLModel Week 15: Video-Audio-Text   " << std::endl;
    std::cout << "       Fusion Pipeline Demo           " << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // Run all demos
        demoBasicContentProcessing();
        demoFusionStrategies();
        demoContentGeneration();
        demoQualityAssessment();
        demoStreamingProcessing();
        demoUtilityFunctions();
        performanceBenchmark();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "   All demos completed successfully!   " << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "\nDemo failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
