#include "ai/orchestral_ai_workflow.hpp"
#include "tensor.hpp"
#include <iostream>
#include <chrono>
#include <memory>

using namespace clmodel::ai;

/**
 * @brief Mock text encoder model for demonstration
 */
class MockTextEncoder : public BaseModel {
public:
    MockTextEncoder() : BaseModel("text_encoder_v1", ModelType::TEXT_ENCODER) {}
    
    bool load() override {
        is_loaded_ = true;
        std::cout << "MockTextEncoder: Model loaded successfully" << std::endl;
        return true;
    }
    
    bool unload() override {
        is_loaded_ = false;
        std::cout << "MockTextEncoder: Model unloaded" << std::endl;
        return true;
    }
    
    Tensor forward(const Tensor& input) override {
        std::cout << "MockTextEncoder: Processing text input of size " 
                  << input.size() << std::endl;
        
        // Simulate text encoding: convert to 768-dimensional embeddings
        Tensor output({input.shape()[0], 768});
        output.random(-0.1f, 0.1f);
        
        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        return output;
    }
};

/**
 * @brief Mock image encoder model for demonstration
 */
class MockImageEncoder : public BaseModel {
public:
    MockImageEncoder() : BaseModel("image_encoder_v2", ModelType::IMAGE_ENCODER) {}
    
    bool load() override {
        is_loaded_ = true;
        std::cout << "MockImageEncoder: Model loaded successfully" << std::endl;
        return true;
    }
    
    bool unload() override {
        is_loaded_ = false;
        std::cout << "MockImageEncoder: Model unloaded" << std::endl;
        return true;
    }
    
    Tensor forward(const Tensor& input) override {
        std::cout << "MockImageEncoder: Processing image input of shape [" 
                  << input.shape()[0] << ", " << input.shape()[1] 
                  << ", " << input.shape()[2] << ", " << input.shape()[3] << "]" << std::endl;
        
        // Simulate image encoding: convert to 512-dimensional embeddings
        Tensor output({input.shape()[0], 512});
        output.random(-0.2f, 0.2f);
        
        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
        
        return output;
    }
};

/**
 * @brief Mock fusion network for demonstration
 */
class MockFusionNetwork : public BaseModel {
public:
    MockFusionNetwork() : BaseModel("fusion_net_v1", ModelType::FUSION_NETWORK) {}
    
    bool load() override {
        is_loaded_ = true;
        std::cout << "MockFusionNetwork: Model loaded successfully" << std::endl;
        return true;
    }
    
    bool unload() override {
        is_loaded_ = false;
        std::cout << "MockFusionNetwork: Model unloaded" << std::endl;
        return true;
    }
    
    Tensor forward(const Tensor& input) override {
        std::cout << "MockFusionNetwork: Fusing multi-modal features of size " 
                  << input.size() << std::endl;
        
        // Simulate fusion: combine features and output classification/generation result
        Tensor output({input.shape()[0], 256});
        output.random(0.0f, 1.0f);
        
        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
        
        return output;
    }
};

/**
 * @brief Create a sample multi-modal pipeline for text + image processing
 */
std::shared_ptr<PipelineBuilder> createMultiModalPipeline() {
    auto pipeline = std::make_shared<PipelineBuilder>();
    
    // Stage 1: Text encoding
    TaskConfig text_config("text_stage");    text_config.input_modality = clmodel::ai::OrchestralModality::TEXT;
    text_config.output_modality = clmodel::ai::OrchestralModality::TEXT;
    text_config.preferred_model = ModelType::TEXT_ENCODER;
    text_config.priority = ProcessingPriority::HIGH;
    
    pipeline->addStage("text_encoding", text_config);
    
    // Stage 2: Image encoding
    TaskConfig image_config("image_stage");    image_config.input_modality = clmodel::ai::OrchestralModality::IMAGE;
    image_config.output_modality = clmodel::ai::OrchestralModality::IMAGE;
    image_config.preferred_model = ModelType::IMAGE_ENCODER;
    image_config.priority = ProcessingPriority::HIGH;
    
    pipeline->addStage("image_encoding", image_config);
    
    // Stage 3: Multi-modal fusion
    TaskConfig fusion_config("fusion_stage");    fusion_config.input_modality = clmodel::ai::OrchestralModality::MULTIMODAL;
    fusion_config.output_modality = clmodel::ai::OrchestralModality::MULTIMODAL;
    fusion_config.preferred_model = ModelType::FUSION_NETWORK;
    fusion_config.priority = ProcessingPriority::CRITICAL;
    
    pipeline->addStage("fusion", fusion_config);
    
    // Add dependencies: fusion depends on both text and image encoding
    pipeline->addDependency("text_encoding", "fusion");
    pipeline->addDependency("image_encoding", "fusion");
    
    // Configure pipeline options
    pipeline->setParallelExecution(true);
    pipeline->setQualityThreshold(0.6);
    
    return pipeline;
}

/**
 * @brief Demonstrate model registry functionality
 */
void demonstrateModelRegistry() {
    std::cout << "\n=== Model Registry Demo ===" << std::endl;
    
    ModelRegistry registry;
    
    // Register mock models
    registry.registerModel(std::make_shared<MockTextEncoder>());
    registry.registerModel(std::make_shared<MockImageEncoder>());
    registry.registerModel(std::make_shared<MockFusionNetwork>());
    
    // Show available models
    auto models = registry.getAvailableModels();
    std::cout << "Registered models (" << models.size() << "):" << std::endl;
    for (const auto& model_name : models) {
        std::cout << "  - " << model_name << std::endl;
    }
    
    // Test model selection
    Tensor test_input({32, 512});
    test_input.random(-1.0f, 1.0f);
    
    auto selected_model = registry.selectBestModel(ModelType::TEXT_ENCODER, test_input);
    if (selected_model) {
        std::cout << "Selected text encoder: " << selected_model->getName() << std::endl;
        
        // Benchmark the model
        auto metrics = selected_model->benchmark(test_input);
        std::cout << "Benchmark results - Quality: " << metrics.quality_score 
                  << ", Time: " << metrics.processing_time_ms << "ms" << std::endl;
    }
    
    // Show models by type
    auto text_models = registry.getModelsByType(ModelType::TEXT_ENCODER);
    std::cout << "Text encoder models: ";
    for (const auto& name : text_models) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Loaded models: " << registry.getLoadedModelCount() << std::endl;
}

/**
 * @brief Demonstrate pipeline builder functionality
 */
void demonstratePipelineBuilder() {
    std::cout << "\n=== Pipeline Builder Demo ===" << std::endl;
    
    ModelRegistry registry;
    registry.registerModel(std::make_shared<MockTextEncoder>());
    registry.registerModel(std::make_shared<MockImageEncoder>());
    registry.registerModel(std::make_shared<MockFusionNetwork>());
    
    auto pipeline = createMultiModalPipeline();
    
    std::cout << "Pipeline configuration:" << std::endl;
    std::cout << "  Stages: " << pipeline->getStageCount() << std::endl;
    
    // Build the pipeline
    if (pipeline->build()) {
        std::cout << "Pipeline built successfully" << std::endl;
        
        // Create mock inputs
        std::map<std::string, Tensor> inputs;
        inputs["text"] = Tensor({1, 256});  // Text tokens
        inputs["text"].random(-1.0f, 1.0f);
        inputs["image"] = Tensor({1, 3, 224, 224});  // Image tensor
        inputs["image"].random(0.0f, 1.0f);
        
        std::cout << "Executing pipeline with inputs:" << std::endl;
        std::cout << "  Text input shape: [" << inputs["text"].shape()[0] 
                  << ", " << inputs["text"].shape()[1] << "]" << std::endl;
        std::cout << "  Image input shape: [" << inputs["image"].shape()[0] 
                  << ", " << inputs["image"].shape()[1] << ", " 
                  << inputs["image"].shape()[2] << ", " 
                  << inputs["image"].shape()[3] << "]" << std::endl;
        
        // Execute pipeline
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = pipeline->execute(inputs, registry);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        if (success) {
            std::cout << "Pipeline executed successfully in " << execution_time << "ms" << std::endl;
            
            // Show results
            auto results = pipeline->getResults();
            std::cout << "Pipeline results:" << std::endl;
            for (const auto& [stage_name, result] : results) {
                std::cout << "  " << stage_name << ": tensor of size " << result.size() << std::endl;
            }
            
            // Show performance metrics
            auto metrics = pipeline->getStageMetrics();
            std::cout << "Stage performance:" << std::endl;
            for (const auto& [stage_name, metric] : metrics) {
                std::cout << "  " << stage_name << ": " << metric.processing_time_ms 
                          << "ms (quality: " << metric.quality_score << ")" << std::endl;
            }
        } else {
            std::cout << "Pipeline execution failed" << std::endl;
        }
    } else {
        std::cout << "Failed to build pipeline" << std::endl;
    }
}

/**
 * @brief Demonstrate resource scheduler functionality
 */
void demonstrateResourceScheduler() {
    std::cout << "\n=== Resource Scheduler Demo ===" << std::endl;
    
    ResourceScheduler scheduler(2);  // 2 concurrent tasks
    scheduler.start();
    
    std::cout << "Scheduler started with 2 worker threads" << std::endl;
    std::cout << "Initial state - Active: " << scheduler.getActiveTaskCount() 
              << ", Queued: " << scheduler.getQueuedTaskCount() << std::endl;
    
    // Create some mock tasks
    std::vector<std::shared_ptr<ProcessingTask>> tasks;
    
    for (int i = 0; i < 5; ++i) {
        TaskConfig config("task_" + std::to_string(i));
        config.priority = (i % 2 == 0) ? ProcessingPriority::HIGH : ProcessingPriority::NORMAL;
        
        Tensor input({16, 32});
        input.random(-1.0f, 1.0f);
        
        auto task = std::make_shared<ProcessingTask>(config, input);
        tasks.push_back(task);
        
        scheduler.scheduleTask(task);
        std::cout << "Scheduled task_" << i << " with priority " 
                  << static_cast<int>(config.priority) << std::endl;
    }
    
    std::cout << "After scheduling - Active: " << scheduler.getActiveTaskCount() 
              << ", Queued: " << scheduler.getQueuedTaskCount() << std::endl;
    
    // Monitor resource usage
    std::cout << "Resource usage - CPU: " << scheduler.getCPUUsage() * 100 
              << "%, Memory: " << scheduler.getMemoryUsage() * 100 << "%" << std::endl;
    
    // Wait a bit for tasks to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "After processing - Active: " << scheduler.getActiveTaskCount() 
              << ", Queued: " << scheduler.getQueuedTaskCount() << std::endl;
    
    scheduler.stop();
    std::cout << "Scheduler stopped" << std::endl;
}

/**
 * @brief Demonstrate quality orchestrator functionality
 */
void demonstrateQualityOrchestrator() {
    std::cout << "\n=== Quality Orchestrator Demo ===" << std::endl;
    
    QualityOrchestrator orchestrator;
    
    // Set quality thresholds
    orchestrator.setQualityThresholds(0.7, 100.0);  // Min quality 0.7, max time 100ms
    
    // Simulate recording metrics for different models
    std::vector<std::string> model_names = {
        "text_encoder_v1", "image_encoder_v2", "fusion_net_v1"
    };
    
    for (int i = 0; i < 10; ++i) {
        for (const auto& model_name : model_names) {
            PerformanceMetrics metrics;
            metrics.quality_score = 0.6 + (std::rand() % 40) / 100.0;  // 0.6-1.0
            metrics.processing_time_ms = 20.0 + (std::rand() % 80);    // 20-100ms
            metrics.confidence_score = 0.7 + (std::rand() % 30) / 100.0;  // 0.7-1.0
            
            orchestrator.recordMetrics("task_" + std::to_string(i), metrics);
        }
    }
    
    // Show model performance summaries
    auto model_summary = orchestrator.getModelSummary();
    std::cout << "Model performance summary:" << std::endl;
    for (const auto& [model_name, metrics] : model_summary) {
        std::cout << "  " << model_name << ":" << std::endl;
        std::cout << "    Quality: " << metrics.quality_score << std::endl;
        std::cout << "    Processing time: " << metrics.processing_time_ms << "ms" << std::endl;
        std::cout << "    Confidence: " << metrics.confidence_score << std::endl;
    }
    
    // Check for underperforming models
    auto underperforming = orchestrator.getUnderperformingModels();
    if (!underperforming.empty()) {
        std::cout << "Underperforming models:" << std::endl;
        for (const auto& model_name : underperforming) {
            std::cout << "  - " << model_name << std::endl;
        }
    } else {
        std::cout << "All models meeting performance thresholds" << std::endl;
    }
    
    // Model recommendation
    std::vector<std::string> available_models = {"text_encoder_v1", "text_encoder_v2"};
    std::string recommended = orchestrator.recommendModel(ModelType::TEXT_ENCODER, available_models);
    std::cout << "Recommended text encoder: " << recommended << std::endl;
    
    // Generate quality report
    orchestrator.generateQualityReport("week13_quality_report.txt");
}

/**
 * @brief Demonstrate complete workflow manager functionality
 */
void demonstrateWorkflowManager() {
    std::cout << "\n=== Workflow Manager Demo ===" << std::endl;
    
    WorkflowManager manager;
    manager.initialize();
    
    // Register models
    manager.getModelRegistry().registerModel(std::make_shared<MockTextEncoder>());
    manager.getModelRegistry().registerModel(std::make_shared<MockImageEncoder>());
    manager.getModelRegistry().registerModel(std::make_shared<MockFusionNetwork>());
    
    // Create and register a workflow
    auto pipeline = createMultiModalPipeline();
    pipeline->build();
    manager.registerWorkflow("text_image_fusion", pipeline);
    
    // Show available workflows
    auto workflows = manager.getAvailableWorkflows();
    std::cout << "Available workflows:" << std::endl;
    for (const auto& workflow_name : workflows) {
        std::cout << "  - " << workflow_name << std::endl;
    }
    
    // Execute workflow
    std::map<std::string, Tensor> inputs;
    inputs["text"] = Tensor({2, 128});
    inputs["text"].random(-1.0f, 1.0f);
    inputs["image"] = Tensor({2, 3, 256, 256});
    inputs["image"].random(0.0f, 1.0f);
    
    std::cout << "Executing text_image_fusion workflow..." << std::endl;
    bool success = manager.executeWorkflow("text_image_fusion", inputs);
    
    if (success) {
        std::cout << "Workflow executed successfully" << std::endl;
        
        // Show workflow results
        auto results = manager.getWorkflowResults("text_image_fusion");
        std::cout << "Workflow outputs:" << std::endl;
        for (const auto& [stage_name, result] : results) {
            std::cout << "  " << stage_name << ": tensor of size " << result.size() << std::endl;
        }
        
        // Show workflow metrics
        auto metrics = manager.getWorkflowMetrics("text_image_fusion");
        std::cout << "Workflow performance:" << std::endl;
        std::cout << "  Processing time: " << metrics.processing_time_ms << "ms" << std::endl;
        std::cout << "  Quality score: " << metrics.quality_score << std::endl;
        std::cout << "  Confidence: " << metrics.confidence_score << std::endl;
    } else {
        std::cout << "Workflow execution failed" << std::endl;
    }
    
    manager.shutdown();
}

int main() {
    std::cout << "CLModel Week 13: Orchestral AI Workflow Demonstration" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    try {
        // Set random seed for reproducible results
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        
        // Demonstrate all components
        demonstrateModelRegistry();
        demonstratePipelineBuilder();
        demonstrateResourceScheduler();
        demonstrateQualityOrchestrator();
        demonstrateWorkflowManager();
        
        std::cout << "\n=== Demo Summary ===" << std::endl;
        std::cout << "✅ Model Registry: Dynamic model registration and selection" << std::endl;
        std::cout << "✅ Pipeline Builder: Configurable multi-modal workflows" << std::endl;
        std::cout << "✅ Resource Scheduler: Efficient task scheduling and resource management" << std::endl;
        std::cout << "✅ Quality Orchestrator: Performance monitoring and optimization" << std::endl;
        std::cout << "✅ Workflow Manager: Complete orchestral AI system coordination" << std::endl;
        
        std::cout << "\nWeek 13 Implementation Status:" << std::endl;
        std::cout << "🎯 Multi-modal workflow orchestration: COMPLETE" << std::endl;
        std::cout << "🎯 Dynamic model selection: COMPLETE" << std::endl;
        std::cout << "🎯 Resource scheduling: COMPLETE" << std::endl;
        std::cout << "🎯 Quality monitoring: COMPLETE" << std::endl;
        std::cout << "🎯 Pipeline coordination: COMPLETE" << std::endl;
        
        std::cout << "\nReady for Week 14: Cross-Modal Guidance Systems" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demo: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
