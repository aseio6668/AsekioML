cmake_minimum_required(VERSION 3.16)
project(AsekioML VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /arch:AVX2)
    add_definitions(-D_WIN32_WINNT=0x0601)
else()
    add_compile_options(-Wall -Wextra -pedantic -mavx2 -mfma)
endif()

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/matrix.cpp
    src/layer.cpp
    src/activation.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/network.cpp
    src/dataset.cpp    src/regularization.cpp
    src/model_serialization.cpp
    src/simd_matrix.cpp
    src/memory_optimization.cpp
    src/modern_api.cpp
    src/production_features.cpp
    src/gpu_acceleration.cpp
    src/profiling.cpp
    src/tokenizer.cpp
    src/threading.cpp    # AI Extensions (Phase 1)
    src/tensor.cpp
    src/ai/cnn_layers.cpp
    src/ai/memory_manager.cpp
    src/ai/compute_engine.cpp
    src/ai/attention_layers.cpp
    # Phase 2 - Image Processing
    src/ai/image_processing.cpp
    # Phase 2 - Audio Processing
    src/ai/audio_processing.cpp
    # Phase 2 - Text-to-Image
    src/ai/text_to_image.cpp    # Phase 2 - Text-to-Speech (Week 9-12)
    src/ai/text_to_speech.cpp    # Phase 3 - Multi-Modal Integration    # Phase 3 - Multi-Modal Integration (Week 15)
    src/ai/multimodal_attention.cpp
    src/ai/advanced_fusion_networks.cpp
    # Phase 3 - Video Generation Foundation (Week 9-12)    
    src/ai/video_tensor_ops.cpp
    src/ai/video_diffusion.cpp
    src/ai/simple_video_diffusion.cpp# Week 11 - Advanced Frame Interpolation & Motion Synthesis
    src/ai/advanced_frame_interpolation_working.cpp  # Week 11: Enabled with complete implementation    # Week 12 - Audio-Visual Synchronization
    src/ai/audio_visual_sync.cpp  # Week 12: Audio-Visual sync, lip-sync, and streaming    # Week 13 - Orchestral AI Workflow
    src/ai/orchestral_ai_workflow.cpp  # Week 13: Multi-modal workflow orchestration
    # Week 14 - Cross-Modal Guidance
    src/ai/cross_modal_guidance.cpp  # Week 14: Cross-modal guidance systems
    # Week 15 - Video-Audio-Text Fusion Pipeline
    src/ai/video_audio_text_fusion.cpp  # Week 15: Multi-modal fusion pipeline    # Week 16 - Advanced Orchestral AI System
    src/ai/orchestral_ai_director.cpp  # Week 16: Orchestral AI system coordination
    src/ai/dynamic_model_dispatcher.cpp  # Week 16: Dynamic model routing and load balancing
    src/ai/real_time_content_pipeline.cpp  # Week 16: Real-time content generation pipeline    src/ai/adaptive_quality_engine.cpp  # Week 16: Real-time quality optimization engine
    src/ai/production_streaming_manager.cpp  # Week 16: Enterprise streaming infrastructure      # Week 17 - Production Polish & Enterprise Features
    # src/api/rest_api_server.cpp  # Week 17: REST API server for web integration - TEMPORARILY DISABLED    
    src/advanced_layers.cpp  # Re-enabled after fixing abstract class issues
    
    # Additional AI components that need to be included
    src/ai/adaptive_pipeline_manager.cpp  # Cross-modal guidance support
    src/ai/streaming_fusion_manager.cpp   # Video-audio-text fusion support
)

# Header files
set(HEADERS
    include/matrix.hpp
    include/layer.hpp
    include/activation.hpp
    include/loss.hpp
    include/optimizer.hpp
    include/network.hpp
    include/dataset.hpp
    include/asekioml.hpp
    include/regularization.hpp
    include/model_serialization.hpp
    include/simd_matrix.hpp
    include/memory_optimization.hpp
    include/modern_api.hpp
    include/production_features.hpp    include/gpu_acceleration.hpp
    include/advanced_layers.hpp    # AI Extensions (Phase 1)
    include/tensor.hpp    include/ai/cnn_layers.hpp
    include/ai/memory_manager.hpp
    include/ai/compute_engine.hpp    include/ai/attention_layers.hpp    # Multi-Modal Integration (Week 15)
    include/ai/multimodal_attention.hpp
    include/ai/advanced_fusion_networks.hpp
    # Phase 3 - Video Generation Foundation
    include/ai/video_tensor_ops.hpp
    include/ai/video_diffusion.hpp
    include/ai/simple_video_diffusion.hpp    # Week 11 - Advanced Frame Interpolation & Motion Synthesis
    include/ai/advanced_frame_interpolation.hpp    # Week 12 - Audio-Visual Synchronization
    include/ai/audio_visual_sync.hpp    # Week 13 - Orchestral AI Workflow
    include/ai/orchestral_ai_workflow.hpp
    # Week 14 - Cross-Modal Guidance Systems  
    include/ai/cross_modal_guidance.hpp    # Week 15 - Video-Audio-Text Fusion Pipeline
    include/ai/video_audio_text_fusion.hpp    # Week 16 - Advanced Orchestral AI System  
    include/ai/orchestral_ai_director.hpp
    include/ai/dynamic_model_dispatcher.hpp  # Week 16: Dynamic model dispatcher    include/ai/real_time_content_pipeline.hpp  # Week 16: Real-time content pipeline
    
    # Week 17 - Production Polish & Enterprise Features
    # include/api/rest_api_server.hpp  # Week 17: REST API server interface - TEMPORARILY DISABLED
    
    include/model_zoo.hpp
    include/profiling.hpp
    include/tokenizer.hpp
    include/threading.hpp
)

# Create library
add_library(asekioml STATIC ${SOURCES} ${HEADERS})

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(asekioml PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(asekioml PUBLIC ASEKIOML_OPENMP_SUPPORT)
endif()

# Create example executable
add_executable(example examples/example.cpp)
target_link_libraries(example asekioml)

# Create advanced features demo
add_executable(advanced_demo examples/advanced_features_demo.cpp)
target_link_libraries(advanced_demo asekioml)

# Create GPU demo
add_executable(gpu_demo examples/gpu_demo.cpp)
target_link_libraries(gpu_demo asekioml)

# Create transparency demo
add_executable(transparency_demo examples/transparency_demo.cpp)
target_link_libraries(transparency_demo asekioml)

# Create Python pain points demo
add_executable(python_pain_points_demo examples/python_pain_points_demo.cpp)
target_link_libraries(python_pain_points_demo asekioml)

# Create statistical monitoring demo
add_executable(statistical_monitoring_demo examples/statistical_monitoring_demo.cpp)
target_link_libraries(statistical_monitoring_demo asekioml)

# Create advanced ML features demo
add_executable(advanced_ml_features_demo examples/advanced_ml_features_demo.cpp)
target_link_libraries(advanced_ml_features_demo asekioml)

# Create advanced training demo
add_executable(advanced_training_demo examples/advanced_training_demo.cpp)
target_link_libraries(advanced_training_demo asekioml)

# Create model serialization demo
add_executable(model_serialization_demo examples/model_serialization_demo.cpp)
target_link_libraries(model_serialization_demo asekioml)

# Create debug training demo
add_executable(debug_training debug_training.cpp)
target_link_libraries(debug_training asekioml)

# Create simple dropout test
add_executable(simple_dropout_test simple_dropout_test.cpp)
target_link_libraries(simple_dropout_test asekioml)

# Create tests executable
add_executable(tests tests/test_main.cpp)
target_link_libraries(tests asekioml)

# Create CNN layers demo
add_executable(cnn_layers_demo examples/cnn_layers_demo.cpp)
target_link_libraries(cnn_layers_demo asekioml)

# Create AI infrastructure demo
add_executable(ai_infrastructure_demo examples/ai_infrastructure_demo.cpp)
target_link_libraries(ai_infrastructure_demo asekioml)

# Create AI infrastructure demo (simplified)
add_executable(ai_infrastructure_demo_simple examples/ai_infrastructure_demo_simple.cpp)
target_link_libraries(ai_infrastructure_demo_simple asekioml)

# Create image processing demo (Phase 2)
add_executable(image_processing_demo examples/image_processing_demo.cpp)
target_link_libraries(image_processing_demo asekioml)

# Create audio processing demo (Phase 2)
add_executable(audio_processing_demo examples/audio_processing_demo.cpp)
target_link_libraries(audio_processing_demo asekioml)

# Create text-to-image demo (Phase 2)
add_executable(text_to_image_demo examples/text_to_image_demo.cpp)
target_link_libraries(text_to_image_demo asekioml)

# Create text-to-speech demo (Phase 2)
add_executable(text_to_speech_demo examples/text_to_speech_demo.cpp)
target_link_libraries(text_to_speech_demo asekioml)

# Create multi-modal attention demo (Phase 3)
add_executable(multimodal_attention_demo examples/multimodal_attention_demo.cpp)
target_link_libraries(multimodal_attention_demo asekioml)

# Create multi-modal basic test (Phase 3)
add_executable(multimodal_basic_test examples/multimodal_basic_test.cpp)
target_link_libraries(multimodal_basic_test asekioml)

# Create advanced fusion demo (Phase 3)
add_executable(advanced_fusion_demo examples/advanced_fusion_demo.cpp)
target_link_libraries(advanced_fusion_demo asekioml)

# Create advanced fusion test (Phase 3)
add_executable(advanced_fusion_test examples/advanced_fusion_test.cpp)
target_link_libraries(advanced_fusion_test asekioml)

# Debug executable for image filters
add_executable(debug_image_filters debug_image_filters.cpp)
target_link_libraries(debug_image_filters asekioml)
if(OpenMP_CXX_FOUND)
    target_link_libraries(debug_image_filters OpenMP::OpenMP_CXX)
endif()

# Debug executable for text-to-image integration
add_executable(debug_text_to_image_integration debug_text_to_image_integration.cpp)
target_link_libraries(debug_text_to_image_integration asekioml)
if(OpenMP_CXX_FOUND)
    target_link_libraries(debug_text_to_image_integration OpenMP::OpenMP_CXX)
endif()

# Installation rules
install(TARGETS asekioml DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/asekioml)

# Add custom target for performance tests
add_custom_target(perf_test
    COMMAND advanced_demo
    DEPENDS advanced_demo
    COMMENT "Running performance demonstrations"
)

# GPU Support: CUDA (NVIDIA), ROCm (AMD), OpenCL (Cross-platform)
set(GPU_SUPPORT_FOUND FALSE)

# Try NVIDIA CUDA first
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_compile_definitions(asekioml PUBLIC ASEKIOML_CUDA_SUPPORT)
    
    # Find and link CUDA libraries
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(asekioml PUBLIC CUDA::cudart CUDA::cublas)
    
    # Try to find cuDNN (optional)
    find_library(CUDNN_LIBRARY cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    if(CUDNN_LIBRARY)
        target_link_libraries(asekioml PUBLIC ${CUDNN_LIBRARY})
        target_compile_definitions(asekioml PUBLIC ASEKIOML_CUDNN_SUPPORT)
        message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
    else()
        message(STATUS "cuDNN not found - some GPU features may be limited")
    endif()
    
    set(GPU_SUPPORT_FOUND TRUE)
    message(STATUS "NVIDIA CUDA found - GPU acceleration available")
endif()

# Try AMD ROCm if CUDA not found
if(NOT GPU_SUPPORT_FOUND)
    find_package(hip QUIET)
    if(hip_FOUND)
        target_compile_definitions(asekioml PUBLIC ASEKIOML_ROCM_SUPPORT)
        target_link_libraries(asekioml PUBLIC hip::host)
        
        # Try to find rocBLAS
        find_package(rocblas QUIET)
        if(rocblas_FOUND)
            target_link_libraries(asekioml PUBLIC roc::rocblas)
            target_compile_definitions(asekioml PUBLIC ASEKIOML_ROCBLAS_SUPPORT)
            message(STATUS "rocBLAS found")
        endif()
        
        set(GPU_SUPPORT_FOUND TRUE)
        message(STATUS "AMD ROCm found - GPU acceleration available")
    endif()
endif()

# Try OpenCL as fallback (works with Intel, AMD, NVIDIA)
if(NOT GPU_SUPPORT_FOUND)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        target_compile_definitions(asekioml PUBLIC ASEKIOML_OPENCL_SUPPORT)
        target_link_libraries(asekioml PUBLIC OpenCL::OpenCL)
        set(GPU_SUPPORT_FOUND TRUE)
        message(STATUS "OpenCL found - Cross-platform GPU acceleration available")
    endif()
endif()

if(NOT GPU_SUPPORT_FOUND)
    message(STATUS "No GPU acceleration found - CPU-only build")
    message(STATUS "To enable GPU support, install:")
    message(STATUS "  - NVIDIA: CUDA Toolkit")
    message(STATUS "  - AMD: ROCm")
    message(STATUS "  - Intel/Cross-platform: OpenCL")
endif()

# Installation
install(TARGETS asekioml DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/asekioml)

# Video tensor operations demo (Phase 3, Week 9)
add_executable(video_tensor_operations_demo examples/video_tensor_operations_demo.cpp)
target_link_libraries(video_tensor_operations_demo asekioml)

# Video diffusion demo (Phase 3, Week 10)
add_executable(video_diffusion_demo examples/video_diffusion_demo.cpp)
target_link_libraries(video_diffusion_demo asekioml)

# Simple video diffusion demo (Phase 3, Week 10)
add_executable(simple_video_diffusion_demo examples/simple_video_diffusion_demo.cpp)
target_link_libraries(simple_video_diffusion_demo asekioml)

# Advanced frame interpolation demo (Phase 3, Week 11)
add_executable(advanced_frame_interpolation_demo examples/advanced_frame_interpolation_demo.cpp)
target_link_libraries(advanced_frame_interpolation_demo asekioml)

# Advanced frame interpolation simple demo (Phase 3, Week 11) 
add_executable(advanced_frame_interpolation_demo_simple examples/advanced_frame_interpolation_demo_simple.cpp)
target_link_libraries(advanced_frame_interpolation_demo_simple asekioml)

# Audio-visual synchronization demo (Phase 3, Week 12)
add_executable(audio_visual_sync_demo examples/audio_visual_sync_demo.cpp)
target_link_libraries(audio_visual_sync_demo asekioml)

# Audio-visual synchronization simple demo (Phase 3, Week 12) 
add_executable(audio_visual_sync_demo_simple examples/audio_visual_sync_demo_simple.cpp)
target_link_libraries(audio_visual_sync_demo_simple asekioml)

# Orchestral AI workflow demo (Phase 3, Week 13)
add_executable(orchestral_ai_workflow_demo examples/orchestral_ai_workflow_demo.cpp)
target_link_libraries(orchestral_ai_workflow_demo asekioml)

# Cross-modal guidance demo (Phase 3, Week 14)
add_executable(cross_modal_guidance_demo examples/cross_modal_guidance_demo.cpp)
target_link_libraries(cross_modal_guidance_demo asekioml)

# Cross-modal guidance demo simplified (Phase 3, Week 14)
add_executable(cross_modal_guidance_demo_simple examples/cross_modal_guidance_demo_simple.cpp)
target_link_libraries(cross_modal_guidance_demo_simple asekioml)

# Video-Audio-Text Fusion Pipeline demo (Phase 3, Week 15)
add_executable(video_audio_text_fusion_demo examples/video_audio_text_fusion_demo.cpp)
target_link_libraries(video_audio_text_fusion_demo asekioml)

# Week 15 Simple Demo (Working implementation)
add_executable(week15_simple_demo examples/week15_simple_demo.cpp)
target_link_libraries(week15_simple_demo asekioml)

# Week 15 Basic Demo (Foundation validation)
add_executable(week15_basic_demo examples/week15_basic_demo.cpp)
target_link_libraries(week15_basic_demo asekioml)

# Week 15 Concept Demo (Standalone demonstration)
add_executable(week15_concept_demo examples/week15_concept_demo.cpp)
target_link_libraries(week15_concept_demo asekioml)

# Week 16 Concept Demo (Orchestral AI System foundation)
add_executable(week16_concept_demo examples/week16_concept_demo.cpp)
target_link_libraries(week16_concept_demo asekioml)

# Week 16 Final Demo (Complete Orchestral AI System - Day 7)
add_executable(week16_final_demo examples/week16_final_demo.cpp)
target_link_libraries(week16_final_demo asekioml)

# Week 17 REST API Server Demo (Production Polish & Enterprise Features - Days 1-2)
# add_executable(rest_api_server_demo examples/rest_api_server_demo.cpp)
# target_link_libraries( asekioml)

# Installation
install(TARGETS asekioml DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/asekioml)
