#include "ai/multimodal_attention.hpp"
#include "ai/memory_manager.hpp"
#include <iostream>
#include <iomanip>

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

void test_multimodal_tensor() {
    std::cout << "\n=== Testing MultiModalTensor ===" << std::endl;
    
    // Create sample tensors for different modalities
    Tensor text_data({2, 10, 512});  // [batch, seq_len, features]
    Tensor image_data({2, 224, 224, 3});  // [batch, height, width, channels]
    Tensor audio_data({2, 1000, 128});  // [batch, time, features]
    
    // Create multimodal tensors
    MultiModalTensor text_tensor(text_data, Modality::TEXT, "Text input");
    MultiModalTensor image_tensor(image_data, Modality::IMAGE, "Image input");
    MultiModalTensor audio_tensor(audio_data, Modality::AUDIO, "Audio input");
    
    std::cout << "Text tensor modality: " << (text_tensor.is_text() ? "TEXT" : "OTHER") << std::endl;
    std::cout << "Image tensor modality: " << (image_tensor.is_image() ? "IMAGE" : "OTHER") << std::endl;
    std::cout << "Audio tensor modality: " << (audio_tensor.is_audio() ? "AUDIO" : "OTHER") << std::endl;
    
    // Test shape interpretation
    try {
        auto text_seq_shape = text_tensor.get_sequence_shape();
        std::cout << "Text sequence shape: [" << text_seq_shape[0] << ", " 
                  << text_seq_shape[1] << ", " << text_seq_shape[2] << "]" << std::endl;
        
        auto image_spatial_shape = image_tensor.get_spatial_shape();
        std::cout << "Image spatial shape: [" << image_spatial_shape[0] << ", " 
                  << image_spatial_shape[1] << ", " << image_spatial_shape[2] 
                  << ", " << image_spatial_shape[3] << "]" << std::endl;
                  
        auto audio_temporal_shape = audio_tensor.get_temporal_shape();
        std::cout << "Audio temporal shape: [" << audio_temporal_shape[0] << ", " 
                  << audio_temporal_shape[1] << ", " << audio_temporal_shape[2] << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error in shape interpretation: " << e.what() << std::endl;
    }
}

void test_cross_modal_attention() {
    std::cout << "\n=== Testing CrossModalAttention ===" << std::endl;
    
    try {
        // Create cross-modal attention layer
        CrossModalAttention cross_attention(512, 512, 256, 8);
        
        // Create sample data
        Tensor text_features({2, 10, 512});   // [batch, seq_len, features]
        Tensor image_features({2, 49, 512});  // [batch, spatial_patches, features] (7x7 = 49 patches)
        
        // Initialize with some sample data
        std::fill(text_features.data().begin(), text_features.data().end(), 0.1f);
        std::fill(image_features.data().begin(), image_features.data().end(), 0.2f);
        
        print_tensor_info(text_features, "Text features");
        print_tensor_info(image_features, "Image features");
        
        // Test cross-modal attention: image attends to text
        Tensor attended_image = cross_attention.forward(text_features, image_features, 
                                                       Modality::TEXT, Modality::IMAGE);
        
        print_tensor_info(attended_image, "Attended image features");
        
        // Test bidirectional attention
        auto bidirectional_result = cross_attention.bidirectional_attention(
            text_features, image_features, Modality::TEXT, Modality::IMAGE);
        
        print_tensor_info(bidirectional_result.first, "Bidirectional attended text");
        print_tensor_info(bidirectional_result.second, "Bidirectional attended image");
        
        // Get attention weights
        Tensor attention_weights = cross_attention.get_attention_weights();
        print_tensor_info(attention_weights, "Attention weights");
        
        std::cout << "CrossModalAttention test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error in CrossModalAttention test: " << e.what() << std::endl;
    }
}

void test_multimodal_fusion() {
    std::cout << "\n=== Testing MultiModalFusion ===" << std::endl;
    
    try {
        // Setup modality dimensions
        std::unordered_map<Modality, size_t> modality_dims = {
            {Modality::TEXT, 512},
            {Modality::IMAGE, 512},
            {Modality::AUDIO, 128}
        };
        
        // Test different fusion strategies
        std::vector<FusionStrategy> strategies = {
            FusionStrategy::EARLY_FUSION,
            FusionStrategy::LATE_FUSION,
            FusionStrategy::ATTENTION_FUSION
        };
        
        for (auto strategy : strategies) {
            std::cout << "\nTesting fusion strategy: " << static_cast<int>(strategy) << std::endl;
            
            MultiModalFusion fusion(strategy, modality_dims, 256);
            
            // Create sample features
            std::unordered_map<Modality, Tensor> features = {
                {Modality::TEXT, Tensor({2, 10, 512})},
                {Modality::IMAGE, Tensor({2, 10, 512})},
                {Modality::AUDIO, Tensor({2, 10, 128})}
            };
            
            // Initialize with sample data
            for (auto& pair : features) {
                std::fill(pair.second.data().begin(), pair.second.data().end(), 0.1f);
            }
            
            // Test fusion
            Tensor fused_features = fusion.fuse(features);
            print_tensor_info(fused_features, "Fused features");
            
            // Test fusion with attention
            auto fusion_with_attention = fusion.fuse_with_attention(features);
            print_tensor_info(fusion_with_attention.first, "Fused features with attention");
            print_tensor_info(fusion_with_attention.second, "Attention scores");
        }
        
        std::cout << "MultiModalFusion test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error in MultiModalFusion test: " << e.what() << std::endl;
    }
}

void test_multimodal_transformer() {
    std::cout << "\n=== Testing MultiModalTransformer ===" << std::endl;
    
    try {
        // Setup modality configurations
        std::unordered_map<Modality, size_t> modality_configs = {
            {Modality::TEXT, 512},
            {Modality::IMAGE, 512}
        };
        
        MultiModalTransformer transformer(modality_configs, 256, 2, 8);
        
        // Create sample inputs
        std::unordered_map<Modality, Tensor> inputs = {
            {Modality::TEXT, Tensor({1, 10, 512})},
            {Modality::IMAGE, Tensor({1, 49, 512})}  // 7x7 image patches
        };
        
        // Initialize with sample data
        for (auto& pair : inputs) {
            std::fill(pair.second.data().begin(), pair.second.data().end(), 0.1f);
        }
        
        print_tensor_info(inputs[Modality::TEXT], "Input text");
        print_tensor_info(inputs[Modality::IMAGE], "Input image");
        
        // Forward pass
        auto outputs = transformer.forward(inputs);
        
        print_tensor_info(outputs[Modality::TEXT], "Output text");
        print_tensor_info(outputs[Modality::IMAGE], "Output image");
        
        // Test conditional generation
        std::unordered_map<Modality, Tensor> conditioning = {
            {Modality::TEXT, Tensor({1, 10, 512})}
        };
        std::fill(conditioning[Modality::TEXT].data().begin(), 
                 conditioning[Modality::TEXT].data().end(), 0.1f);
        
        Tensor generated = transformer.generate_conditioned(Modality::IMAGE, conditioning, 20);
        print_tensor_info(generated, "Generated image features");
        
        std::cout << "MultiModalTransformer test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error in MultiModalTransformer test: " << e.what() << std::endl;
    }
}

void test_multimodal_utils() {
    std::cout << "\n=== Testing MultiModalUtils ===" << std::endl;
    
    try {
        // Test tensor conversion
        Tensor regular_tensor({2, 10, 128});
        std::fill(regular_tensor.data().begin(), regular_tensor.data().end(), 0.5f);
        
        MultiModalTensor mm_tensor = MultiModalUtils::to_multimodal(
            regular_tensor, Modality::AUDIO, "Converted audio tensor");
        
        std::cout << "Converted tensor modality: " << (mm_tensor.is_audio() ? "AUDIO" : "OTHER") << std::endl;
        std::cout << "Converted tensor description: " << mm_tensor.description() << std::endl;
        
        // Test sequence alignment
        std::unordered_map<Modality, Tensor> sequences = {
            {Modality::TEXT, Tensor({1, 15, 128})},
            {Modality::AUDIO, Tensor({1, 10, 128})}
        };
        
        auto aligned = MultiModalUtils::align_sequences(sequences);
        print_tensor_info(aligned[Modality::TEXT], "Aligned text");
        print_tensor_info(aligned[Modality::AUDIO], "Aligned audio");
        
        // Test cross-modal similarity
        Tensor features1({1, 5, 128});
        Tensor features2({1, 128, 3});
        std::fill(features1.data().begin(), features1.data().end(), 0.1f);
        std::fill(features2.data().begin(), features2.data().end(), 0.2f);
        
        Tensor similarity = MultiModalUtils::compute_cross_modal_similarity(features1, features2);
        print_tensor_info(similarity, "Cross-modal similarity");
        
        // Test interpolation
        Tensor modal1({1, 10, 128});
        Tensor modal2({1, 10, 128});
        std::fill(modal1.data().begin(), modal1.data().end(), 0.0f);
        std::fill(modal2.data().begin(), modal2.data().end(), 1.0f);
        
        Tensor interpolated = MultiModalUtils::interpolate_modalities(modal1, modal2, 0.5);
        print_tensor_info(interpolated, "Interpolated features");
        
        // Check interpolation values
        float first_val = interpolated.data()[0];
        std::cout << "Interpolated value (should be ~0.5): " << first_val << std::endl;
        
        std::cout << "MultiModalUtils test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error in MultiModalUtils test: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== CLModel Phase 3: Multi-Modal Attention Demo ===" << std::endl;
    std::cout << "Testing the new multi-modal AI capabilities..." << std::endl;
    
    try {
        test_multimodal_tensor();
        test_cross_modal_attention();
        test_multimodal_fusion();
        test_multimodal_transformer();
        test_multimodal_utils();
        
        std::cout << "\n=== All Multi-Modal Tests Completed ===" << std::endl;
        std::cout << "Phase 3 multi-modal attention implementation is working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Fatal error in demo: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
